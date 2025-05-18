import jax
from jax.experimental import sparse
from functools import partial
from picard.rideshare_dispatch import (
    ManhattanRideshareDispatch,
    ManhattanRidesharePricing,
    GreedyPolicy,
    SimplePricingPolicy,
    EnvParams,
    obs_to_state,
    RideshareEvent,
)
from picard.nn import Policy
from jax import numpy as jnp
from typing import Dict, Callable, Tuple
import chex
from jax import Array
from jaxtyping import Integer, Float, Bool
from flax import struct
from sacred import Experiment
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import funcy as f
import pandas as pd
import haversine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ex = Experiment("rideshares")


@ex.config
def config():
    n_cars = 300  # Number of cars
    # Pricing choice model parameters
    w_price = -0.3
    w_eta = -0.005
    w_intercept = 4
    n_events = 10000  # Number of events to simulate per trial
    k = 10  # Total number of trials
    batch_size = 100  # Number of environments to run in parallel
    switch_every = 1000  # Switchback duration
    p = 0.5  # Treatment probability
    output = "results.csv"
    config_output = "config.csv"
    max_km = 2
    lookahead_seconds = 600
    chunk_size = 1000  # Number of steps to process in each chunk


@struct.dataclass
class ExperimentInfo:
    """
    Contains treatment assignment and cluster information for each step
    """

    t: Integer[Array, "n_steps"]
    space_id: Integer[Array, "n_steps"]
    cluster_id: Integer[Array, "n_steps"]
    is_treat: Bool[Array, "n_steps"]
    key: chex.PRNGKey


@struct.dataclass
class EstimatorState:
    """
    Contains the current state of an estimator
    """

    counts: Integer[Array, "n_clusters"]
    estimates: Float[Array, "n_clusters"]


def naive_update(
    est: EstimatorState, reward: float, info: ExperimentInfo, p=0.5
):
    return EstimatorState(
        est.counts.at[info.cluster_id].add(1),
        est.estimates.at[info.cluster_id].add(reward),
    )


def naive(est: EstimatorState, z: Bool[Array, "n_clusters"], p=0.5):
    eta = z / p - (1 - z) / (1 - p)
    N = est.counts.sum()
    return (eta * est.estimates).sum() / N


def dq_update(
    time_ids: Integer[Array, "n_clusters"],
    space_ids: Integer[Array, "n_clusters"],
    space_adj: Bool[Array, "n_spaces n_spaces"],
    est: EstimatorState,
    reward: float,
    info: ExperimentInfo,
    p=0.5,
    lookahead_seconds=600,
    switch_every=1
):
    # Eta to be included later
    z = info.is_treat
    xi = z * (1 - p) / p + (1 - z) * p / (1 - p)
    update_val = xi * reward
    is_adjacent_t = (time_ids >= info.t - lookahead_seconds) & (
        time_ids <= (info.t // switch_every + 1) * switch_every
    )
    is_adjacent_space = space_adj[space_ids, info.space_id]
    # jax.debug.print("{eea}", eea=(is_adjacent_space & is_adjacent_t).sum())
    update_ests = est.estimates + jnp.where(
        is_adjacent_t & is_adjacent_space,
        update_val,
        0.0,
    )
    update_ests = update_ests.at[info.cluster_id].add(reward - update_val)
    return EstimatorState(est.counts.at[info.cluster_id].add(1), update_ests)


def dq(est: EstimatorState, z: Bool[Array, "n_clusters"], p=0.5):
    N = est.counts.sum()
    mask = est.counts > 0
    eta = z / p - (1 - z) / (1 - p)
    avg_y = (mask * est.estimates).sum() / est.counts.sum()
    baseline = est.counts * avg_y
    return (mask * eta * (est.estimates - baseline)).sum() / N


estimator_fns = {
    "naive": naive,
    "dq": dq,
}


def stepper(
    estimators: Dict[str, Callable],
    env,
    env_params,
    A: Policy,
    B: Policy,
    carry: Tuple[Array, Array, Dict[str, EstimatorState]],
    info: ExperimentInfo,
):
    obs, state, ests = carry
    key, policy_key = jax.random.split(info.key)
    action, action_info = jax.lax.cond(
        info.is_treat,
        lambda: B.apply(env_params, dict(), obs, policy_key),
        lambda: A.apply(env_params, dict(), obs, policy_key),
    )

    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)

    new_ests = {
        est_name: est_update(ests[est_name], reward, info)
        for est_name, est_update in estimators.items()
    }

    return ((new_obs, new_state, new_ests), None)


def run_trials(
    env,
    env_params,
    A,
    B,
    key,
    n_envs=10,
    n_steps=1000,
    switch_every=1,
    p=0.5,
    spatial_clusters=None,
    space_adj=None,
    lookahead_seconds=600,
    chunk_size=1000,
):
    time_ids = (
        env_params.events.t // switch_every + 1
    ) * switch_every  # Identifies the end of the period
    space_ids = spatial_clusters.loc[env_params.events.src]["zone_id"].values
    ab_key, key = jax.random.split(key)
    unq_times, unq_time_ids = jnp.unique(time_ids, return_inverse=True)
    # unq_spaces, unq_space_ids = jnp.unique(space_ids, return_inverse=True)
    unq_spaces = jnp.arange(spatial_clusters["zone_id"].max() + 1)
    cluster_ids = unq_time_ids * len(unq_spaces) + space_ids
    n_clusters = len(unq_times) * len(unq_spaces)
    cluster_treats = jax.random.bernoulli(ab_key, p, (n_envs, n_clusters))
    is_treat = cluster_treats[:, cluster_ids]

    reset_key, step_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_envs)
    step_keys = jax.random.split(step_key, (n_envs, n_steps))

    estimators = {
        "naive": partial(naive_update, p=p),
        "dq": partial(
            dq_update,
            jnp.repeat(unq_times, len(unq_spaces)),
            jnp.tile(unq_spaces, len(unq_times)),
            space_adj,
            p=p,
            lookahead_seconds=lookahead_seconds,
            switch_every=switch_every,
        ),
    }

    # Initial inputs
    init_ests = {
        est_name: EstimatorState(
            jnp.zeros((n_envs, n_clusters)), jnp.zeros((n_envs, n_clusters))
        )
        for est_name in estimators.keys()
    }

    infos = ExperimentInfo(
        t=jnp.tile(env_params.events.t.reshape(1, -1), (n_envs, 1)),
        space_id=jnp.tile(space_ids.reshape(1, -1), (n_envs, 1)),
        cluster_id=jnp.tile(cluster_ids.reshape(1, -1), (n_envs, 1)),
        is_treat=is_treat,
        key=step_keys,
    )

    def scanner(obs_and_states_and_ests_initial_batched, global_batched_infos):
        
        # This function is vmapped. It processes one batch element over one chunk of steps.
        def scan_fn_for_one_batch_element_one_chunk(carry_one_batch_element, infos_one_batch_element_one_chunk):
            # carry_one_batch_element is (obs_i, state_i, ests_i_dict) for the i-th batch element.
            # infos_one_batch_element_one_chunk is an ExperimentInfo instance where each field
            # (e.g., .t, .key) has shape (current_chunk_length,) for the i-th batch element.
            
            # jax.lax.scan iterates over the leading axis of infos_one_batch_element_one_chunk.
            # stepper is partially applied with (estimators, env, env_params, A, B) from the outer scope.
            # stepper expects: carry=(obs, state, ests_dict_for_one_batch), info=ExperimentInfo_for_one_step.
            final_carry_one_batch_element, _ = jax.lax.scan(
                partial(stepper, estimators, env, env_params, A, B),
                carry_one_batch_element,
                infos_one_batch_element_one_chunk, # This is 'xs' for the inner scan.
            )
            # stepper returns (new_carry, None), so final_carry_one_batch_element is (obs_final, state_final, ests_dict_final)
            return final_carry_one_batch_element

        # Vmap scan_fn_for_one_batch_element_one_chunk to run in parallel for all batch elements.
        vmapped_chunk_processor = jax.vmap(
            scan_fn_for_one_batch_element_one_chunk,
            in_axes=(0, 0), # Process 0-th axis of carry_batched and 0-th axis of infos_batched_for_chunk.
            out_axes=0      # The output (final_carry_one_batch_element) is also batched along axis 0.
        )

        current_full_carry_batched = obs_and_states_and_ests_initial_batched

        n_steps = global_batched_infos.t.shape[1]
        # chunk_size is from the outer scope (run_trials parameter)
        n_chunks = (n_steps + chunk_size - 1) // chunk_size

        for chunk_idx in tqdm(range(n_chunks)):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_steps)

            # Slice global_batched_infos to get data for the current chunk across all batch elements.
            # Each field in current_chunk_infos_batched will have shape (batch_size, current_chunk_length).
            current_chunk_infos_batched = ExperimentInfo(
                t=global_batched_infos.t[:, chunk_start:chunk_end],
                space_id=global_batched_infos.space_id[:, chunk_start:chunk_end],
                cluster_id=global_batched_infos.cluster_id[:, chunk_start:chunk_end],
                is_treat=global_batched_infos.is_treat[:, chunk_start:chunk_end],
                key=global_batched_infos.key[:, chunk_start:chunk_end],
            )
            
            # Apply the vmapped function to process the current chunk for all batch elements.
            current_full_carry_batched = vmapped_chunk_processor(
                current_full_carry_batched,
                current_chunk_infos_batched
            )
            current_full_carry_batched[0].block_until_ready()
            # Optional: Add current_full_carry_batched.block_until_ready() here if needed for debugging,
            # especially for memory profiling or ensuring sequential execution visibility.

        # The final carry contains (final_obs_batched, final_state_batched, final_ests_dict_batched).
        # We need to return the final_ests_dict_batched part.
        _, _, final_ests_dict_batched = current_full_carry_batched
        return final_ests_dict_batched

    obs_and_states = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, env_params
    )
    obs_and_states_and_ests = (*obs_and_states, init_ests)
    estimator_results = scanner(obs_and_states_and_ests, infos)
    return {
        est_name: jax.vmap(estimator_fns[est_name], in_axes=(0, 0, None))(
            est_state, cluster_treats, p
        )
        for est_name, est_state in estimator_results.items()
    }


def load_spatial_clusters():
    zones = pd.read_parquet("taxi-zones.parquet")
    unq_zones, unq_zone_ids = np.unique(zones["zone"], return_inverse=True)
    zones["zone_id"] = unq_zone_ids
    nodes = pd.read_parquet("manhattan-nodes.parquet")
    nodes["lng"] = nodes["lng"].astype(float)
    nodes["lat"] = nodes["lat"].astype(float)
    nodes_zones = nodes.merge(zones, on="osmid")

    centroids = nodes_zones.groupby("zone_id").aggregate(
        {"lat": "mean", "lng": "mean"}
    )
    dist = np.zeros((len(centroids), len(centroids)))
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            dist[i, j] = haversine.haversine(
                (centroids.iloc[i]["lat"], centroids.iloc[i]["lng"]),
                (centroids.iloc[j]["lat"], centroids.iloc[j]["lng"]),
            )
    return nodes_zones, dist


@ex.automain
def main(
    n_cars,
    w_price,
    w_eta,
    w_intercept,
    n_events,
    seed,
    k,
    batch_size,
    switch_every,
    p,
    output,
    _config,
    max_km,
    lookahead_seconds,
    chunk_size,
):
    logging.info("Starting simulation with parameters: %s", _config)
    key = jax.random.PRNGKey(seed)
    env = ManhattanRidesharePricing(n_cars=n_cars, n_events=n_events)
    env_params = env.default_params
    env_params = env_params.replace(
        w_price=w_price, w_eta=w_eta, w_intercept=w_intercept
    )

    nodes_zones, zone_dists = load_spatial_clusters()
    logging.info("Loaded spatial clusters and distances.")

    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.01)
    B = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.02)
    logging.info("Initialized pricing policies.")

    print(
        "Simulation time (mins)",
        (env_params.events.t.max() - env_params.events.t[5]) / 60,
    )
    print(
        "Simulation time (hrs)",
        (env_params.events.t.max() - env_params.events.t[5]) / 3600,
    )

    unq_times = jnp.unique(env_params.events.t // switch_every)
    n_times = unq_times.shape[0]
    n_spaces = zone_dists.shape[0]
    n_clusters = n_times * n_spaces
    logging.info("Calculated number of clusters: %d", n_clusters)

    all_results = []
    keys = jax.random.split(key, k // batch_size + 1)
    for i, key in enumerate(keys):
        logging.info("Starting batch %d/%d", i + 1, len(keys))
        ests = run_trials(
            env,
            env_params,
            A,
            B,
            key,
            n_envs=batch_size,
            n_steps=n_events,
            switch_every=switch_every,
            p=p,
            spatial_clusters=nodes_zones,
            space_adj=jnp.asarray(zone_dists < max_km),
            lookahead_seconds=lookahead_seconds,
            chunk_size=chunk_size,
        )
        all_results.append(ests)
        logging.info("Completed batch %d/%d", i + 1, len(keys))

    pd.DataFrame.from_dict([_config]).to_csv(
        _config["config_output"], index=False
    )
    results_df = pd.concat(map(pd.DataFrame, all_results))
    results_df.to_csv(output, index=False)
    logging.info("Simulation completed.")
