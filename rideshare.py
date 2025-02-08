import jax
from functools import partial
from picard.environments.rideshare_dispatch import (
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
from typing import Dict
import chex
from jax import Array
from jaxtyping import Integer, Float, Bool
from flax import struct
from sacred import Experiment
from tqdm import tqdm, trange
import pandas as pd
import numpy as np

ex = Experiment("rideshares")


@ex.config
def config():
    n_cars = 300  # Number of cars
    # Pricing choice model parameters
    w_price = -0.3
    w_eta = -0.005
    w_intercept = 4
    n_events = 10000  # Number of events to simulate per trial
    k = 1000  # Total number of trials
    batch_size = 100  # Number of environments to run in parallel
    switch_every = 1000  # Switchback duration
    p = 0.5  # Treatment probability
    max_km = 2.0  # Max distance between spatial clusters to be considered "adjacent" for DQ

    output = "results.csv"
    config_output = "config.csv"


def lead(x, k=1, axis=-1, fill=0):
    result = jnp.roll(x, shift=-k, axis=axis)
    slice_obj = [slice(None)] * x.ndim
    slice_obj[axis] = slice(-k, None)
    result = result.at[tuple(slice_obj)].set(fill)
    return result


def naive(
    y: Float[Array, "n"],
    z: Bool[Array, "n"],
    p: Float[Array, "n"],
    mask: Bool[Array, "n"],
    baseline: float = 10,
) -> Float[Array, "1"]:
    y1 = y * z / p
    y0 = y * (1 - z) / (1 - p)
    return ((y1 - y0) * mask).sum()


def dq(
    y: Float[Array, "n"],
    z: Bool[Array, "n"],
    p: Float[Array, "n"],
    mask: Bool[Array, "n"],
    A: Bool[Array, "n n"],
    baseline: float = 10,
) -> Float[Array, "1"]:
    y1 = y * z / p
    y0 = y * (1 - z) / (1 - p)
    eta = z / p - (1 - z) / (1 - p)
    xi = z * (1 - p) / p + (1 - z) * p / (1 - p)
    affected_y = (A - jnp.eye(*A.shape)) @ (xi * y)
    return (mask * eta * (y + affected_y - affected_y.mean())).sum()


def dq_grad(
    y: Float[Array, "n"],
    z: Bool[Array, "n"],
    p: Float[Array, "n"],
    mask: Bool[Array, "n"],
    A: Bool[Array, "n n"],
    baseline: float = 10,
) -> Float[Array, "1"]:
    y1 = y * z / p
    y0 = y * (1 - z) / (1 - p)
    eta = z / p - (1 - z) / (1 - p)
    affected_y = (A - jnp.eye(*A.shape)) @ y
    return (mask * eta * (y + affected_y - affected_y.mean())).sum()


def stepper(env, env_params, A, B, obs_and_state, key_and_treat):
    key, is_treat = key_and_treat
    obs, state = obs_and_state
    key, policy_key = jax.random.split(key)
    action, action_info = jax.lax.cond(
        is_treat,
        lambda: B.apply(env_params, dict(), obs, policy_key),
        lambda: A.apply(env_params, dict(), obs, policy_key),
    )
    new_obs, new_state, reward, _, info = env.step(
        key, state, action, env_params
    )
    action_info = {
        "is_B": is_treat,
    }
    return ((new_obs, new_state), (obs, action, reward, info, action_info))


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
):
    time_ids = env_params.events.t // switch_every
    space_ids = spatial_clusters.loc[env_params.events.src]["zone_id"].values
    ab_key, key = jax.random.split(key)
    unq_times, unq_time_ids = jnp.unique(time_ids, return_inverse=True)
    # unq_spaces, unq_space_ids = jnp.unique(space_ids, return_inverse=True)
    unq_spaces = jnp.arange(spatial_clusters["zone_id"].max() + 1)
    cluster_ids = unq_time_ids * len(unq_spaces) + space_ids
    is_treat = jax.random.bernoulli(
        ab_key, p, (n_envs, len(unq_times) * len(unq_spaces))
    )[:, cluster_ids]

    reset_key, step_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_envs)
    step_keys = jax.random.split(step_key, (n_envs, n_steps))

    def scanner(obs_and_state, keys, treats):
        return jax.lax.scan(
            partial(stepper, env, env_params, A, B),
            obs_and_state,
            (keys, treats),
        )[1]

    vmap_scan = jax.vmap(scanner, in_axes=(0, 0, 0))
    o, a, r, i, ai = vmap_scan(
        jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params),
        step_keys,
        is_treat,
    )
    return {
        "o": o,
        "a": a,
        "z": is_treat,
        "cluster_ids": cluster_ids,
        "time_ids": time_ids,
        "space_ids": space_ids,
        "r": r,
        "i": i,
        "ai": ai,
    }


def single_stepper(env, env_params, policy, obs_and_state, key):
    obs, state = obs_and_state
    key, policy_key = jax.random.split(key)
    action, action_info = policy.apply(env_params, dict(), obs, policy_key)
    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)
    return (new_obs, new_state), (obs, action, reward, action_info)


def load_spatial_clusters(max_km):
    # scratch
    import pandas as pd
    import haversine

    zones = pd.read_parquet("~/Downloads/taxi-zones.parquet")
    unq_zones, unq_zone_ids = np.unique(zones["zone"], return_inverse=True)
    zones["zone_id"] = unq_zone_ids
    nodes = pd.read_parquet("../ridesharing-gpu/manhattan-nodes.parquet")
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
    return nodes_zones, dist < max_km


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
    max_km,
    output,
    _config,
):
    key = jax.random.PRNGKey(seed)
    env = ManhattanRidesharePricing(n_cars=n_cars, n_events=n_events)
    env_params = env.default_params
    env_params = env_params.replace(
        w_price=w_price, w_eta=w_eta, w_intercept=w_intercept
    )

    nodes_zones, spatial_adj = load_spatial_clusters(max_km)

    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.01)
    B = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.02)
    keys = jax.random.split(key, 10000)
    _, results_A = jax.lax.scan(
        partial(single_stepper, env, env_params, A),
        env.reset(key, env_params),
        keys,
    )

    _, results_B = jax.lax.scan(
        partial(single_stepper, env, env_params, B),
        env.reset(key, env_params),
        keys,
    )

    ate = results_B[2].mean() - results_A[2].mean()
    print("ATE=", ate)
    print(
        "Simulation time (mins)",
        (env_params.events.t.max() - env_params.events.t[5]) / 60,
    )
    print(
        "Simulation time (hrs)",
        (env_params.events.t.max() - env_params.events.t[5]) / 3600,
    )

    def Ak(n, k):
        """
        A time adjacency matrix with dependencies k steps into the future
        """
        return jnp.triu(jnp.ones((n, n))) - jnp.triu(jnp.ones((n, n)), k=1 + k)

    unq_times = jnp.unique(env_params.events.t // switch_every)
    n_times = unq_times.shape[0]
    n_spaces = spatial_adj.shape[0]
    n_clusters = n_times * n_spaces

    def Adjk(n, k):
        """
        A space-time adjacency matrix with dependencies k steps into the future
        """
        return jnp.kron(Ak(n, k), spatial_adj)

    def estimate_clusters(y, z, c, p):
        # Not all clusters enter the experiment; this identifes those that do
        mask = (jnp.zeros(n_clusters).at[c].add(1)) > 0
        y_by_c = jnp.zeros(n_clusters).at[c].add(y)
        z_by_c = (jnp.zeros(n_clusters).at[c].add(z)) > 0
        dq_lookaheads = [
            (look, np.ceil(look * 60 / switch_every))
            for look in [0, 5, 10, 30, 60, 120]
        ]
        print(dq_lookaheads)

        return {
            "mean": y.mean(),
            "naive": naive(y_by_c, z_by_c, p, mask) / y.shape[0],
            "dq": dq(
                y_by_c, z_by_c, p, mask, Adjk(n_times, n_times), baseline=0
            )
            / y.shape[0],
            **{
                f"dq_{minutes:03d}m": dq(
                    y_by_c, z_by_c, p, mask, Adjk(n_times, periods)
                ) / y.shape[0]
                for minutes, periods in dq_lookaheads
            },
        }

    all_results = []
    for i in trange(k // batch_size + 1):
        results = run_trials(
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
        )

        res = jax.vmap(estimate_clusters, in_axes=(0, 0, None, None))(
            results["r"],
            results["z"],
            results["cluster_ids"],
            0.5,
        )

        all_results.append(res)

    pd.DataFrame.from_dict([_config]).to_csv(
        _config["config_output"], index=False
    )
    results_df = pd.concat(map(pd.DataFrame, all_results))
    results_df["ATE"] = ate
    results_df.to_csv(output, index=False)
