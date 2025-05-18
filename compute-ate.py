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
from typing import Dict
import chex
from jax import Array
from jaxtyping import Integer, Float, Bool
from flax import struct
from sacred import Experiment
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import funcy as f
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ex = Experiment("compute-ate")


@ex.config
def config():
    n_cars = 300  # Number of cars
    # Pricing choice model parameters
    w_price = -0.3
    w_eta = -0.005
    w_intercept = 4
    n_events = 10000  # Number of events to simulate per trial
    batch_size = 100  # Number of environments to run in parallel
    k = 100  # Total number of trials
    chunk_size = 100  # Number of steps to process in each chunk
    output = "ate.csv"


def stepper(env, env_params, policy, carry, key):
    obs, state, total_reward = carry
    key, policy_key = jax.random.split(key)
    action, action_info = policy.apply(env_params, dict(), obs, policy_key)
    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)
    return (new_obs, new_state, total_reward + reward), None


def run(env, env_params, policy, key, n_steps):
    keys = jax.random.split(key, n_steps)
    obs, state = env.reset(key, env_params)
    final, _ = jax.lax.scan(
        partial(stepper, env, env_params, policy),
        (obs, state, 0),
        keys,
    )
    _, _, total_reward = final
    return total_reward / n_steps


def run_chunk(env, env_params, policy, keys, chunk_size, init_states=None):
    # Process a chunk of steps using vmap
    if init_states is None:
        # Initialize states for each batch
        obs_states = jax.vmap(lambda k: env.reset(k, env_params))(keys)
        # Add total_reward to the carry state
        carry_states = jax.vmap(lambda x: (x[0], x[1], 0.))(obs_states)
    else:
        # Add total_reward to the carry state from previous chunk
        carry_states = jax.vmap(lambda x: (x[0], x[1], 0.))(init_states)
    
    # Generate step keys for each batch
    step_keys = jax.vmap(lambda k: jax.random.split(k, chunk_size))(keys)
    
    # Vmap the scan operation over the batch dimension
    def scan_fn(carry, step_key):
        return jax.lax.scan(
            partial(stepper, env, env_params, policy),
            carry,
            step_key,
        )
    
    finals, _ = jax.vmap(scan_fn)(carry_states, step_keys)
    rewards = finals[2]  # Extract rewards from the final states
    return rewards, finals[:2]  # Return rewards and final states


def run_batch(env, env_params, A, B, key, n_steps, batch_size, chunk_size):
    # Split steps into chunks
    n_chunks = (n_steps + chunk_size - 1) // chunk_size
    
    # Generate keys for each batch
    batch_keys = jax.random.split(key, batch_size)
    
    # Process each batch in chunks
    total_rewards_A = jnp.zeros(batch_size)
    total_rewards_B = jnp.zeros(batch_size)
    states_A = None
    states_B = None
    
    for chunk_idx in tqdm(range(n_chunks)):
        # Get keys for this chunk
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_steps)
        chunk_size_actual = chunk_end - chunk_start
        
        # Process chunk for both policies
        chunk_rewards_A, states_A = run_chunk(env, env_params, A, batch_keys, chunk_size_actual, states_A)
        chunk_rewards_B, states_B = run_chunk(env, env_params, B, batch_keys, chunk_size_actual, states_B)
        
        # Accumulate rewards
        total_rewards_A += chunk_rewards_A
        total_rewards_B += chunk_rewards_B 

        total_rewards_A.block_until_ready()
        total_rewards_B.block_until_ready()

    
    # Normalize by total number of steps
    return {
        "A": total_rewards_A / n_steps,
        "B": total_rewards_B / n_steps,
    }


@ex.automain
def main(
    n_cars, w_price, w_eta, w_intercept, n_events, k, chunk_size, output, seed, batch_size, results_file=None
):
    logging.info("Starting ATE computation with parameters: n_cars=%d, n_events=%d, k=%d, batch_size=%d, chunk_size=%d", 
                n_cars, n_events, k, batch_size, chunk_size)
    
    if results_file:
        logging.info("Reading results from file: %s", results_file)
        # Read results from file
        results_df = pd.read_csv(results_file)
        logging.info("Saving results to: %s", output)
        results_df.to_csv(output, index=False)
        logging.info("ATE computation completed")
        return

    logging.info("Initializing environment and policies")
    env = ManhattanRidesharePricing(n_cars=n_cars, n_events=n_events)
    env_params = env.default_params
    env_params = env_params.replace(
        w_price=w_price, w_eta=w_eta, w_intercept=w_intercept
    )
    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.01)
    B = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.02)
    logging.info("Environment and policies initialized")

    n_batches = k // batch_size
    logging.info("Running %d batches with batch size %d", n_batches, batch_size)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_batches)
    results = [
        run_batch(env, env_params, A, B, key, n_events, batch_size, chunk_size)
        for key in keys
    ]
    logging.info("All batches completed, processing results")
    results_df = pd.concat(map(pd.DataFrame, results))
    logging.info("Saving results to: %s", output)
    results_df.to_csv(output, index=False)
    logging.info("ATE computation completed")
