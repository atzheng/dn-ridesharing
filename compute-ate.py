import jax
from jax.experimental import sparse
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
import funcy as f
import json

ex = Experiment("compute-ate")

@ex.config
def config():
    n_cars = 300  # Number of cars
    # Pricing choice model parameters
    w_price = -0.3
    w_eta = -0.005
    w_intercept = 4
    n_events = 10000  # Number of events to simulate per trial
    k = 100  # Total number of trials
    output = "ate.json"


def stepper(env, env_params, policy, obs_and_state, key):
    obs, state = obs_and_state
    key, policy_key = jax.random.split(key)
    action, action_info = policy.apply(env_params, dict(), obs, policy_key)
    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)
    return (new_obs, new_state), reward


def run(env, env_params, policy, key, n_steps):
    keys = jax.random.split(key, n_steps)
    obs, state = env.reset(key, env_params)
    _, results = jax.lax.scan(
        partial(stepper, env, env_params, policy),
        (obs, state),
        keys,
    )
    return results

vmap_run = jax.vmap(run, in_axes=(None, None, None, 0, None))

@ex.automain
def main(n_cars, w_price, w_eta, w_intercept, n_events, k, output, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), k)
    env = ManhattanRidesharePricing(n_cars=n_cars, n_events=n_events)
    env_params = env.default_params
    env_params = env_params.replace(
        w_price=w_price, w_eta=w_eta, w_intercept=w_intercept
    )
    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.01)
    B = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.02)

    results_A = vmap_run(env, env_params, A, keys, n_events)
    results_B = vmap_run(env, env_params, B, keys, n_events)
    mean_A = results_A.mean()
    mean_B = results_B.mean()

    out_dict = f.walk_values(float, {
        "mean_A": mean_A,
        "mean_A_sd": results_A.std(),
        "mean_B": mean_B,
        "mean_B_sd": results_B.std(),
        "ate": mean_B - mean_A,
        "ate_se": (results_B - results_A).std() / np.sqrt(k),
    })

    json.dump(out_dict, open(output, "w"))
