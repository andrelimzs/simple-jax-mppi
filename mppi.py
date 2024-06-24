import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from functools import partial

import flax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState


def wrap2pi(x):
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))

@partial(jax.jit, static_argnames=['dyn', 'state_dim', 'action_dim', 'dt', 'T', 'N', 'la', 'gamma'])
def mppi(
    dyn, dyn_state,
    initial_state, goal, initial_control,
    key, state_dim, action_dim,
    dt, control_min, control_max,
    T, N, la, sigma, gamma,
):
    """
    Arguments
        dyn : Dynamics network
        dyn_state : Dynamics network state
        initial_state : State to start planning from
        goal : State to attempt to reach
        initial_control : Unexecuted portion of previous plan
        key : JAX random key
    
    Parameters
        dt : Timestep
        control_min :
        control_max :  
        T : Length of rollouts
        N : Number of rollouts/samples
        la : Inverse temperature
            Controls how tightly peaked the optimal distribution is
            Higher values are closer to an unweighted average
        sigma : covariance of low-level controller
        gamma : Control cost parameter
        alpha : Parameter that tradeoffs base distribution between 
            uncontrolled dynamics and previous control sequence.
            0 corresponds to base distribution
            1 corresponds to previous control sequence

    Returns
        optimal_control : Computed from the information theoretic weighted average of rollouts
        optimal_trajectory : Trajectory predicted from dynamics and optimal control
    """
    # Control cost parameter
    # gamma = la * (1 - alpha)
    alpha = 1 - gamma / la 

    def dynamics_step(state, control):
        """Predict future state x_t+1 from current state x_t and control u_t"""
        # new_state = state + dt * doubleintegrator2d(state, control)
        new_state = state + dt * dyn.apply(dyn_state.params, state, control)
        return new_state, new_state
    
    @partial(jax.jit, static_argnames="distribution")
    def compute_cost(trajectory, initial_control, control_noise, distribution):
        """Compute running and terminal cost"""
        if distribution == "uncontrolled":
            v = initial_control + control_noise
        elif distribution == "previous":
            v = control_noise

        position = trajectory[:,0:2]
        velocity = trajectory[:,3:5]
        psi = trajectory[:,2:3]
        rate = trajectory[:,5:6]
        
        pos_error = position - goal[0:2].reshape(1,-1)
        vel_error = velocity - goal[3:5].reshape(1,-1)
        
        target_heading = jnp.arctan2(pos_error[:,1:2], pos_error[:,0:1])
        heading_error = wrap2pi(psi - target_heading)

        position_cost = jnp.linalg.norm(pos_error, axis=-1)
        velocity_cost = jnp.linalg.norm(vel_error, axis=-1)
        heading_cost = jnp.linalg.norm(heading_error, axis=-1)

        obs_pos = jnp.array([-1,-0.5]).reshape(1,-1)
        dist_to_obstacle = jnp.linalg.norm(position - obs_pos, axis=-1) - 0.2
        obstacle_cost = (dist_to_obstacle <= 0)

        time_cost = 1e-2 * ((pos_error > 1e-1) & (vel_error > 1e-2))

        running_cost = (
            + 2 * (position_cost[:-1])
            + 1 * (velocity_cost[:-1])
            + 1e-1 * heading_cost[:-1]
            + 1e6 * obstacle_cost[:-1]
            # + time_cost[:-1]
            + gamma * jnp.vecdot(initial_control[:-1], v[1:] / sigma.reshape(1,-1), axis=-1)
        ) * ((position_cost > 0.05) | (velocity_cost > 0.1))[:-1]
        terminal_cost = 10 * (
            + 2 * position_cost[-1]
            + velocity_cost[-1] * jnp.exp(-position_cost[-1]**2)
            + 1e6 * obstacle_cost[-1]
        )

        return terminal_cost + jnp.sum(running_cost)

    def rollout(state, initial_control, T, distribution, key):
        """Perform a rollout of length T"""
        # Generate and clip control perturbations
        control_perturbations = sigma*jax.random.normal(key, (T,action_dim))
        control = jax.vmap(jnp.clip,(0,None,None))(initial_control+control_perturbations, control_min, control_max)
        # Simulate trajectory and cost under perturbations
        _, trajectory = jax.lax.scan(dynamics_step, state, control)
        rollout_cost = compute_cost(trajectory, initial_control, control_perturbations, distribution)
        return trajectory, control_perturbations, rollout_cost

    def compute_optimal(initial_control, control_perturbations, rollout_cost):
        """Compute the optimal control from MPPI weighted average of rollout costs"""
        rho = rollout_cost.min()
        eta = jnp.sum(jnp.exp(-(rollout_cost - rho)/la))
        weights = jnp.exp(-(rollout_cost - rho)/la) / eta
        optimal_control = initial_control + (weights.reshape(-1,1,1) * control_perturbations).sum(axis=0)
        return jax.vmap(jnp.clip, (0,None,None))(optimal_control,control_min,control_max)

    N1 = round((1 - alpha) * N)
    N2 = N - N1

    # Rollouts with uncontrolled distribution
    vrollout = jax.vmap(rollout, in_axes=(None,None,None,None,0))
    if N1 > 0:
        key, *_key = jax.random.split(key, N1+1)
        _key = jnp.stack(_key)
        trajectory1, control_perturbations1, rollout_cost1 = vrollout(
            initial_state, initial_control, T, "uncontrolled", _key)
    
    # Rollouts with previous distribution
    if N2 > 0:
        key, *_key = jax.random.split(key, N2+1)
        _key = jnp.stack(_key)
        trajectory2, control_perturbations2, rollout_cost2 = vrollout(
            initial_state, initial_control, T, "previous", _key)

    # Combine rollouts from both distributions
    if N1 > 0 and N2 > 0:
        trajectory = jnp.concat([trajectory1, trajectory2])
        control_perturbations = jnp.concat([control_perturbations1, control_perturbations2])
        rollout_cost = jnp.concat([rollout_cost1, rollout_cost2])
    elif N1 > 0 and N2 == 0:
        trajectory = trajectory1
        control_perturbations = control_perturbations1
        rollout_cost = rollout_cost1
    elif N1 == 0 and N2 > 0:
        trajectory = trajectory2
        control_perturbations = control_perturbations2
        rollout_cost = rollout_cost2
    
    # Compute optimal control from reward weighted control perturbations
    optimal_control = compute_optimal(initial_control, control_perturbations, rollout_cost)

    # Compute optimal trajectory
    _, optimal_trajectory = jax.lax.scan(dynamics_step, initial_state, optimal_control)
    
    return optimal_control, optimal_trajectory, # trajectory
