import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
from dataclasses import dataclass
from copy import deepcopy
import pickle

import flax
import optax
import tyro
from flax import linen as nn
from flax.training.train_state import TrainState

import matplotlib.pyplot as plt
import seaborn as sns

from mppi import mppi

@dataclass
class Args:
    train_dynamics: bool = False
    """train a dynamics model"""
    dynamics_horizon: int = 2
    """prediction horizon to train the dynamics model on"""
    model_path: str = "model/dynamics.p"
    """location to save/load the dynamics network"""

def doubleintegrator2d(state, control):
    x,y,psi, vx,vy,r = state
    F,M = control
    ax = F * jnp.cos(psi)
    ay = F * jnp.sin(psi)
    cd = 0.05
    return jnp.array([
        vx,
        vy,
        r,
        ax - cd*vx,
        ay - cd*vy,
        M,
    ])

def wrap2pi(x):
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))

class Dynamics(nn.Module):
    state_dim: int

    @nn.compact
    def __call__(self, x, a):
        x = jnp.concat([x,a], axis=-1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.state_dim)(x)
        return x

@partial(jax.jit, static_argnames=['f', 'batch_size', 'epochs', 'horizon'])
def train_dynamics(f, dyn_state, batch_size, epochs, horizon, key):
    def step(carry, unused):
        dyn_state, key = carry

        def dynamics_loss(params, x, U, Y):
            """Minimize the N horizon prediction loss"""
            Y_hat = []
            for u in U:
                x = dyn.apply(params, x, u)
                Y_hat.append(x)
            Y_hat = jnp.concat(Y_hat)
            return ((Y_hat.flatten() - Y.flatten())**2).mean()

        key, _key = jax.random.split(key)
        x0 = jax.random.uniform(_key, (batch_size, 6), minval=-5, maxval=5)

        U, Y = [], []
        x = deepcopy(x0)
        for _ in range(horizon):
            key, _key = jax.random.split(key)
            u = jax.random.uniform(_key, (batch_size, 2), minval=-15, maxval=15)
            y = jax.vmap(f, in_axes=(0,0))(x,u)
            x = y
            
            key, _key = jax.random.split(key)
            U.append(deepcopy(u))
            Y.append(deepcopy(y) + jax.random.normal(_key, y.shape) * 1e-3)
            
        U = jnp.stack(U)
        Y = jnp.concat(Y)

        loss, grads = jax.value_and_grad(dynamics_loss)(dyn_state.params, x0,U,Y)
        dyn_state = dyn_state.apply_gradients(grads=grads)
        # jax.experimental.io_callback(lambda l : print(f"loss = {l:0.2e}"), None, loss)
        return (dyn_state, key), loss
    
    (dyn_state, key), losses = jax.lax.scan(step, (dyn_state, key), None, epochs)
    return dyn_state, losses


if __name__ == "__main__":
    args = tyro.cli(Args)

    key = jax.random.PRNGKey(time.time_ns())

    # Colorize output
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

    # Configure MPPI
    goal = np.array([0.,0.,0.,0.,0.,0.])
    kw = {
        'state_dim' : 6,
        'action_dim' : 2,
        'dt' : 1/50,
        'control_min' : jnp.array([-10.0, -10.0]),
        'control_max' : jnp.array([ 10.0,  10.0]),
        'T' : 100,
        'N' : 5_000,
        # Inverse temp : Higher values are more unweighted 
        'la' : 10,
        # Sigma : Control distribution covariance
        'sigma' : jnp.array([5.0, 10.0]),
        # Gamma : Control cost parameter
        'gamma' : 0.1,
    }

    # Train dynamcis
    dyn = Dynamics(kw['state_dim'])
    key, _key = jax.random.split(key)
    dyn_state = TrainState.create(
        apply_fn=dyn.apply,
        params=dyn.init(_key, jnp.zeros(kw['state_dim']), jnp.zeros(kw['action_dim'])),
        tx=optax.adam(learning_rate=3e-4)
    )

    # (Re)train the dynamics network
    if args.train_dynamics:
        start_time = time.time()
        key, _key = jax.random.split(key)
        dyn_state, losses = train_dynamics(doubleintegrator2d, dyn_state, 2048, 100_000, args.dynamics_horizon, _key)
        print(f"Dynamics final loss = {losses[-1]:0.1e} in {time.time() - start_time:0.0f}s")
        
        with open(args.model_path, "wb") as f:
            pickle.dump(dyn_state.params, f)
    
    # Load the dynamics network
    else:
        with open(args.model_path, "rb") as f:
            dyn_state = dyn_state.replace(params=pickle.load(f))

    # Precompile
    initial_control = jnp.zeros((kw['T'], kw['action_dim']))
    state = np.array([-2.,-1.,np.deg2rad(30),0,0,0])
    key, _key = jax.random.split(key)
    first_control, first_trajectory = mppi(dyn, dyn_state, state, goal, initial_control, _key, **kw)

    """MPPI"""
    start_time = time.time()
    initial_state = np.array([-2.,-1.,np.deg2rad(30),0,0,0])
    state = initial_state.copy()
    trajectory = []
    control = []
    initial_control = jnp.zeros((kw['T'], kw['action_dim']))
    for i in range(1000):
        # Run MPPI  
        key, _key = jax.random.split(key)
        optimal_control, optimal_trajectory = mppi(dyn, dyn_state, state, goal, initial_control, _key, **kw)
        initial_control = np.roll(initial_control.copy(), -1)
        initial_control[-1] = 0.0
        
        # Step dynamics
        state = deepcopy(state + kw['dt'] * doubleintegrator2d(state, optimal_control[0]))
        trajectory.append(state)
        control.append(optimal_control[0])

        # Termination criteria
        if np.linalg.norm(state[0:2] - goal[0:2]) < 0.05 and np.linalg.norm(state[3:5] - goal[3:5]) < 0.1:
            break
    
    else:
        print(f"Timed out at {i+1}")

    print(f"{i+1} iterations, {1e3*(time.time() - start_time)/(i+1):0.0f}ms per iter")
    optimal_trajectory = np.stack(trajectory)
    optimal_control = np.stack(control)

    """XY Plot"""
    fig, ax = plt.subplots(1,1, dpi=300)

    # # First trajectory
    # ax.plot(first_trajectory[:,0], first_trajectory[:,1], 'k--')

    # Closed-loop trajectory
    vel_norm = jnp.linalg.norm(optimal_trajectory[:,2:4], axis=-1)
    ax.scatter(optimal_trajectory[:,0], optimal_trajectory[:,1], c=vel_norm)
    # ax.quiver(
    #     optimal_trajectory[:,0], optimal_trajectory[:,1],
    #     optimal_trajectory[:,3], optimal_trajectory[:,4],
    #     vel_norm[:]
    # )

    # Goal & Obstacle
    ax.add_patch(plt.Circle(goal, 0.05, color='tab:orange', fill=False))
    ax.add_patch(plt.Circle((-1.0,-0.5), 0.2, color='k', fill=False))

    ax.axis("equal")
    plt.savefig("plots/xy.jpg")

    """Control Plot"""
    fig, ax = plt.subplots(3,3, figsize=(12,8), dpi=200)
    t = np.arange(len(optimal_trajectory)) * kw['dt']
    ax[0,0].plot(t, optimal_trajectory[:,0])
    ax[1,0].plot(t, optimal_trajectory[:,1])
    ax[2,0].plot(t, (optimal_trajectory[:,2]) * np.rad2deg(1))

    ax[0,0].set_ylabel("x (m)")
    ax[1,0].set_ylabel("y (m)")
    ax[2,0].set_ylabel("psi (deg)")
    ax[2,0].set_xlabel("t (s)")

    ax[0,1].plot(t, optimal_trajectory[:,3])
    ax[1,1].plot(t, optimal_trajectory[:,4])
    ax[2,1].plot(t, wrap2pi(optimal_trajectory[:,5]) * np.rad2deg(1))

    ax[0,1].set_ylabel("vx (m/s)")
    ax[1,1].set_ylabel("vy (m/s)")
    ax[2,1].set_ylabel("r (deg/s)")
    ax[2,1].set_xlabel("t (s)")
    
    ax[0,2].plot(t, optimal_control[:,0])
    ax[1,2].plot(t, wrap2pi(optimal_control[:,1]) * np.rad2deg(1))

    pos_error = jnp.linalg.norm(optimal_trajectory[:,0:2] - goal[0:2].reshape(1,-1), axis=-1)
    vel_error = jnp.linalg.norm(optimal_trajectory[:,3:5] - goal[3:5].reshape(1,-1), axis=-1)
    ax[2,2].plot(t, pos_error)
    ax[2,2].plot(t, vel_error)

    ax[0,2].set_ylabel("F (m/s2)")
    ax[1,2].set_ylabel("M (deg/s2)")
    ax[2,2].set_ylabel("Errors")
    ax[2,2].set_xlabel("t (s)")

    plt.savefig("plots/control.jpg")