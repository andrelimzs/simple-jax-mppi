import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
from copy import deepcopy

import flax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

import matplotlib.pyplot as plt
import seaborn as sns

from mppi import mppi, Dynamics

def doubleintegrator2d(state, control):
    x,y,vx,vy = state
    ax,ay = control
    cd = 0.05
    return jnp.array([vx, vy, ax-cd*vx, ay-cd*vy])

@partial(jax.jit, static_argnames=['f', 'epochs', 'horizon'])
def train_dynamics(f, dyn_state, epochs, horizon, key):
    def step(carry, unused):
        dyn_state, key = carry

        def l2_loss(x, alpha):
            return alpha * (x**2).mean()

        def dynamics_loss(params, x, U, Y):
            """Minimize the N horizon prediction loss"""
            Y_hat = []
            for u in U:
                x = dyn.apply(params, x, u)
                Y_hat.append(x)
            Y_hat = jnp.concat(Y_hat)
            # L2_regularization = sum(
            #     l2_loss(w, alpha=1e5) for w in jax.tree.leaves(params)
            # )
            return (jnp.linalg.norm(Y_hat - Y)**2).mean() / horizon # + L2_regularization

        key, _key = jax.random.split(key)
        x0 = jax.random.uniform(_key, (batch_size, 4), minval=-2, maxval=2)

        U, Y = [], []
        x = deepcopy(x0)
        for _ in range(horizon):
            key, _key = jax.random.split(key)
            u = jax.random.uniform(_key, (batch_size, 2))
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

    batch_size = 256
    (dyn_state, key), losses = jax.lax.scan(step, (dyn_state, key), None, epochs)
        
    return dyn_state, losses


if __name__ == "__main__":
    key = jax.random.PRNGKey(time.time_ns())

    # Colorize output
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

    # Train dynamcis
    dyn = Dynamics()
    key, _key = jax.random.split(key)
    dyn_state = TrainState.create(
        apply_fn=dyn.apply,
        params=dyn.init(_key, jnp.zeros(4), jnp.zeros(2)),
        tx=optax.adam(learning_rate=3e-4)
    )

    key, _key = jax.random.split(key)
    horizon = 10
    dyn_state, losses = train_dynamics(doubleintegrator2d, dyn_state, 10_000, horizon, _key)
    print(f"Dynamics final loss = {losses[-1]:0.1e}")

    # Configure MPPI
    goal = np.array([0.,0.,0.,0.])
    kw = {
        'state_dim' : 4,
        'action_dim' : 2,
        'dt' : 0.1,
        'control_min' : jnp.array([-1.0, -1.0]),
        'control_max' : jnp.array([ 1.0,  1.0]),
        'T' : 200,
        'N' : 5000,
        'la' : 1.0,
        'sigma' : jnp.array([0.05, 0.05]),
        'alpha' : 0.99, # gamma = la * (1 - alpha)
    }

    # Precompile
    initial_control = jnp.zeros((kw['T'], kw['action_dim']))
    state = np.array([-2.,-1.,0.,0.])
    key, _key = jax.random.split(key)
    first_control, first_trajectory = mppi(dyn, dyn_state, state, goal, initial_control, _key, **kw)

    """MPPI"""
    start_time = time.time()
    initial_state = np.array([-2.,-1.,0.,0.])
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
        if np.linalg.norm(state[0:2] - goal[0:2]) < 0.05 and np.linalg.norm(state[2:4] - goal[2:4]) < 1e-2:
            break
    else:
        print(f"Timed out at {i+1}")

    print(f"{i+1} iterations, {1e3*(time.time() - start_time)/(i+1):0.0f}ms per iter")
    optimal_trajectory = np.stack(trajectory)
    optimal_control = np.stack(control)

    """XY Plot"""
    fig, ax = plt.subplots(1,1, dpi=300)

    # Closed-loop trajectory
    vel_norm = jnp.linalg.norm(optimal_trajectory[:,2:4], axis=-1)
    ax.scatter(optimal_trajectory[:,0], optimal_trajectory[:,1], c=vel_norm)

    # Goal & Obstacle
    ax.add_patch(plt.Circle(goal, 0.05, color='tab:orange', fill=False))
    ax.add_patch(plt.Circle((-1.0,-0.5), 0.2, color='k', fill=False))

    ax.axis("equal")
    plt.savefig("xy.jpg")

    """Control Plot"""
    fig, ax = plt.subplots(3,2, figsize=(12,8), dpi=200)
    t = np.arange(len(optimal_trajectory)) * kw['dt']
    ax[0,0].plot(t, optimal_trajectory[:,0])
    ax[1,0].plot(t, optimal_trajectory[:,2])
    ax[2,0].plot(t, optimal_control[:,0])

    ax[0,0].set_ylabel("x (m)")
    ax[1,0].set_ylabel("vx (m/s)")
    ax[2,0].set_ylabel("ax (m/s)")
    ax[2,0].set_xlabel("t (s)")

    ax[0,1].plot(t, optimal_trajectory[:,1])
    ax[1,1].plot(t, optimal_trajectory[:,3])
    ax[2,1].plot(t, optimal_control[:,1])

    ax[0,1].set_ylabel("y (m)")
    ax[1,1].set_ylabel("vy (m/s)")
    ax[2,1].set_ylabel("ay (m/s)")
    ax[2,1].set_xlabel("t (s)")

    plt.savefig("plot.jpg")