import jax
import jax.numpy as jnp
import immrax as irx
from testtt.utilities import sim_constants # Import simulation constants


# Some configurations
jax.config.update("jax_enable_x64", True)
def jit (*args, **kwargs): # A simple wrapper for JAX's jit function to set the backend device
    device = 'cpu'
    kwargs.setdefault('backend', device)
    return jax.jit(*args, **kwargs)


class ThreeDMultirotorTransformed(irx.System):
    def __init__(self):
        self.xlen = 9
        self.evolution = 'continuous'
        self.G = sim_constants.GRAVITY  # gravitational acceleration in m/s^2
        self.M = sim_constants.MASS  # mass of the multirotor in kg
        self.C = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])



    def f(self, t, state, input, w):
        """Quadrotor dynamics. xdot = f(x, u, w)."""
        x, y, z, vx, vy, vz, roll, pitch, yaw = state
        curr_thrust = input[0]
        body_rates = input[1:]
        GRAVITY = self.G
        MASS = self.M
        wz = w # horizontal wind disturbance as a function of height

        T = jnp.array([[1, jnp.sin(roll) * jnp.tan(pitch), jnp.cos(roll) * jnp.tan(pitch)],
                        [0, jnp.cos(roll), -jnp.sin(roll)],
                        [0, jnp.sin(roll) / jnp.cos(pitch), jnp.cos(roll) / jnp.cos(pitch)]])
        curr_rolldot, curr_pitchdot, curr_yawdot = T @ body_rates

        sr = jnp.sin(roll)
        sy = jnp.sin(yaw)
        sp = jnp.sin(pitch)
        cr = jnp.cos(roll)
        cp = jnp.cos(pitch)
        cy = jnp.cos(yaw)

        vxdot = -(curr_thrust / MASS) * (sr * sy + cr * cy * sp)
        vydot = -(curr_thrust / MASS) * (cr * sy * sp - cy * sr)
        vzdot = GRAVITY - (curr_thrust / MASS) * (cr * cp)

        return jnp.hstack([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot])


class PlanarMultirotorTransformed(irx.System) :
    def __init__ (self) :
        self.xlen = 5
        self.evolution = 'continuous'
        self.G = sim_constants.GRAVITY # gravitational acceleration in m/s^2
        self.M = sim_constants.MASS # mass of the multirotor in kg

    def f(self, t, x, u, w): #jax version of Eq.30 in "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics"
        py, pz, h, v, theta = x
        u1, u2 = u
        wz = w # horizontal wind disturbance as a function of height
        G = self.G
        M = self.M

        return jnp.hstack([
            h*jnp.cos(theta) - v*jnp.sin(theta), #py_dot = hcos(theta) - vsin(theta)
            h*jnp.sin(theta) + v*jnp.cos(theta), #pz_dot = hsin(theta) + vcos(theta)
            (wz/M) * jnp.cos(theta) + G*jnp.sin(theta), #hdot = (wz/m)*cos(theta) + G*sin(theta)
            -(u1/M) * jnp.cos(theta) + G*jnp.cos(theta) - (wz/M) * jnp.sin(theta), #vdot = -(u1/m)*cos(theta) + G*cos(theta) - (wz/m)*sin(theta)
            u2
        ])

def jax_linearize_system(sys, x0, u0, w0):
    """Compute the Jacobian of the system dynamics function with respect to state and input at the initial conditions."""
    A, B = jax.jacfwd(sys.f, argnums=(1, 2))(0, x0, u0, w0) # Compute the Jacobian of the system dynamics function with respect to state and input at the initial conditions
    return A, B

jitted_linearize_system = jax.jit(jax_linearize_system, static_argnums=0)
quad_sys_planar = PlanarMultirotorTransformed()
ulim_planar = irx.interval([0, -1],[21, 1]) # Input saturation interval -> -5 <= u1 <= 15, -5 <= u2 <= 5
Q_planar = jnp.array([1, 1, 1, 1, 1]) * jnp.eye(quad_sys_planar.xlen) # weights that prioritize overall tracking of the reference (defined below)
R_planar = jnp.array([1, 1]) * jnp.eye(2)

quad_sys_3D = ThreeDMultirotorTransformed()
ulim_3D = irx.interval([0, -1, -1, -1], [21, 1, 1, 1])  # Input saturation interval -> 0 <= u1 <= 20, -1 <= u2 <= 1, -1 <= u3 <= 1
Q_3D = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) * jnp.eye(quad_sys_3D.xlen)  # weights that prioritize overall tracking of the reference (defined below)
R_3D = jnp.array([1, 1, 1, 1]) * jnp.eye(4)  # weights for the control input (thrust and body rates)

