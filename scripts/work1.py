import sys
print(f"{sys.path=}")
print(f"{sys.executable=}")
print(f"{sys.version=}")
print(f"{sys.version_info=}")
print(f"{sys.platform=}")
print(f"{sys.argv=}")

import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node class from rclpy to create a ROS2 node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy # Import ROS2 QoS policy modules

from px4_msgs.msg import OffboardControlMode, VehicleCommand #Import basic PX4 ROS2-API messages for switching to offboard mode
from px4_msgs.msg import TrajectorySetpoint, ActuatorMotors, VehicleThrustSetpoint, VehicleTorqueSetpoint, VehicleRatesSetpoint # Msgs for sending setpoints to the vehicle in various offboard modes
from px4_msgs.msg import VehicleOdometry, VehicleStatus, RcChannels #Import PX4 ROS2-API messages for receiving vehicle state information

import time
import traceback
from typing import Optional

import numpy as np  
import math as m
from scipy.spatial.transform import Rotation as R

from testtt.utilities import test_function
from testtt.Logger import Logger
from testtt.jax_nr import NR_tracker_original, dynamics, predict_output, get_jac_pred_u, fake_tracker, NR_tracker_flat, NR_tracker_linpred
from testtt.utilities import sim_constants # Import simulation constants
from testtt.jax_mm_rta import GPR, TVGPR # Import the transformed multirotor system dynamics class

import jax
import jax.numpy as jnp
import immrax as irx
import control

# Some configurations
jax.config.update("jax_enable_x64", True)
def jit (*args, **kwargs): # A simple wrapper for JAX's jit function to set the backend device
    device = 'cpu'
    kwargs.setdefault('backend', device)
    return jax.jit(*args, **kwargs)

# system dynamics and initial conditions
class PlanarMultirotorTransformed (irx.System) :
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


# Initial conditions
x0 = jnp.array([-1.5, -2., 0., 0., 0.1]) # [x1=py, x2=pz, x3=h, x4=v, x5=theta]
u0 = jnp.array([sim_constants.MASS*sim_constants.GRAVITY, 0.0]) # [u1=thrust, u2=roll angular rate]
w0 = jnp.array([0.01]) # [w1= unkown horizontal wind disturbance]
x0_pert = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])
ix0 = irx.icentpert(x0, x0_pert)
ulim = irx.interval([-5, -5],[15, 5]) # Input saturation interval -> -5 <= u1 <= 15, -5 <= u2 <= 5

quad_sys = PlanarMultirotorTransformed()
dx = quad_sys.f(0, x0, u0, w0) # Example call to the system dynamics function
print(f"System dynamics output: {dx}")


A = jnp.array([[0, 0, 1, 0, 0], 
               [0, 0, 0, 1, 0], 
               [0, 0, 0, 0, sim_constants.GRAVITY], 
               [0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0]])
B = jnp.array([[0, 0], 
               [0, 0], 
               [0, 0], 
               [-1/sim_constants.MASS, 0], 
               [0, 1]])
Q = jnp.array([1000, 20, 500, 500, 1]) * jnp.eye(quad_sys.xlen) # Different weights that prioritize reference reaching origin
R = jnp.array([20, 20]) * jnp.eye(2)
K, P, _ = control.lqr(A, B, Q, R)
reference_K = -1 * K
print(f"{reference_K=}")

A, B = jax.jacfwd(quad_sys.f, argnums=(1, 2))(0, x0, u0, jnp.array([0])) # Compute the Jacobian of the system dynamics function with respect to state and input at the initial conditions
print(f"{A = },\n{B = }")



# LQR Weight Matrices
Q = jnp.array([1, 1, 1, 1, 1]) * jnp.eye(quad_sys.xlen) # weights that prioritize overall tracking of the reference (defined below)
R = jnp.array([1, 1]) * jnp.eye(2)
# Call LQR to get initial K, P
K, P, _ = control.lqr(A, B, Q, R)
feedback_K = -1 * K
print(f"{feedback_K=}\n")




t0 = 0.     # Initial time
dt = 0.01  # Time step
T = 30.0   # Reachable tube horizon
tt = jnp.arange(t0, T+dt, dt)

sys_mjacM = irx.mjacM(quad_sys.f) # create a mixed Jacobian inclusion matrix for the system dynamics function
perm = irx.Permutation((0, 1, 2, 3, 4, 5, 6, 7, 8)) # create a permutation for the inclusion system calculation




@jit
def rollout (t_init, ix, xc, K_feed, K_reference, obs) :
    def mean_disturbance (t, x) :
            return GP.mean(jnp.hstack((t, x[1]))).reshape(-1)    

    def sigma(t, x):
        return jnp.sqrt(GP.variance(jnp.hstack((t, x[1]))).reshape(-1))

    def sigma_bruteforce_if(t, ix):
        div = 100
        x_list = [ix.lower + ((ix.upper - ix.lower)/div)*i for i in range(div)]
        sigma_list = 3.0*jnp.array([sigma(t, x) for x in x_list]) # # TODO: get someone to explain: why 3.0?
        return irx.interval(jnp.array([jnp.min(sigma_list)]), jnp.array([jnp.max(sigma_list)]))
        
    def step (carry, t) :
        xt_emb, xt_ref, MS = carry

        u_reference = (K_reference @ xt_ref) + jnp.array([sim_constants.GRAVITY, 0.0]) # get the reference input from the linearized planar quad LQR feedback K and current state
        u_ref_clipped = jnp.clip(u_reference, ulim.lower, ulim.upper) # clip the reference input to the input saturation limits


        # GP Interval Work
        GP_mean_t = GP.mean(jnp.array([t, xt_ref[1]])).reshape(-1) # get the mean of the disturbance at the current time and height
        xint = irx.ut2i(xt_emb) # buffer sampled sigma bound with lipschitz constant to recover guarantee
        x_div = (xint.upper - xint.lower)/(100*2) # x_div is 
        sigma_lip = MS @ x_div.T # Lipschitz constant for sigma function above
        w_diff = sigma_bruteforce_if(t, irx.ut2i(xt_emb)) # TODO: Explain
        w_diffint = irx.icentpert(0.0, w_diff.upper + sigma_lip.upper[1]) # TODO: Explain
        wint = irx.interval(GP_mean_t) + w_diffint

        
        # Compute the mixed Jacobian inclusion matrix for the system dynamics function and the disturbance function
        Mt, Mx, Mu, Mw = sys_mjacM( irx.interval(t), irx.ut2i(xt_emb), ulim, wint,
                                    centers=((jnp.array([t]), xt_ref, u_ref_clipped, GP_mean_t),), 
                                    permutations=(perm,))[0]
        
        _, MG = G_mjacM(irx.interval(jnp.array([t])), irx.ut2i(xt_emb), 
                        centers=((jnp.array([t]), xt_ref,),), 
                        permutations=(G_perm,))[0]
        Mt = irx.interval(Mt)
        Mx = irx.interval(Mx)
        Mu = irx.interval(Mu)
        Mw = irx.interval(Mw)

        

        # Embedding system for reachable tube overapproximation due to state/input/disturbance uncertainty around the quad_sys.f reference system under K_ref
        F = lambda t, x, u, w: (Mx + Mu@K_feed + Mw@MG)@(x - xt_ref) + Mw@w_diffint + quad_sys.f(0., xt_ref, u_ref_clipped, GP_mean_t) # with GP Jac
        embsys = irx.ifemb(quad_sys, F)
        xt_emb_p1 = xt_emb + dt*embsys.E(irx.interval(jnp.array([t])), xt_emb, u_ref_clipped, wint)

        # Move the reference forward in time as well
        xt_ref_p1 = xt_ref + dt*quad_sys.f(t, xt_ref, u_ref_clipped, GP_mean_t)

        
        return ((xt_emb_p1, xt_ref_p1, MS), (xt_emb_p1, xt_ref_p1, u_ref_clipped))
    


    GP = TVGPR(obs, sigma_f = 5.0, l=2.0, sigma_n = 0.01, epsilon = 0.25) # define the GP model for the disturbance
    tt = jnp.arange(0, T, dt) + t_init # define the time horizon for the rollout
    MS0 = jax.jacfwd(sigma, argnums=(1,))(t_init, xc)[0] #TODO: Explain

    G_mjacM = irx.mjacM(mean_disturbance) # TODO: Explain
    G_perm = irx.Permutation((0, 1, 2, 4, 5, 3))

    _, xx = jax.lax.scan(step, (irx.i2ut(ix), xc, irx.interval(MS0)), tt) #TODO: change variable names to be more descriptive
    return jnp.vstack((irx.i2ut(ix), xx[0])), jnp.vstack((xc, xx[1])), jnp.vstack(xx[2]) #TODO: change variable names to be more descriptive


GP_instantiation_values = jnp.array([[-2, 0.0], #make the second column all zeros
                                    [0, 0.0],
                                    [2, 0.0],
                                    [4, 0.0],
                                    [6, 0.0],
                                    [8, 0.0],
                                    [10, 0.0],
                                    [12, 0.0]]) # at heights of y in the first column, disturbance to the values in the second column

# add a time dimension at t=0 to the GP instantiation values for TVGPR instantiation
actual_disturbance_GP = TVGPR(jnp.hstack((jnp.zeros((GP_instantiation_values.shape[0], 1)), GP_instantiation_values)), 
                                       sigma_f = 5.0, 
                                       l=2.0, 
                                       sigma_n = 0.01,
                                       epsilon=0.1,
                                       discrete=False
                                       )


def actual_disturbance_test(t, x) :
    return actual_disturbance_GP.mean(jnp.array([t, x[1]])).reshape(-1)


n_obs = 9
obs = jnp.tile(jnp.array([[0, x0[1], actual_disturbance_test(0.0, x0)[0]]]),(n_obs,1))
print(f"{actual_disturbance_test(0.0, jnp.array([0, 0])) = }")
print(f"{actual_disturbance_test(0.0, jnp.array([0, -0.55])) = }")
print(f"{actual_disturbance_test(0.0, jnp.array([0, -5])) = }")
print(f"{actual_disturbance_test(0.0, jnp.array([0, -10])) = }")

print(f"{obs = }")


reachable_tube, reference, feedfwd_input = rollout(0.0, ix0, x0, feedback_K, reference_K, obs)



@jit
def collection_id_jax(xref, xemb, threshold=0.3):
    diff1 = jnp.abs(xref - xemb[:, :xref.shape[1]]) > threshold
    diff2 = jnp.abs(xref - xemb[:, xref.shape[1]:]) > threshold
    nan_mask = jnp.isnan(xref).any(axis=1) | jnp.isnan(xemb).any(axis=1)
    fail_mask = diff1.any(axis=1) | diff2.any(axis=1) | nan_mask

    # Safe handling using lax.cond
    return jax.lax.cond(
        jnp.any(fail_mask),
        lambda _: jnp.argmax(fail_mask),  # return first failing index
        lambda _: -1,                     # otherwise -1
        operand=None
    )

time0 = time.time()
violation_safety_time_idx = collection_id_jax(reference, reachable_tube)
print(f"Time taken for collection_id_jax: {time.time() - time0} , Collection ID: {violation_safety_time_idx}")

time0 = time.time()
violation_safety_time_idx = collection_id_jax(reference, reachable_tube)
print(f"After jitting: time taken is {time.time()-time0}, CollectionID: {violation_safety_time_idx}")


def u_applied(x, xref, uref, K_feedback):
    u_nom = (K_feedback @ (xref - x)) + uref
    u_applied_clipped = jnp.clip(u_nom, ulim.lower, ulim.upper) # clip the applied input to the input saturation limits
    return u_applied_clipped
 
 

# exit(0)

class OffboardControl(Node):


    def __init__(self, sim: bool) -> None:
        super().__init__('px4_rta_mm_gpr_node')
        test_function()
        # Initialize essential variables
        self.sim: bool = sim
        self.GRAVITY: float = 9.806 # m/s^2, gravitational acceleration

        print(quad_sys.f)

        if self.sim:
            print("Using simulator constants and functions")
            from testtt.utilities import sim_constants # Import simulation constants
            self.MASS = sim_constants.MASS
            self.THRUST_CONSTANT = sim_constants.THRUST_CONSTANT #x500 gazebo simulation motor thrust constant
            self.MOTOR_VELOCITY_ARMED = sim_constants.MOTOR_VELOCITY_ARMED #x500 gazebo motor velocity when armed
            self.MAX_ROTOR_SPEED = sim_constants.MAX_ROTOR_SPEED #x500 gazebo simulation max rotor speed
            self.MOTOR_INPUT_SCALING = sim_constants.MOTOR_INPUT_SCALING #x500 gazebo simulation motor input scaling

        elif not self.sim:
            print("Using hardware constants and functions")
            #TODO: do the hardware version of the above here
            try:
                from testtt.utilities import hardware_constants
                self.MASS = hardware_constants.MASS
            except ImportError:
                raise ImportError("Hardware not implemented yet.")


        # Logging related variables
        self.time_log = []
        self.x_log, self.y_log, self.z_log, self.yaw_log = [], [], [], []
        self.ctrl_comp_time_log = []
        # self.m0_log, self.m1_log, self.m2_log, self.m3_log = [], [], [], [] # direct actuator control logs
        # self.f_log, self.M_log = [], [] # force and moment logs
        self.throttle_log, self.roll_rate_log, self.pitch_rate_log, self.yaw_rate_log = [], [], [], [] # throttle and rate logs
        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                ])

##########################################################################################
        # Time variables
        self.T0 = time.time() # (s) initial time of program
        self.time_from_start = time.time() - self.T0 # (s) time from start of program 
        self.begin_actuator_control = 10 # (s) time after which we start sending actuator control commands
        self.land_time = self.begin_actuator_control + 15 # (s) time after which we start sending landing commands

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.actuator_motors_publisher = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
        self.vehicle_thrust_setpoint_publisher = self.create_publisher(
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.vehicle_torque_setpoint_publisher = self.create_publisher(
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.vehicle_rates_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_subscriber_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_subscriber_callback, qos_profile)
            
        self.offboard_mode_rc_switch_on: bool = True if self.sim else False   # RC switch related variables and subscriber
        self.MODE_CHANNEL: int = 5 # Channel for RC switch to control offboard mode (-1: position, 0: offboard, 1: land)
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" for position v offboard v land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_subscriber_callback, qos_profile
        )
        
        # MoCap related variables
        self.mocap_initialized: bool = False
        self.full_rotations: int = 0

        # PX4 variables
        self.offboard_heartbeat_counter: int = 0
        self.vehicle_status = VehicleStatus()
        # self.takeoff_height = -5.0

        # Callback function time constants
        self.heartbeat_period: float = 0.1 # (s) We want 10Hz for offboard heartbeat signal
        self.control_period: float = 0.01 # (s) We want 1000Hz for direct control algorithm
        self.traj_idx = 0 # Index for trajectory setpoint
        # Timers for my callback functions
        self.offboard_timer = self.create_timer(self.heartbeat_period,
                                                self.offboard_heartbeat_signal_callback) #Offboard 'heartbeat' signal should be sent at 10Hz
        self.control_timer = self.create_timer(self.control_period,
                                               self.control_algorithm_callback) #My control algorithm needs to execute at >= 100Hz

        # Initialize newton-raphson algorithm parameters
        self.last_input: np.ndarray = np.array([self.MASS * self.GRAVITY, 0.01, 0.02, 0.03]) # last input to the controller
        self.T_LOOKAHEAD: float = 0.8 # (s) lookahead time for the controller in seconds
        self.T_LOOKAHEAD_PRED_STEP: float = 0.1 # (s) we do state prediction for T_LOOKAHEAD seconds ahead in intervals of T_LOOKAHEAD_PRED_STEP seconds
        self.INTEGRATION_TIME: float = self.control_period # integration time constant for the controller in seconds

        # print("Initializing and jit-compiling the NR tracker function")
        init_state = np.array([0.1, 0.1, 0.1, 0.02, 0.03, 0.02, 0.01, 0.01, 0.03]) # Initial state vector for testing
        init_input = self.last_input  # Initial input vector for testing
        init_ref = np.array([0.0, 0.0, -3.0, 0.0])  # Initial reference vector for testing  
        # print(f"{init_state=}, {init_input=}, {init_ref=}")
        # print(f"{init_state.shape=}, {init_input.shape=}, {init_ref.shape=}")
        # u,v = NR_tracker_original(init_state, init_input, init_ref, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS)
        # fake_tracker(init_state, init_input, init_ref, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS)
        NR_tracker_original(init_state, init_input, init_ref, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS) # JIT-compile the NR tracker function


    def rc_channel_subscriber_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print('In RC Channel Callback')
        flight_mode = rc_channels.channels[self.MODE_CHANNEL-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on: bool = True if flight_mode >= 0.75 else False


    def adjust_yaw(self, yaw: float) -> float:
        """Adjust yaw angle to account for full rotations and return the adjusted yaw.

        This function keeps track of the number of full rotations both clockwise and counterclockwise, and adjusts the yaw angle accordingly so that it reflects the absolute angle in radians. It ensures that the yaw angle is not wrapped around to the range of -pi to pi, but instead accumulates the full rotations.
        This is particularly useful for applications where the absolute orientation of the vehicle is important, such as in control algorithms or navigation systems.
        The function also initializes the first yaw value and keeps track of the previous yaw value to determine if a full rotation has occurred.

        Args:
            yaw (float): The yaw angle in radians from the motion capture system after being converted from quaternion to euler angles.

        Returns:
            psi (float): The adjusted yaw angle in radians, accounting for full rotations.
        """        
        mocap_psi = yaw
        psi = None

        if not self.mocap_initialized:
            self.mocap_initialized = True
            self.prev_mocap_psi = mocap_psi
            psi = mocap_psi
            return psi

        # MoCap angles are from -pi to pi, whereas the angle state variable should be an absolute angle (i.e. no modulus wrt 2*pi)
        #   so we correct for this discrepancy here by keeping track of the number of full rotations.
        if self.prev_mocap_psi > np.pi*0.9 and mocap_psi < -np.pi*0.9: 
            self.full_rotations += 1  # Crossed 180deg in the CCW direction from +ve to -ve rad value so we add 2pi to keep it the equivalent positive value
        elif self.prev_mocap_psi < -np.pi*0.9 and mocap_psi > np.pi*0.9:
            self.full_rotations -= 1 # Crossed 180deg in the CW direction from -ve to +ve rad value so we subtract 2pi to keep it the equivalent negative value

        psi = mocap_psi + 2*np.pi * self.full_rotations
        self.prev_mocap_psi = mocap_psi
        
        return psi


    def vehicle_odometry_subscriber_callback(self, msg) -> None:
        """Callback function for vehicle odometry topic subscriber."""
        print("==" * 30)
        print("\n\n")
        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2] #+ (2.25 * self.sim) # Adjust z for simulation, new gazebo model has ground level at around -1.39m 

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]


        self.roll, self.pitch, yaw = R.from_quat(msg.q, scalar_first=True).as_euler('xyz', degrees=False)
        self.yaw = self.adjust_yaw(yaw)  # Adjust yaw to account for full rotations
        r_final = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=False)         # Final rotation object
        self.rotation_object = r_final  # Store the final rotation object for further use

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.full_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw, self.p, self.q, self.r])
        self.nr_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw])
        self.flat_state_vector = np.array([self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0.])
        self.rta_mm_gpr_state_vector = np.array([self.y, self.vy, self.z, self.vz, self.roll])  # For the transformed multirotor system dynamics
        self.output_vector = np.array([self.x, self.y, self.z, self.yaw])
        self.position = np.array([self.x, self.y, self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.quat = self.rotation_object.as_quat()  # Quaternion representation (xyzw)
        self.ROT = self.rotation_object.as_matrix()
        self.omega = np.array([self.p, self.q, self.r])

        # print(f"{self.full_state_vector=}")
        print(f"{self.nr_state_vector=}")
        # print(f"{self.flat_state_vector=}")
        print(f"{self.output_vector=}")
        print(f"{self.roll = }, {self.pitch = }, {self.yaw = }(rads)")
        # print(f"{self.rotation_object.as_euler('xyz', degrees=True) = } (degrees)")
        # print(f"{self.ROT = }")


    def vehicle_status_subscriber_callback(self, vehicle_status) -> None:
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self) -> None:
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self) -> None:
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self) -> None:
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self) -> None:
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal_position(self) -> None:
        """Publish the offboard control mode heartbeat for position-only setpoints."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.offboard_control_mode_publisher.publish(msg)
        # self.get_logger().info("Switching to position control mode")

    def publish_offboard_control_heartbeat_signal_actuators(self) -> None:
        """Publish the offboard control mode heartbeat for actuator-only setpoints."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = True
        self.offboard_control_mode_publisher.publish(msg)
        # self.get_logger().info("Switching to actuator control mode")

    def publish_offboard_control_heartbeat_signal_thrust_moment(self) -> None:
        """Publish the offboard control mode heartbeat for actuator-only setpoints."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = True
        msg.direct_actuator = False
        self.offboard_control_mode_publisher.publish(msg)
        # self.get_logger().info("Switching to force and moment control mode")

    def publish_offboard_control_heartbeat_signal_body_rate(self) -> None:
        """Publish the offboard control mode heartbeat for body rate setpoints."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.offboard_control_mode_publisher.publish(msg)
        # self.get_logger().info("Switching to body rate control mode")
    

    def publish_position_setpoint(self, x: float = 0.0, y: float = 0.0, z: float = -3.0, yaw: float = 0.0) -> None:
        """Publish the trajectory setpoint.

        Args:
            x (float, optional): Desired x position in meters. Defaults to 0.0.
            y (float, optional): Desired y position in meters_. Defaults to 0.0.
            z (float, optional): Desired z position in meters. Defaults to -3.0.
            yaw (float, optional): Desired yaw position in radians. Defaults to 0.0.

        Returns:
            None

        Raises:
            TypeError: If x, y, z, or yaw are not of type float.
        Raises:
            ValueError: If x, y, z are not within the expected range.
        """
        for name, val in zip(("x","y","z","yaw"), (x,y,z,yaw)):
            if not isinstance(val, float):
                raise TypeError(
                                f"\n{'=' * 60}"
                                f"\nInvalid input type for {name}\n"
                                f"Expected float\n"
                                f"Received {type(val).__name__}\n"
                                f"{'=' * 60}"
                                )
               
        # if not (-2.0 <= x <= 2.0) or not (-2.0 <= y <= 2.0) or not (-3.0 <= z <= -0.2):
        #     raise ValueError("x must be between -2.0 and 2.0, y must be between -2.0 and 2.0, z must be between -0.2 and -3.0")
        
        msg = TrajectorySetpoint()
        msg.position = [x, y, z] # position in meters
        msg.yaw = yaw # yaw in radians
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z, yaw]}")


    def publish_actuator_setpoint(self, m0: float = 0.0, m1: float = 0.0, m2: float = 0.0, m3: float = 0.0) -> None:

        """Publish the actuator setpoint.

        Args:
            m0 (float): Desired throttle for motor 0.
            m1 (float): Desired throttle for motor 1.
            m2 (float): Desired throttle for motor 2.
            m3 (float): Desired throttle for motor 3.

        Returns:
            None

        Raises:
            ValueError: If m0, m1, m2, or m3 are not within 0-1.
        """
        for name, val in zip(("m0", "m1", "m2", "m3"), (m0, m1, m2, m3)):
            if not (0 <= val <= 1):
                raise ValueError(
                                f"\n{'=' * 60}"
                                f"\nInvalid input for {name}\n"
                                f"Expected value between 0 and 1\n"
                                f"Received {val}\n"
                                f"{'=' * 60}"
                                )        

        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        # 0th motor seems to be back left motor when we're aligned with quad's 0 yaw position
        # 1st motor seems to be front right motor when we're aligned with quad's 0 yaw position
        # 2nd motor seems to be back right motor when we're aligned with quad's 0 yaw position
        # 3rd motor seems to be front left motor when we're aligned with quad's 0 yaw position
        msg.control = [m0, m1, m2, m3] + 8 * [0.0]
        self.actuator_motors_publisher.publish(msg)
        self.get_logger().info(f"Publishing actuator setpoints: {msg.control}")

    def publish_force_moment_setpoint(self, f: float = 0.0, M: list[float] = [0.0, 0.0, 0.0]) -> None:
        """Publish the force and moment setpoint.
        
        Args:
            f (float): Desired force in Newtons.
            M (float): Desired moment in Newtons.
        
        Returns:
            None
                    
        Raises:
            ValueError: If f is not within 0-1 or M is not within -1 to 1.
        """
        if not (0 <= f <= 1):
            raise ValueError(
                            f"\n{'=' * 60}"
                            f"\nInvalid input for force\n"
                            f"Expected value between 0 and 1\n"
                            f"Received {f}\n"
                            f"{'=' * 60}"
                            )
        if not all(-1 <= m <= 1 for m in M):
            raise ValueError(
                            f"\n{'=' * 60}"
                            f"\nInvalid input for moment\n"
                            f"Expected values between -1 and 1 for each component\n"
                            f"Received {M}\n"
                            f"{'=' * 60}"
                            )
        
        msg1 = VehicleThrustSetpoint()
        msg1.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg1.xyz = [0.,0.,-1.0]  # Thrust in Newtons
        self.vehicle_thrust_setpoint_publisher.publish(msg1)
        self.get_logger().info(f"Publishing thrust setpoint: {msg1.xyz}")

        msg2 = VehicleTorqueSetpoint()
        msg2.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg2.xyz = [0.0, 0.0, 0.0]  # Torque in Newton-meters
        self.vehicle_torque_setpoint_publisher.publish(msg2)
        self.get_logger().info(f"Publishing torque setpoint: {msg2.xyz}")

    def publish_body_rate_setpoint(self, throttle: float = 0.0, p: float = 0.0, q: float = 0.0, r: float = 0.0) -> None:
        """Publish the body rate setpoint.
        
        Args:
            p (float): Desired roll rate in radians per second.
            q (float): Desired pitch rate in radians per second.
            r (float): Desired yaw rate in radians per second.
            throttle (float): Desired throttle in normalized from [-1,1] in NED body frame

        Returns:
            None
        
        Raises:
            ValueError: If p, q, r, or throttle are not within expected ranges.
        """

        
        # print(f"Publishing body rate setpoint: roll={p}, pitch={q}, yaw={r}, throttle={throttle}")
        msg = VehicleRatesSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.roll = p
        msg.pitch = q
        msg.yaw = r
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -1 * float(throttle)
        self.vehicle_rates_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing body rate setpoint: roll={p}, pitch={q}, yaw={r}, thrust_body={throttle}")

        # exit(0)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def offboard_heartbeat_signal_callback(self) -> None:
        """Callback function for the heartbeat signals that maintains flight controller in offboard mode and switches between offboard flight modes."""
        self.time_from_start = time.time() - self.T0

        if not self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard to send heartbeat signal, engage offboard, and arm
            print(f"Offboard Callback: RC Flight Mode Channel {self.MODE_CHANNEL} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
            self.offboard_heartbeat_counter = 0
            return

        if self.time_from_start <= self.begin_actuator_control:
            self.publish_offboard_control_heartbeat_signal_position()
        elif self.time_from_start <= self.land_time:  
            self.publish_offboard_control_heartbeat_signal_body_rate()
        elif self.time_from_start > self.land_time:
            self.publish_offboard_control_heartbeat_signal_position()
        else:
            raise ValueError("Unexpected time_from_start value")

        if self.offboard_heartbeat_counter <= 10:
            if self.offboard_heartbeat_counter == 10:
                self.engage_offboard_mode()
                self.arm()
            self.offboard_heartbeat_counter += 1

        
        
    def control_algorithm_callback(self) -> None:
        """Callback function to handle control algorithm once in offboard mode."""

        if not (self.offboard_mode_rc_switch_on and (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD) ):
            print(f"Not in offboard mode.\n"
                  f"Current state: {self.vehicle_status.nav_state}\n"
                  f"Expected offboard state: {VehicleStatus.NAVIGATION_STATE_OFFBOARD}\n"
                  f"Offboard RC switch status: {self.offboard_mode_rc_switch_on}")
            return

        if self.time_from_start <= self.begin_actuator_control:
            self.publish_position_setpoint(0., 0., -10.0, 0.0)

        elif self.time_from_start <= self.land_time:
            # f, M = self.control_administrator()
            # self.publish_force_moment_setpoint(f, M)
            self.control_administrator()

        elif self.time_from_start > self.land_time:
            print("Landing...")
            self.publish_position_setpoint(0.0, 0.0, -0.55, 0.0)
            if abs(self.x) < 0.2 and abs(self.y) < 0.2 and abs(self.z) <= 0.52:
                print("Vehicle is close to the ground, preparing to land.")
                self.land()                    
                exit(0)

        else:
            raise ValueError("Unexpected time_from_start value")

    def control_administrator(self) -> None:

        ref = None
        ctrl_T0 = time.time()
        self.time_from_start = ctrl_T0 -self.T0
        new_u, _ = NR_tracker_original(self.nr_state_vector, self.last_input, ref, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS)
        rta_input = self.rta_mm_gpr_administrator()
        control_comp_time = time.time() - ctrl_T0 # Time taken for control computation
        print(f"Control Computation Time: {control_comp_time:.4f} seconds, Good for {1/control_comp_time:.2f}Hz control loop")


        print(f"{new_u= }")
        print(f"{rta_input = }")
        exit(0)


        self.last_input = new_u  # Update the last input for the next iteration
        new_force = new_u[0]
        new_throttle = float(self.get_throttle_command_from_force(new_force))
        new_roll_rate = float(new_u[1])  # Convert jax.numpy array to float
        new_pitch_rate = float(new_u[2])  # Convert jax.numpy array to float
        new_yaw_rate = float(new_u[3])    # Convert jax.numpy array to float
        self.publish_body_rate_setpoint(new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate)
        # exit(0)

        # Log the states, inputs, and reference trajectories for data analysis
        state_input_ref_log_info = [self.time_from_start,
                                    float(self.x), float(self.y), float(self.z), float(self.yaw),
                                    control_comp_time,
                                    new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate,
                                    ]
        self.update_logged_data(state_input_ref_log_info)
        print("==" * 30)
        print("\n\n")


    def rta_mm_gpr_administrator(self):
        """Run the RTA-MM administrator to compute the control inputs."""
        thresh = 0.3
        tube_horizon = 30.0
        current_time = self.time_from_start
        current_state = self.rta_mm_gpr_state_vector
        current_state_interval = irx.icentpert(current_state, x0_pert)

        applied_input = u_applied(current_state, self.reference[self.traj_idx, :], self.feedfwd_input[self.traj_idx, :], feedback_K)
        self.traj_idx += 1

        if current_time >= self.collection_time:
            print("Unsafe region begins now")
            self.reachable_tube, self.reference, self.feedfwd_input = rollout(current_time, current_state_interval, current_state, feedback_K, reference_K, obs)
            t_arr = np.arange(current_time, current_time + tube_horizon, dt)
            t_index = collection_id_jax(self.reference, self.reachable_tube, thresh)
            self.collection_time = t_arr[t_index]
            self.traj_idx = 0

        
        return applied_input

    def get_throttle_command_from_force(self, collective_thrust): #Converts force to throttle command
        """ Convert the positive collective thrust force to a positive throttle command. """
        print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
        if self.sim:
            try:
                motor_speed = m.sqrt(collective_thrust / (4.0 * self.THRUST_CONSTANT))
                throttle_command = (motor_speed - self.MOTOR_VELOCITY_ARMED) / self.MOTOR_INPUT_SCALING
                return throttle_command
            except Exception as e:
                print(f"Error in throttle conversion: {e}")
                exit(1)
                # return 0.0

        if not self.sim: # I got these parameters from a curve fit of the throttle command vs collective thrust from the hardware spec sheet
            a = 0.00705385408507030
            b = 0.0807474474438391
            c = 0.0252575818743285
            throttle_command = a*collective_thrust + b*m.sqrt(collective_thrust) + c  # equation form is a*x + b*sqrt(x) + c = y
            return throttle_command


# ~~ The following functions handle the log update and data retrieval for analysis ~~
    def update_logged_data(self, data):
        print("Updating Logged Data")
        self.time_log.append(data[0])
        self.x_log.append(data[1])
        self.y_log.append(data[2])
        self.z_log.append(data[3])
        self.yaw_log.append(data[4])
        self.ctrl_comp_time_log.append(data[5])
        self.throttle_log.append(data[6])
        self.roll_rate_log.append(data[7])
        self.pitch_rate_log.append(data[8])
        self.yaw_rate_log.append(data[9])


    def get_time_log(self): return np.array(self.time_log).reshape(-1, 1)
    def get_x_log(self): return np.array(self.x_log).reshape(-1, 1)
    def get_y_log(self): return np.array(self.y_log).reshape(-1, 1)
    def get_z_log(self): return np.array(self.z_log).reshape(-1, 1)
    def get_yaw_log(self): return np.array(self.yaw_log).reshape(-1, 1)
    def get_ctrl_comp_time_log(self): return np.array(self.ctrl_comp_time_log).reshape(-1, 1)
    # def get_m0_log(self): return np.array(self.m0_log).reshape(-1, 1)
    # def get_m1_log(self): return np.array(self.m1_log).reshape(-1, 1)
    # def get_m2_log(self): return np.array(self.m2_log).reshape(-1, 1)
    # def get_m3_log(self): return np.array(self.m3_log).reshape(-1, 1)
    # def get_f_log(self): return np.array(self.f_log).reshape(-1, 1)
    # def get_M_log(self): return np.array(self.M_log).reshape(-1, 1)
    def get_throttle_log(self): return np.array(self.throttle_log).reshape(-1, 1)
    def get_roll_rate_log(self): return np.array(self.roll_rate_log).reshape(-1, 1)
    def get_pitch_rate_log(self): return np.array(self.pitch_rate_log).reshape(-1, 1)
    def get_yaw_rate_log(self): return np.array(self.yaw_rate_log).reshape(-1, 1)

    def get_metadata(self): return self.metadata.reshape(-1, 1)




# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main(args=None):
    sim: Optional[bool] = None
    logger = None 
    offboard_control: Optional[OffboardControl] = None

    def shutdown_logging():
        print(
              f"Interrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...\n"
              f"{'=' * 65}"
              )
        if logger:
            logger.log(offboard_control)
        if offboard_control:
            offboard_control.destroy_node()
        rclpy.shutdown()

    try:

        print(              
            f"{65 * '='}\n"
            f"Initializing ROS 2 node: '{__name__}' for offboard control\n"
            f"{65 * '='}\n"
        )

        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        sim_val = int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: "))
        if sim_val not in (0, 1):
            raise ValueError(
                            f"\n{65 * '='}\n"
                            f"Invalid input for sim: {sim_val}, expected 0 or 1\n"
                            f"{65 * '='}\n")
        sim = bool(sim_val)
        print(f"{'SIMULATION' if sim else 'HARDWARE'}")

        rclpy.init(args=args)
        offboard_control = OffboardControl(sim)

        logger = Logger([sys.argv[1]])  # Create logger with passed filename
        rclpy.spin(offboard_control)    # Spin the ROS 2 node

    except KeyboardInterrupt:
        print(
              f"\n{65 * '='}\n"
              f"Keyboard interrupt detected (Ctrl+C), exiting...\n"
              )
    except Exception as e:
        # print(f"\nError in main: {e}")
        traceback.print_exc()
    finally:
        shutdown_logging()
        print("\nNode has shut down.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError in __main__: {e}")
        traceback.print_exc()