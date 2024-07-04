import numpy as np
from scipy.linalg import expm
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import time as tm

start = tm.time()  # to check how slowly we are going :)

## Physics
def rot_mat(omega, dt):
    """
    Compute the rotation matrix for the increment of time dt given an angular velocity vector.

    Parameters:
    omega (array-like): Angular velocity vector (3D).
    dt (float): Time parameter.

    Returns:
    np.ndarray: 3x3 rotation matrix.
    """
    omega = np.asarray(omega)
    S = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])
    R = expm(S * dt)
    return R

def inertia_mat(Ixx, Iyy, Izz):
    """
    Compute the inertia matrix

    Parameters:
    (float): Inertias on main axes.

    Returns:
    np.ndarray: 3x3 inertia matrix.
    """
    I = np.array([
        [Ixx, 0, 0],
        [0, Iyy, 0],
        [0, 0, Izz]
    ])
    return I

def uni_angular_acc(alpha, omega0, dt):
    """
    Compute the final angular velocity vector

    Parameters:
    alpha (array-like): Angular acceleration vector (3D)
    omega0 (array-like): Initial angular velocity vector (3D).
    dt (float): Time parameter.

    Returns:
    omega (array-like): Final angular velocity vector (3D).
    """
    omega = alpha * dt + omega0
    return omega

def alpha_calc(M, I, omega):
    """
    Compute the angular acceleration vector

    Parameters:
    M (array-like): Torque.
    omega (array-like): Initial angular velocity vector (3D).

    Returns:
    alpha (np.array): angular acceleration vector (3D).
    """
    alphax = (M[0] + (I[1, 1] - I[2, 2]) * omega[2] * omega[1]) / I[0, 0]
    alphay = (M[1] + (I[2, 2] - I[0, 0]) * omega[0] * omega[2]) / I[1, 1]
    alphaz = (M[2] + (I[0, 0] - I[1, 1]) * omega[1] * omega[0]) / I[2, 2]

    alpha = np.array([alphax, alphay, alphaz])
    return alpha

## Bodies
class Body:
    def __init__(self, m, x, vx, omega, I, u):
        """
        m (float): Mass
        x (array): Position coordinates 0-x 1-y 2-z
        vx (array): Velocity coordinates 0-x 1-y 2-z
        omega (array): Angular velocity coordinates 0-x 1-y 2-z
        I (array): Inertia main components 0-x' 1-y' 2-z' (fixed to the body)
        u (array): Orientation 0-x 1-y 2-z
        """
        self.m = m
        self.x, self.y, self.z = x
        self.vx, self.vy, self.vz = vx
        self.I = inertia_mat(*I)
        self.omega = omega
        self.u = u

    def change_orientation(self, dt):
        """
        Change the orientation of the body given its angular velocity.
        """
        self.u = rot_mat(self.omega, dt) @ self.u

## Simulation parameters
duration = 10  # Duration of the simulation in seconds
frames = 10000 # Number of frames
dt = duration / frames

# Initialize body
m = 1.0
x = [0.0, 0.0, 0.0]
vx = [0.0, 0.0, 0.0]
omega = [0, 20, 1]  # Angular velocity (spin around intermediate axis)
I = [1.0, 2.0, 3.0]
u = np.eye(3)  # Initial orientation (identity matrix)
M=[0,0,0]

body = Body(m, x, vx, omega, I, u)

# Arrays to store the orientation over time
u_history = []
omega_history=[]

# Simulation loop
for _ in range(frames):
    body.change_orientation(dt)
    body.omega=uni_angular_acc(alpha_calc(M,body.I,body.omega),body.omega,dt)
    u_history.append(body.u.copy())
    omega_history.append(body.omega.copy())

# Convert orientation history to a NumPy array for easy manipulation
u_history = np.array(u_history)
omega_history=np.array(omega_history)

## Plotting
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

dot1, = ax.plot([], [], [], 'r.', markersize=10)
line1, = ax.plot([], [], [], 'r-', lw=2)

dot2, = ax.plot([], [], [], 'g.', markersize=10)
line2, = ax.plot([], [], [], 'g-', lw=2)

dot3, = ax.plot([], [], [], 'b.', markersize=10)
line3, = ax.plot([], [], [], 'b-', lw=2)

omega_vector,= ax.plot([],[],[],'k-',markersize=10)
omega_point, = ax.plot([], [], [], 'k-', lw=2)

def init():

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    line1.set_data([], [])
    line1.set_3d_properties([])
    dot1.set_data([], [])
    dot1.set_3d_properties([])

    line2.set_data([], [])
    line2.set_3d_properties([])
    dot2.set_data([], [])
    dot2.set_3d_properties([])

    line3.set_data([], [])
    line3.set_3d_properties([])
    dot3.set_data([], [])
    dot3.set_3d_properties([])

    omega_vector.set_data([], [])
    omega_vector.set_3d_properties([])
    omega_point.set_data([], [])
    omega_point.set_3d_properties([])



    return []

def animate(i):

    line1.set_data([0,  u_history[i,0, 0]], [0, u_history[i,0, 1]])
    line1.set_3d_properties([0, u_history[i,0, 2]])
    dot1.set_data([u_history[i,0, 0]], [u_history[i,0, 1]])
    dot1.set_3d_properties( [u_history[i,0, 2]])

    line2.set_data([0,  u_history[i,1, 0]], [0, u_history[i,1, 1]])
    line2.set_3d_properties([0, u_history[i,1, 2]])
    dot2.set_data([u_history[i,1, 0]], [u_history[i,1, 1]])
    dot2.set_3d_properties( [u_history[i,1, 2]])

    line3.set_data([0,  u_history[i,2, 0]], [0, u_history[i,2, 1]])
    line3.set_3d_properties([0, u_history[i,2, 2]])
    dot3.set_data([u_history[i,2, 0]], [u_history[i,2, 1]])
    dot3.set_3d_properties( [u_history[i,2, 2]])

    omega_vector.set_data([0,  omega_history[i, 0]], [0, omega_history[i, 1]])
    omega_vector.set_3d_properties([0, omega_history[i, 2]])
    omega_point.set_data([omega_history[i, 0]], [omega_history[i, 1]])
    omega_point.set_3d_properties( [omega_history[i, 2]])

    return line1, dot1, line2, dot2, line3, dot3


video = anim.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=1, blit=True, repeat=True)

plt.show()
print(f"Elapsed time: {tm.time() - start:.2f} seconds")


