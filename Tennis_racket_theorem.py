import numpy as np
from scipy.linalg import expm
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D

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

def uni_angular_acc(alfa, omega0, dt):
    """
    Compute the final angular velocity vector

    Parameters:
    alfa (array-like): Angular acceleration vector (3D)
    omega0 (array-like): Initial angular velocity vector (3D).
    dt (float): Time parameter.

    Returns:
    omega (array-like): Final angular velocity vector (3D).
    """
    omega = alfa * dt + omega0
    return omega

def alfa_calc(M, I, omega):
    """
    Compute the angular acceleration vector

    Parameters:
    M (array-like): Torque.
    omega (array-like): Initial angular velocity vector (3D).

    Returns:
    alfa (np.array): angular acceleration vector (3D).
    """
    alfax = (M[0] + (I[1, 1] - I[2, 2]) * omega[2] * omega[1]) / I[0, 0]
    alfay = (M[1] + (I[2, 2] - I[0, 0]) * omega[0] * omega[2]) / I[1, 1]
    alfaz = (M[2] + (I[0, 0] - I[1, 1]) * omega[1] * omega[0]) / I[2, 2]

    alfa = np.array([alfax, alfay, alfaz])
    return alfa

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

## Iterations
duration = 100  # Duration of the simulation in seconds
frames = 100000 # Number of frames
dt = duration / frames

# Initialize body
m = 1.0
x = [0.0, 0.0, 0.0]
vx = [0.0, 0.0, 0.0]
omega = [0.1, 10, 0.3]  # Angular velocity
I = [1.0, 5.0, 10.0]
u = [1.0, 0.0, 0.0]
M=[0,0,0]

body = Body(m, x, vx, omega, I, u)

# Arrays to store the orientation over time
u_history = []

# Simulation loop
for _ in range(frames):
    body.change_orientation(dt)
    body.omega=uni_angular_acc(alfa_calc(M,body.I,body.omega),body.omega,dt)
    u_history.append(body.u.copy())

# Convert orientation history to a NumPy array for easy manipulation
u_history = np.array(u_history)

## Plotting and animating

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], 'r-', lw=2)
def init():
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    return []

def update(num, u_history, line):
    line.set_data(u_history[:num, 0], u_history[:num, 1])
    line.set_3d_properties(u_history[:num, 2])
    return line,



ani = anim.FuncAnimation(fig, update, frames=frames, fargs=(u_history, line),
                         init_func=init, blit=True, interval=10)

plt.show()




