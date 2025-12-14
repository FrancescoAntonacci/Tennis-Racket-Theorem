from scipy.linalg import expm
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time as tm

# -------------------- Plotting Style --------------------
plt.rcParams['figure.facecolor'] = "#ffffff"
plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#d0d0d0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.7
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 20

# -------------------- Data --------------------
t, wy, wz, wx, wtot = np.loadtxt("./dati/data25.txt", unpack=True)
t = t - t.min()
sw = 0.4
start = tm.time()

# -------------------- Good Functions   ---------------------
def format_with_uncertainty(value, error, nsig=1):
    """
    Format value ± error with correct significant figures.
    nsig = number of significant digits in the uncertainty (default 2)
    """
    if error == 0:
        return f"{value:.3g} ± 0"

    err_exp = int(np.floor(np.log10(abs(error))))
    err_round = round(error, -err_exp + (nsig - 1))

    val_round = round(value, -err_exp + (nsig - 1))

    decimals = max(0, -err_exp + (nsig - 1))

    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(val_round)} ± {fmt.format(err_round)}"


# -------------------- Physics Functions --------------------
def rot_mat(omega, dt):
    omega = np.asarray(omega)
    S = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])
    return expm(S * dt)

def inertia_mat(Iyy, Izz):
    return np.array([
        [1, 0, 0],  # Ixx fixed
        [0, Iyy, 0],
        [0, 0, Izz]
    ])



def uni_angular_acc(alpha, omega0, dt):
    return omega0 + alpha * dt

def alpha_calc(gamma, I, omega):
    gamma = np.asarray(gamma)
    omega = np.asarray(omega)

    # Quadratic damping torque
    M = -gamma * omega * np.abs(omega)

    alphax = (M[0] + (I[1,1] - I[2,2]) * omega[1] * omega[2]) / I[0,0]
    alphay = (M[1] + (I[2,2] - I[0,0]) * omega[0] * omega[2]) / I[1,1]
    alphaz = (M[2] + (I[0,0] - I[1,1]) * omega[0] * omega[1]) / I[2,2]

    return np.array([alphax, alphay, alphaz])


# -------------------- Body Class --------------------
class Body:
    def __init__(self, omega, I, u):
        self.I = inertia_mat(*I)
        self.omega = omega
        self.u = u

    def change_orientation(self, dt):
        self.u = rot_mat(self.omega, dt) @ self.u

# -------------------- Simulation Model --------------------
def model(t_exp, omega0, I, omega_off,friction=[0,0,0], frames=1000):
    t_exp = np.asarray(t_exp)
    t_min, t_max = t_exp.min(), t_exp.max()
    t_fine = np.linspace(t_min, t_max, frames)
    dt = t_fine[1] - t_fine[0]

    body_sim = Body(omega0.copy(), I, np.eye(3))
    omega_fine = np.zeros((frames, 3))
    omega_fine[0] = body_sim.omega

    for i in range(1, frames):
        body_sim.change_orientation(dt)
        alpha = alpha_calc(friction, body_sim.I, body_sim.omega)
        body_sim.omega = uni_angular_acc(alpha, body_sim.omega, dt)
        omega_fine[i] = body_sim.omega

    # Efficient sampling
    indices = np.searchsorted(t_fine, t_exp)
    omega_out = omega_fine[indices] - omega_off
    omega_out[:,0] = -omega_out[:,0]  # axis convention
    return omega_out

# -------------------- Initial Parameters --------------------
p0 = np.array([
    wx[0], wy[0], wz[0],  # omega0
    5.2, 6.0,             # Iyy, Izz
    0, 0, 0,              # omega offsets
    1e-1,1e-1,1e-1
])

# -------------------- Flatten Experimental Data --------------------
ydata = np.vstack([wx, wy, wz]).T.flatten()

# -------------------- Wrapper for curve_fit --------------------
def model_wrapper(t_exp, *params):
    omega0 = np.array(params[0:3])
    I = np.array(params[3:5])       # only Iyy, Izz
    omega_off = np.array(params[5:8])
    friction=params[8:11]
    return model(t_exp, omega0, I, omega_off,friction).flatten()

# -------------------- Curve Fitting --------------------

popt, pcov = curve_fit(model_wrapper, t, ydata, p0=p0, maxfev=10000)

# -------------------- Extract Fitted Omega --------------------
omega_history = model(t, popt[0:3], popt[3:5], popt[5:8],popt[8:11])

# -------------------- Plot --------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
plt.subplots_adjust(right=0.72)

fig.suptitle("Teorema della racchetta da Tennis", fontsize=20)

# Simulation
ax1.plot(t*1e3, omega_history[:,0], "-g", label=r"Best-fit $\omega_x$")
ax1.plot(t*1e3, omega_history[:,1], "-b", label=r"Best-fit $\omega_y$")
ax1.plot(t*1e3, omega_history[:,2], "-r", label=r"Best-fit $\omega_z$")

# Experimental data
ax1.errorbar(t*1e3, wx, sw, fmt='xg', label=r"Dati $\omega_x$")
ax1.errorbar(t*1e3, wy, sw, fmt='ob', label=r"Dati $\omega_y$")
ax1.errorbar(t*1e3, wz, sw, fmt='.r', label=r"Dati $\omega_z$")

ax1.set_ylabel(r"$\omega$ [rad s$^{-1}$]")
ax1.legend()
ax1.grid(True)

## Results
# -------------------- Fit summary inside figure --------------------
residuals = (np.vstack([wx, wy, wz]).T - omega_history) / sw


perr = np.sqrt(np.diag(pcov))

Ndata = residuals.size
Npar  = len(popt)
chi2_red = np.sum(residuals**2) / (Ndata - Npar)

param_names = [
    r"$\omega_{0x}$", r"$\omega_{0y}$", r"$\omega_{0z}$",
    r"$\frac{I_{yy}}{I_{xx}}$", r"$\frac{I_{zz}}{I_{xx}}$",
    r"$\omega_{\mathrm{off},x}$", r"$\omega_{\mathrm{off},y}$", r"$\omega_{\mathrm{off},z}$",
    r"$\gamma_x$", r"$\gamma_y$", r"$\gamma_z$"
]

units=["[rad s$^{-1}$]","[rad s$^{-1}$]","[rad s$^{-1}$]"," "," ","[rad s$^{-1}$]","[rad s$^{-1}$]","[rad s$^{-1}$]","","",""]
text_lines = ["Risultati del fit:"]
for name, val, err, un in zip(param_names, popt, perr, units):
    formatted = format_with_uncertainty(val, 2*err, nsig=1)
    text_lines.append(f"{name} = {formatted} {un}")


text_lines.append(rf"$\chi^2_\nu$ = {chi2_red:.2f}")

fit_text = "\n".join(text_lines)

fig.text(
    0.74, 0.5, fit_text,     # ← coordinate della FIGURA
    ha="left",
    va="center",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9
    )
)


# Residuals
ax2.plot(t*1e3, residuals[:,0], "xg", label=r"Residual $\omega_x$")
ax2.plot(t*1e3, residuals[:,1], "ob", label=r"Residual $\omega_y$")
ax2.plot(t*1e3, residuals[:,2], ".r", label=r"Residual $\omega_z$")
ax2.axhline(0, color='k', linestyle='--', lw=1)
ax2.set_xlabel("t [ms]")
ax2.set_ylabel("Residuals")
ax2.legend()
ax2.grid(True)
plt.savefig("./grafici/tennisfit.pdf")
plt.show()


# -------------------- Print Fit Results --------------------
perr = np.sqrt(np.diag(pcov))
param_names = ["omega0_x", "omega0_y", "omega0_z", "Iyy", "Izz", "off_x", "off_y", "off_z","friction_coefficient_x","friction_coefficient_y","friction_coefficient_z"]

print("Fit results:")
for name, val, err in zip(param_names, popt, perr):
    print(f"{name} = {val:.5e} ± {2*err:.5e}")
chi2 = np.sum(residuals**2/len(residuals.flatten()))
print("χ² =", chi2)

print(f"Elapsed time: {tm.time() - start:.2f} seconds")

