"""3D lift-element network: adjacency matrix A for each snapshot.

    L_E = -(L_x dphi/dx + L_y dphi/dy),   L = omega x u
    A[i, j] = | L_E(u + du_j) - L_E(u) |(x_i) / eps

du_j = curl(psi q) is a small divergence-free perturbation with a Gaussian psi
at node j.  Nodes inside the wing are left out.  Run:  python adjacency_3d.py
"""
from pathlib import Path

import numpy as np

DATA = "sample/g2_d0.5_101-110.npz"
OUT_DIR = Path("outputs/le3d")
SKIP = 4                          # every SKIP-th grid point (in x, y, z) is a node
# SKIP = 2                        # same nodes as the paper (18,018); A ~1.2 GB per frame, comm needs ~10 GB
EPS = 1e-3                        # perturbation amplitude
Q = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)   # axis of the perturbation


def lift_element(u, v, w, omega_x, omega_y, omega_z, grad_x, grad_y, dx, dy, dz):
    # Vorticity: the sample has omega_z only, so omega_x and omega_y are computed here.
    # Comment out a line below to use that component from the data instead.
    omega_x = np.gradient(w, dy, axis=1) - np.gradient(v, dz, axis=2)
    omega_y = np.gradient(u, dz, axis=2) - np.gradient(w, dx, axis=0)
    # omega_z = np.gradient(v, dx, axis=0) - np.gradient(u, dy, axis=1)
    lx = omega_y * w - omega_z * v
    ly = omega_z * u - omega_x * w
    return -(lx * grad_x + ly * grad_y)


def perturbation(X, Y, Z, xj, yj, zj, r_xy, r_z, eps):
    """Velocity curl(psi Q) and its vorticity curl(curl(psi Q)) for a Gaussian psi."""
    rx, ry, rz = X - xj, Y - yj, Z - zj
    a, b = r_xy**2, r_z**2
    psi = 0.5 * eps * a * np.exp(-(rx**2 + ry**2) / (2 * a) - rz**2 / (2 * b))
    px, py, pz = -rx / a * psi, -ry / a * psi, -rz / b * psi             # grad psi
    pxx, pyy, pzz = (rx**2 / a**2 - 1 / a) * psi, (ry**2 / a**2 - 1 / a) * psi, (rz**2 / b**2 - 1 / b) * psi
    pxy, pxz, pyz = rx * ry / a**2 * psi, rx * rz / (a * b) * psi, ry * rz / (a * b) * psi
    lap = pxx + pyy + pzz
    du = py * Q[2] - pz * Q[1]
    dv = pz * Q[0] - px * Q[2]
    dw = px * Q[1] - py * Q[0]
    domega_x = Q[0] * pxx + Q[1] * pxy + Q[2] * pxz - Q[0] * lap
    domega_y = Q[0] * pxy + Q[1] * pyy + Q[2] * pyz - Q[1] * lap
    domega_z = Q[0] * pxz + Q[1] * pyz + Q[2] * pzz - Q[2] * lap
    return du, dv, dw, domega_x, domega_y, domega_z


d = np.load(DATA)
x, y, z = d["x"][::SKIP], d["y"][::SKIP], d["z"][::SKIP]
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
r_xy, r_z = 1.5 * min(dx, dy), 1.5 * dz
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
grad_x = np.repeat(d["grad_phi_x"][::SKIP, ::SKIP, None], len(z), axis=2)
grad_y = np.repeat(d["grad_phi_y"][::SKIP, ::SKIP, None], len(z), axis=2)
node = np.flatnonzero(~d["wing_mask"][::SKIP, ::SKIP, ::SKIP])   # grid points outside the wing
nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])[node]
OUT_DIR.mkdir(parents=True, exist_ok=True)

for k, frame in enumerate(d["frames"]):
    u = d["u"][k][::SKIP, ::SKIP, ::SKIP]
    v = d["v"][k][::SKIP, ::SKIP, ::SKIP]
    w = d["w"][k][::SKIP, ::SKIP, ::SKIP]
    omega_z = d["omega_z"][k][::SKIP, ::SKIP, ::SKIP]
    omega_x = omega_y = np.zeros_like(u)       # not in the sample (computed in lift_element)
    # omega_x = d["omega_x"][k][::SKIP, ::SKIP, ::SKIP]   # for data with all components
    # omega_y = d["omega_y"][k][::SKIP, ::SKIP, ::SKIP]
    le = lift_element(u, v, w, omega_x, omega_y, omega_z, grad_x, grad_y, dx, dy, dz)

    A = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for j, (xj, yj, zj) in enumerate(nodes):
        du, dv, dw, dox, doy, doz = perturbation(X, Y, Z, xj, yj, zj, r_xy, r_z, EPS)
        le_p = lift_element(u + du, v + dv, w + dw, omega_x + dox, omega_y + doy, omega_z + doz,
                            grad_x, grad_y, dx, dy, dz)
        A[:, j] = np.abs(le_p - le).ravel()[node] / EPS

    np.savez(OUT_DIR / f"frame_{frame:07d}.npz", A=A, x=x, y=y, z=z, node=node, le=le, time=d["time"][k])
    print(f"frame {frame}: {len(nodes)} nodes")
