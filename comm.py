"""Time-evolving Katz communicability and its SVD modes.

    C_k = (I - alpha A_k)^(-1) (I + exp(-dt*) (C_{k-1} - I)),   C_0 = I
    alpha = beta dt,   dt* = dt / gamma
    C_K = R S B^T  ->  B: forcing,  R: receive,  H = R * B: hub

Run:  python comm.py outputs/le3d
"""
import sys
from pathlib import Path

import numpy as np

ADJ_DIR = Path(sys.argv[1])
BETA = 1.0
GAMMA = 1.0
NEV = 10          # number of modes to keep

files = sorted(ADJ_DIR.glob("frame_*.npz"))
t = [float(np.load(f)["time"]) for f in files]
dt = t[1] - t[0]
alpha = BETA * dt
decay = np.exp(-dt / GAMMA)

n = len(np.load(files[0])["A"])
I = np.eye(n)
C = I.copy()
for f in files:
    A = np.load(f)["A"]
    C = np.linalg.solve(I - alpha * A, I + decay * (C - I))

R, S, BT = np.linalg.svd(C)
R, S, B = R[:, :NEV], S[:NEV], BT[:NEV].T
sign = np.sign(B[np.abs(B).argmax(axis=0), np.arange(NEV)])   # largest entry of each B mode > 0
R, B = R * sign, B * sign
H = R * B

np.savez(ADJ_DIR / "comm.npz", C=C, S=S, B=B, R=R, H=H, beta=BETA, dt=dt)
print(f"{len(files)} frames, {n} nodes, dt={dt:g}, alpha={alpha:g}: saved {ADJ_DIR / 'comm.npz'}")
