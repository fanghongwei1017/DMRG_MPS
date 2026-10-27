"""Two-site finite DMRG for the open spin-1/2 antiferromagnetic chain.

H = sum_i S_i . S_(i+1), J = 1, hbar = 1. No quantum-number sectors,
block-sparse tensors, or spin/translation/reflection symmetry are used.
Only NumPy and SciPy are required for computation; Matplotlib plots results.

Run: python dmrg_heisenberg.py --validate
     python dmrg_heisenberg.py --length 64 --bonds 4 8 16 32 64 128
"""

import os

# Small dense matrices run efficiently with one BLAS thread. Set before NumPy.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

IDENTITY = np.eye(2)
SZ = np.diag([0.5, -0.5])
SP = np.array([[0.0, 1.0], [0.0, 0.0]])
SM = SP.T
SX = (SP + SM) / 2
ISY = (SP - SM) / 2  # i*S_y: real; <S_y S_y> = -<ISY ISY>.


@dataclass
class Block:
    h: np.ndarray
    z: np.ndarray
    p: np.ndarray


def empty_block():
    return Block(*(np.zeros((1, 1)) for _ in range(3)))


def enlarged_h(block, side):
    """Hamiltonian on block + site (left) or site + block (right)."""
    h, z, p = block.h, block.z, block.p
    if side == "left":
        return (np.kron(h, IDENTITY) + np.kron(z, SZ)
                + 0.5 * (np.kron(p, SM) + np.kron(p.T, SP)))
    return (np.kron(IDENTITY, h) + np.kron(SZ, z)
            + 0.5 * (np.kron(SP, p.T) + np.kron(SM, p)))


def grow_block(block, tensor, side):
    """Project an enlarged block using a left/right isometric MPS tensor."""
    if side == "left":
        q = tensor.reshape(-1, tensor.shape[2])
        eye = np.eye(tensor.shape[0])
        z, p = np.kron(eye, SZ), np.kron(eye, SP)
    else:
        q = tensor.reshape(tensor.shape[0], -1).T
        eye = np.eye(tensor.shape[2])
        z, p = np.kron(SZ, eye), np.kron(SP, eye)
    h = q.T @ enlarged_h(block, side) @ q
    return Block((h + h.T) / 2, q.T @ z @ q, q.T @ p @ q)


def random_product_state(length, seed):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(length):
        a = rng.normal(size=2)
        result.append((a / np.linalg.norm(a)).reshape(1, 2, 1))
    return result


def optimize_pair(a, b, left, right, max_bond, direction, eig_tol):
    """Solve H_eff theta = E theta, then SVD and truncate to max_bond."""
    dl, dr = a.shape[0], b.shape[2]
    shape = (dl, 2, 2, dr)
    hl, hr = enlarged_h(left, "left"), enlarged_h(right, "right")
    theta = np.tensordot(a, b, axes=(2, 0))

    def matvec(vector):
        x = vector.reshape(2 * dl, 2 * dr)
        y = (hl @ x + x @ hr.T).reshape(shape)
        x = x.reshape(shape)
        # Central bond S_i.S_(i+1); block-site bonds are already in hl/hr.
        y[:, 0, 0, :] += 0.25 * x[:, 0, 0, :]
        y[:, 1, 1, :] += 0.25 * x[:, 1, 1, :]
        y[:, 0, 1, :] += -0.25 * x[:, 0, 1, :] + 0.5 * x[:, 1, 0, :]
        y[:, 1, 0, :] += -0.25 * x[:, 1, 0, :] + 0.5 * x[:, 0, 1, :]
        return y.ravel()

    n = theta.size
    if n <= 16:
        dense = np.column_stack([matvec(v) for v in np.eye(n)])
        _, vectors = np.linalg.eigh(dense)
        vector = vectors[:, 0]
    else:
        operator = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        _, vectors = eigsh(operator, k=1, which="SA", v0=theta.ravel(),
                           tol=eig_tol, maxiter=2000, ncv=min(n, 24))
        vector = vectors[:, 0]
    u, s, vh = np.linalg.svd(vector.reshape(2 * dl, 2 * dr), full_matrices=False)
    keep = min(max_bond, len(s))
    discarded = float(np.sum(s[keep:] ** 2) / np.sum(s ** 2))
    u, s, vh = u[:, :keep], s[:keep], vh[:keep, :]
    s = s / np.linalg.norm(s)
    if direction == "right":
        a, b = u.reshape(dl, 2, keep), (s[:, None] * vh).reshape(keep, 2, dr)
    else:
        a, b = (u * s).reshape(dl, 2, keep), vh.reshape(keep, 2, dr)
    return a, b, discarded


def dmrg(length, max_bond, mps=None, seed=7, max_sweeps=40,
         energy_tol=1e-9, eig_tol=1e-11, verbose=True):
    """A sweep means a complete left-to-right AND right-to-left pass.

    Warm starts must have their orthogonality center at site zero, as returned
    by this function. Energy is measured after truncation and normalization.
    """
    if length < 2 or max_bond < 1 or max_sweeps < 3:
        raise ValueError("Require length >= 2, max_bond >= 1, max_sweeps >= 3")
    if mps is None:
        mps = random_product_state(length, seed)
    else:
        if len(mps) != length:
            raise ValueError("MPS length does not match length")
        mps = [a.copy() for a in mps]
    left = [None] * (length + 1)
    right = [None] * (length + 1)
    left[0], right[length] = empty_block(), empty_block()
    for i in range(length - 1, 0, -1):
        right[i] = grow_block(right[i + 1], mps[i], "right")
    history, stable = [], 0
    start = time.perf_counter()
    for sweep in range(1, max_sweeps + 1):
        discarded = []
        for i in range(length - 1):
            mps[i], mps[i + 1], loss = optimize_pair(
                mps[i], mps[i + 1], left[i], right[i + 2],
                max_bond, "right", eig_tol)
            left[i + 1] = grow_block(left[i], mps[i], "left")
            discarded.append(loss)
        for i in range(length - 2, -1, -1):
            mps[i], mps[i + 1], loss = optimize_pair(
                mps[i], mps[i + 1], left[i], right[i + 2],
                max_bond, "left", eig_tol)
            right[i + 1] = grow_block(right[i + 2], mps[i + 1], "right")
            discarded.append(loss)
        center = mps[0].ravel()
        energy = float(center @ enlarged_h(right[1], "right") @ center)
        change = abs(energy - history[-1]["energy"]) if history else None
        stable = stable + 1 if change is not None and change < energy_tol else 0
        row = {"m": max_bond, "sweep": sweep, "energy": energy,
               "energy_per_site": energy / length, "delta_energy": change,
               "max_discarded_weight": max(discarded),
               "elapsed_seconds": time.perf_counter() - start}
        history.append(row)
        if verbose:
            print(f"L={length} m={max_bond:3d} sweep={sweep:2d} "
                  f"E={energy:.12f} dE={change if change is not None else float('nan'):.3e} "
                  f"discard={max(discarded):.3e}", flush=True)
        if sweep >= 3 and stable >= 2:
            break
    summary = dict(history[-1])
    summary["sweeps"] = summary.pop("sweep")
    summary["converged"] = stable >= 2
    summary["actual_max_bond"] = max(a.shape[2] for a in mps)
    return mps, summary, history


def transfer_left(environment, tensor, operator=IDENTITY):
    result = np.zeros((tensor.shape[2],) * 2)
    for s, t in zip(*np.nonzero(operator)):
        result += operator[s, t] * (tensor[:, s, :].T @ environment @ tensor[:, t, :])
    return result


def correlations(mps, operator):
    """All <O_i O_j> and <O_i> by transfer contractions, without symmetry."""
    length = len(mps)
    right = [None] * (length + 1)
    right[length] = np.ones((1, 1))
    for i in range(length - 1, -1, -1):
        a = mps[i]
        right[i] = sum(a[:, s, :] @ right[i + 1] @ a[:, s, :].T for s in range(2))
    norm = float(right[0][0, 0])
    result, means = np.zeros((length, length)), np.zeros(length)
    left = np.ones((1, 1))
    for i in range(length):
        inserted = transfer_left(left, mps[i], operator)
        means[i] = np.sum(inserted * right[i + 1]) / norm
        diagonal = transfer_left(left, mps[i], operator @ operator)
        result[i, i] = np.sum(diagonal * right[i + 1]) / norm
        for j in range(i + 1, length):
            both = transfer_left(inserted, mps[j], operator)
            result[i, j] = result[j, i] = np.sum(both * right[j + 1]) / norm
            inserted = transfer_left(inserted, mps[j])
        left = transfer_left(left, mps[i])
    return result, means, norm


def exact_hamiltonian(length):
    """Full 2**L Hilbert space; used only to validate short chains."""
    states = np.arange(2 ** length)
    diagonal = np.zeros(len(states))
    rows, columns, values = [], [], []
    for i in range(length - 1):
        bit1, bit2 = length - 1 - i, length - 2 - i
        opposite = ((states >> bit1) & 1) != ((states >> bit2) & 1)
        diagonal += np.where(opposite, -0.25, 0.25)
        chosen = states[opposite]
        rows.extend(chosen)
        columns.extend(chosen ^ (1 << bit1) ^ (1 << bit2))
        values.extend(np.full(len(chosen), 0.5))
    rows.extend(states)
    columns.extend(states)
    values.extend(diagonal)
    return coo_matrix((values, (rows, columns)), shape=(len(states),) * 2).tocsr()


def dense_state(mps):
    result = mps[0]
    for a in mps[1:]:
        result = np.tensordot(result, a, axes=(-1, 0))
    return result.ravel()


def validate():
    rows = []
    for length in (2, 4, 6, 8, 10):
        h = exact_hamiltonian(length)
        v0 = np.random.default_rng(123).normal(size=h.shape[0])
        eigenvalues, eigenvectors = eigsh(h, k=1, which="SA", tol=1e-13, v0=v0)
        state, summary, _ = dmrg(length, 2 ** (length // 2), verbose=False)
        vector = dense_state(state)
        zz, _, norm = correlations(state, SZ)
        xx, _, _ = correlations(state, SX)
        tt, _, _ = correlations(state, ISY)
        dot = xx - tt + zz
        exact = eigenvectors[:, 0]
        bits = ((np.arange(h.shape[0])[:, None] >> np.arange(length - 1, -1, -1)) & 1)
        z = 0.5 - bits
        zz_exact = (z * (exact ** 2)[:, None]).T @ z
        row = {"L": length, "E_exact": float(eigenvalues[0]),
               "E_dmrg": summary["energy"],
               "energy_error": abs(summary["energy"] - eigenvalues[0]),
               "state_infidelity": abs(1 - float(vector @ exact) ** 2),
               "zz_max_error": float(np.max(np.abs(zz - zz_exact))),
               "norm_error": abs(norm - 1),
               "bond_energy_error": abs(np.trace(dot, offset=1) - summary["energy"]),
               "isotropy_error": float(max(np.max(abs(xx - zz)), np.max(abs(-tt - zz)))),
               "total_spin_squared": float(np.sum(dot))}
        assert row["energy_error"] < 1e-9, row
        assert row["zz_max_error"] < 1e-8, row
        assert row["norm_error"] < 1e-10, row
        assert row["bond_energy_error"] < 1e-9, row
        assert row["isotropy_error"] < 1e-8, row
        assert abs(row["total_spin_squared"]) < 1e-8, row
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(folder, energy_rows, center_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    bonds = [r["m"] for r in energy_rows]
    energies = np.array([r["energy"] for r in energy_rows])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    axes[0].plot(bonds, energies, "o-", color="#006b88")
    axes[0].set(xscale="log", xlabel="Retained states m", ylabel="Total energy E(m)")
    axes[0].set_xticks(bonds, [str(m) for m in bonds])
    axes[1].loglog(bonds[:-1], energies[:-1] - energies[-1], "o-", color="#cb6438")
    axes[1].set(xlabel="Retained states m", ylabel=f"E(m) - E({bonds[-1]})")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(folder / "energy_convergence.png", dpi=240)
    plt.close(fig)
    r = np.array([c["r"] for c in center_rows if c["r"] > 0])
    dot = np.array([c["C_dot"] for c in center_rows if c["r"] > 0])
    zz = np.array([c["C_zz"] for c in center_rows if c["r"] > 0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    axes[0].plot(r, dot, "o-", ms=3, label=r"$\langle\mathbf{S}_i\cdot\mathbf{S}_{i+r}\rangle$", color="#006b88")
    axes[0].plot(r, 3 * zz, "x", ms=3, label=r"$3\langle S_i^z S_{i+r}^z\rangle$", color="#cb6438")
    axes[0].axhline(0, color="0.6", lw=0.7)
    axes[0].set(xlabel="Distance r", ylabel="Spin correlation")
    axes[0].legend(fontsize=8)
    axes[1].loglog(r, np.abs(dot), "o", ms=4, color="#006b88")
    axes[1].set(xlabel="Distance r", ylabel=r"$|\langle\mathbf{S}_i\cdot\mathbf{S}_{i+r}\rangle|$")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(folder / "spin_correlations.png", dpi=240)
    plt.close(fig)


def run(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if args.validate:
        tests = validate()
        (folder / "validation.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
        return
    if args.bonds != sorted(set(args.bonds)) or any(m < 1 for m in args.bonds):
        raise ValueError("--bonds must be strictly increasing positive integers")
    mps, results, history = None, [], []
    for m in args.bonds:
        mps, summary, sweeps = dmrg(args.length, m, mps, args.seed,
                                   args.max_sweeps, args.energy_tol, args.eig_tol)
        results.append(summary)
        history.extend(sweeps)
        write_csv(folder / "energies.csv", results)
        write_csv(folder / "sweep_history.csv", history)
        np.savez_compressed(folder / f"mps_m{m}.npz", **{f"A{i}": a for i, a in enumerate(mps)})
    zz, mz, norm = correlations(mps, SZ)
    xx, mx, _ = correlations(mps, SX)
    tt, _, _ = correlations(mps, ISY)
    yy, dot = -tt, xx - tt + zz
    i = args.length // 2 - 1  # One-based center site i=L/2 for even L.
    center = [{"i": i + 1, "j": j + 1, "r": j - i, "C_zz": zz[i, j],
               "C_xx": xx[i, j], "C_yy": yy[i, j], "C_dot": dot[i, j],
               "C_zz_connected": zz[i, j] - mz[i] * mz[j]}
              for j in range(i, args.length)]
    write_csv(folder / "correlations_center.csv", center)
    for name, matrix in (("C_zz", zz), ("C_xx", xx), ("C_yy", yy), ("C_dot", dot)):
        np.savetxt(folder / f"{name}_matrix.csv", matrix, delimiter=",", fmt="%.14e")
    write_csv(folder / "local_spins.csv", [{"site": j + 1, "Sx": mx[j], "Sz": mz[j]}
                                         for j in range(args.length)])
    checks = {"norm": norm, "energy_from_correlations": float(np.trace(dot, offset=1)),
              "bond_energy_error": abs(float(np.trace(dot, offset=1)) - results[-1]["energy"]),
              "max_isotropy_error": float(max(np.max(abs(xx - zz)), np.max(abs(yy - zz)))),
              "max_abs_local_spin": float(max(np.max(abs(mx)), np.max(abs(mz)))),
              "total_spin_squared": float(np.sum(dot)),
              "max_abs_dot_minus_3zz": float(np.max(abs(dot - 3 * zz)))}
    assert checks["bond_energy_error"] < 1e-8, checks
    assert abs(norm - 1) < 1e-9, checks
    metadata = {"L": args.length, "J": 1, "boundary": "open", "seed": args.seed,
                "bonds": args.bonds, "energy_tol": args.energy_tol, "eig_tol": args.eig_tol,
                "max_sweeps": args.max_sweeps, "warm_start": True,
                "convergence_rule": "At least 3 full sweeps; 2 consecutive |delta E| < energy_tol",
                "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                "checks": checks, "energies": results}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_plots(folder, results, center)
    print("FINAL " + json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--bonds", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-sweeps", type=int, default=40)
    parser.add_argument("--energy-tol", type=float, default=1e-9)
    parser.add_argument("--eig-tol", type=float, default=1e-11)
    parser.add_argument("--output", default="results")
    parser.add_argument("--validate", action="store_true")
    run(parser.parse_args())
