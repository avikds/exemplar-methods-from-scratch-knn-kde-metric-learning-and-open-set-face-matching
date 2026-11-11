"""
Exemplar Methods from Scratch: kNN, KDE, Metric Learning and Open-Set Face Matching scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Exemplar Methods from Scratch (Murphy, Probabilistic Machine Learning, chapter 16).

Runs every method you built on the book's own setups: the KNN curves of Figure 16.2, the factor-of-2 claim, the curse
of dimensionality, Parzen windows and bandwidth selection, KNN derived from balloon densities, Nadaraya-Watson and
LOESS, NCA and LMNN against noise dimensions, a paired open-set bake-off on Olivetti faces split by person, an online
gallery, and exact versus approximate search. No data leaves the machine, and everything is fast enough to rerun."""
import math
import time
import numpy as np
import dml_data
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from model import (pairwise_sq_dists, knn_proba, error_vs_k, one_nn_vs_bayes, cube_edge_length, noise_dimension_curve,
                      density_kernel, kde_density, silverman_bandwidth, kde_logpdf, loo_bandwidth, balloon_posterior,
                      kde_classifier, nadaraya_watson, loo_nw_bandwidth, loess, pair_gradient, nca_objective,
                      target_neighbors, lmnn_objective, fit_linear_metric, split_identities, whitened_pca,
                      apply_whitening, unit_rows, contrastive_loss, mine_triplets, triplet_loss, npairs_loss, centroid_loss,
                      pk_batch, sphere_gradient, train_embedding, open_set_predict, open_set_metrics, tune_threshold,
                      open_set_trial, face_bakeoff, online_gallery, KDTree, build_lsh, query_lsh)


def blobs3(n, r):
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    labels = r.integers(0, 3, n)
    return centres[labels] + r.normal(size=(n, 2)), labels


def noisy_blobs(seed, d=32):
    r = np.random.default_rng(seed)
    Xa, ya = blobs3(300, r)
    Xb, yb = blobs3(1000, r)
    A = np.hstack([Xa, 2.0 * r.normal(size=(300, d))])
    B = np.hstack([Xb, 2.0 * r.normal(size=(1000, d))])
    mu, sd = A.mean(axis=0), A.std(axis=0)
    return (A - mu) / sd, ya, (B - mu) / sd, yb


def knn_accuracy(W, A, ya, B, yb):
    return float(np.mean(knn_proba(A @ W.T, ya, B @ W.T, 5, 3).argmax(axis=1) == yb))


def part_knn():
    print("== Part 1. K nearest neighbours (Figures 16.1 and 16.2) ==")
    X_tr = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]])
    print("  Figure 16.1(a), neighbours {1, 1, 1, 0, 0}: p(y=1) =", knn_proba(X_tr, np.array([1, 1, 1, 0, 0]), [[1.5]], 5)[0, 1])
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=6, random_state=42)
    Xtr, ytr, Xte, yte = X[:100], y[:100], X[100:], y[100:]
    ks = [1, 2, 5, 10, 20, 50, 70, 79]
    a, b = error_vs_k(Xtr, ytr, Xte, yte, ks)
    print("  Figure 16.2(d)  K    ", ks)
    print("                  train", np.round(a, 3).tolist())
    print("                  test ", np.round(b, 3).tolist())
    print("  1-NN vs the Bayes error, delta = 1:")
    for D in (1, 5, 20):
        r = one_nn_vs_bayes(1.0, D, [100, 1000, 10000], n_test=1000, seed=0)
        print(f"    D = {D:2d}: Bayes {r['bayes']:.3f}, large-N limit {r['asymptote']:.3f}, bound {r['bound']:.3f}, 1-NN at N = 100, 1000, 10000:", np.round(r["errors"], 3).tolist())


def part_curse():
    print("\n== Part 2. The curse of dimensionality ==")
    for p, D in ((0.1, 1), (0.1, 10), (0.01, 10)):
        t, e = cube_edge_length(p, D, 100000, 0)
        print(f"  edge of a cube holding {p:.0%} of the data in {D:2d} dimensions: theory {t:.3f}, simulated {e:.3f}")
    r = np.random.default_rng(0)
    Xa, ya = blobs3(300, r)
    Xb, yb = blobs3(1000, r)
    print("  5-NN accuracy with 0, 8, 32, 128, 512 noise columns (scale 2):", np.round(noise_dimension_curve(Xa, ya, Xb, yb, [0, 8, 32, 128, 512], 5, 2.0, 0), 3).tolist())


def part_kde():
    print("\n== Part 3. Kernel density estimation ==")
    data = np.array([-2.1, -1.3, -0.4, 1.9, 5.1, 6.2])
    grid = np.linspace(-20.0, 25.0, 45001)
    for kernel in ("boxcar", "gaussian"):
        for h in (1.0, 2.0):
            p = kde_density(data, grid, h, kernel)
            print(f"  Figure 16.9 data, {kernel:8s} h = {h}: peak {p.max():.3f}, integral {p.sum() * (grid[1] - grid[0]):.3f}")
    laws = {"Gaussian": lambda r, n: r.normal(size=n),
            "bimodal": lambda r, n: np.where(r.random(n) < 0.5, r.normal(-3, 0.5, n), r.normal(3, 0.5, n)),
            "Student t3": lambda r, n: r.standard_t(3, size=n)}
    print("  mean test log density on 300 points, over 5 draws:")
    for name, law in laws.items():
        rows = []
        for seed in range(5):
            r = np.random.default_rng(seed)
            tr, te = law(r, 300), law(r, 3000)
            hs = silverman_bandwidth(tr)
            hl, _ = loo_bandwidth(tr, np.geomspace(0.02, 3.0, 30))
            rows.append((hs, hl, kde_logpdf(tr, te, hs).mean(), kde_logpdf(tr, te, hl).mean()))
        m = np.mean(rows, axis=0)
        print(f"    {name:10s} Silverman h {m[0]:.3f} -> {m[2]:.3f}   leave-one-out h {m[1]:.3f} -> {m[3]:.3f}")


def part_balloon():
    print("\n== Part 4. From KDE back to KNN ==")
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=6, random_state=42)
    Xtr, ytr, Xte, yte = X[:100], y[:100], X[100:], y[100:]
    _, post = balloon_posterior(Xtr, ytr, Xte, 7)
    print("  largest gap between the balloon posterior and eq 16.1:", float(np.abs(post - knn_proba(Xtr, ytr, Xte, 7)).max()))
    x = np.random.default_rng(0).normal(size=200)
    mass = []
    for R in (10.0, 100.0, 1000.0, 10000.0):
        g = np.r_[-np.geomspace(R, 1e-3, 1000), np.geomspace(1e-3, R, 1000)]
        c, _ = balloon_posterior(x[:, None], np.zeros(200, dtype=int), g[:, None], 5)
        mass.append(float(np.sum(0.5 * (c[1:, 0] + c[:-1, 0]) * np.diff(g))))
    print("  total mass of the balloon estimate over ranges of 10, 100, 1000, 10000:", np.round(mass, 2).tolist())
    hs = np.geomspace(0.5, 20.0, 15)
    errs = [float(np.mean(kde_classifier(Xtr, ytr, Xte, h).argmax(axis=1) != yte)) for h in hs]
    print(f"  fixed-bandwidth KDE classifier: best test error {min(errs):.3f} at h = {hs[int(np.argmin(errs))]:.2f}")
    print(f"  best KNN test error on the same data: {min(error_vs_k(Xtr, ytr, Xte, yte, range(1, 60, 2))[1]):.3f}")


def part_regression():
    print("\n== Part 5. Kernel regression ==")
    np.random.seed(0)
    x = 10 * np.linspace(-1, 1, 100)
    ytrue = np.sin(np.abs(x)) / np.abs(x)
    y = ytrue + 0.1 * np.random.randn(100)
    x = (x - x.mean()) / x.std()
    print("  the Figure 16.10 notebook grid picks h =", loo_nw_bandwidth(x, y, np.linspace(0.1, 1.0, 10))[0], "(its smallest value)")
    h, _ = loo_nw_bandwidth(x, y, np.linspace(0.02, 0.5, 49))
    print(f"  a finer grid picks h = {h:.2f}, and the MSE against the true curve is {np.mean((nadaraya_watson(x, y, x, h)[0] - ytrue) ** 2):.5f}")
    r = np.random.default_rng(0)

    def sim(n):
        xs = np.sort(r.uniform(0, 1, n))
        f = np.sin(2 * np.pi * xs) + 2 * xs
        return xs, f + 0.2 * r.normal(size=n)

    xs, ys = sim(300)
    xt, yt = sim(3000)
    h, _ = loo_nw_bandwidth(xs, ys, np.geomspace(0.005, 0.3, 25))
    for hh, sg in ((h, 0.0), (h, 0.2), (0.0005, 0.0), (0.0005, 0.2)):
        m, v = nadaraya_watson(xs, ys, xt, hh, sigma=sg)
        print(f"  eq 16.45 at h = {hh:.4f}, sigma = {sg}: coverage of the nominal 95% interval {np.mean(np.abs(yt - m) <= 1.96 * np.sqrt(v)):.3f}")
    bias = []
    for seed in range(20):
        r2 = np.random.default_rng(seed)
        xs = np.sort(r2.uniform(0, 1, 200))
        ys = 2 * xs + 1 + 0.1 * r2.normal(size=200)
        q = np.array([0.0, 0.5, 1.0])
        bias.append(np.r_[nadaraya_watson(xs, ys, q, 0.1)[0] - (2 * q + 1), loess(xs, ys, q, 0.1) - (2 * q + 1)])
    b = np.mean(bias, axis=0)
    print("  bias on a straight line at x = 0, 0.5, 1:  Nadaraya-Watson", np.round(b[:3], 3).tolist(), " LOESS", np.round(b[3:], 3).tolist())


def part_metric():
    print("\n== Part 6. Learning a metric, 32 noise dimensions, 5-NN test accuracy ==")
    for seed in range(3):
        A, ya, B, yb = noisy_blobs(seed)

        def nca_obj(V):
            J, g = nca_objective(V, A, ya)
            return 1 - J / len(A), -g / len(A)

        Wn, hist = fit_linear_metric(nca_obj, 0.5 * np.eye(34), 200, 0.02)
        T = target_neighbors(A, ya, 3)
        Wl, _ = fit_linear_metric(lambda V: lmnn_objective(V, A, ya, T, 1.0, 0.5)[:2], 0.5 * np.eye(34), 200, 0.02)
        print(f"  draw {seed}: Euclidean {knn_accuracy(np.eye(34), A, ya, B, yb):.3f}, NCA {knn_accuracy(Wn, A, ya, B, yb):.3f} (training 1-J/N {hist[-1]:.3f}), LMNN {knn_accuracy(Wl, A, ya, B, yb):.3f}")


def part_losses():
    print("\n== Part 7. Ranking losses ==")
    r = np.random.default_rng(0)
    E, _ = unit_rows(r.normal(size=(8, 3)))
    y = np.repeat([0, 1], 4)
    print("  Figure 16.6(b): triples for 8 points in 2 classes:", len(mine_triplets(E, y, 0.2, "all")), "against", 8 * 2, "centroid distances")
    y = np.repeat(np.arange(3), 4)
    c = np.eye(3)

    def triple_sums(E):
        Lt = Lu = 0.0
        for i in range(12):
            for j in range(12):
                if j == i or y[j] != y[i]:
                    continue
                for k in range(12):
                    if y[k] != y[i]:
                        Lt += np.linalg.norm(E[i] - E[j]) - np.linalg.norm(E[i] - E[k])
                        Lu += np.linalg.norm(E[i] - c[y[i]]) - np.linalg.norm(E[i] - c[y[k]]) + np.linalg.norm(E[j] - c[y[i]]) + np.linalg.norm(E[k] - c[y[k]])
        return Lt, Lu

    Cp = 3 * 2 * (12 / 3 - 1) * 12 / 3
    for name, E in (("random unit vectors", unit_rows(r.normal(size=(12, 3)))[0]), ("corners + noise 0.3", unit_rows(c[y] + 0.3 * r.normal(size=(12, 3)))[0]), ("corners + noise 0.02", unit_rows(c[y] + 0.02 * r.normal(size=(12, 3)))[0])):
        Lt, Lu = triple_sums(E)
        print(f"  eq 16.21-16.22, {name:20s}: triplet sum {Lt:8.2f} <= bound {Lu:8.2f}, and the O(NC) form gives {Cp * 12 * centroid_loss(E, y, c)[0]:8.2f}")


def method_raw(X, y, seed):
    return lambda A: A


def method_white(X, y, seed):
    p = whitened_pca(X, 100)
    return lambda A: apply_whitening(A, *p)


def on_whitened(train):
    def make(X, y, seed):
        p = whitened_pca(X, 100)
        W = train(apply_whitening(X, *p), y, seed)
        return lambda A: apply_whitening(A, *p) @ W.T
    return make


def nca_train(Z, y, seed):
    def obj(V):
        J, g = nca_objective(V, Z, y)
        return 1 - J / len(Z), -g / len(Z)
    return fit_linear_metric(obj, 0.2 * np.eye(100), 100, 0.01)[0]


def lmnn_train(Z, y, seed):
    T = target_neighbors(Z, y, 3)
    return fit_linear_metric(lambda V: lmnn_objective(V, Z, y, T, 1.0, 0.5)[:2], 0.2 * np.eye(100), 300, 0.003)[0]


def rank_train(**kw):
    return lambda Z, y, seed: train_embedding(Z, y, np.eye(100), seed=seed, **kw)[0]


def centroid_train(Z, y, seed):
    C = len(np.unique(y))
    return train_embedding(Z, y, np.eye(C, 100), loss="centroid", seed=seed)[0]


def part_faces():
    print("\n== Part 8. Open-set matching on Olivetti faces, split by person ==")
    d = dml_data.load("olivetti-faces")
    X, ids = np.asarray(d.data, dtype=float), np.asarray(d.target)
    methods = {"whitened PCA": method_white, "raw pixels": method_raw, "NCA": on_whitened(nca_train), "LMNN": on_whitened(lmnn_train),
               "contrastive": on_whitened(rank_train(loss="contrastive", m=1.0)),
               "triplet, all": on_whitened(rank_train(loss="triplet", mine="all")),
               "triplet, semi-hard": on_whitened(rank_train(loss="triplet", mine="semi-hard")),
               "triplet, hard": on_whitened(rank_train(loss="triplet", mine="hard")),
               "N-pairs": on_whitened(rank_train(loss="npairs", tau=0.1)), "centroid": on_whitened(centroid_train)}
    table = face_bakeoff(methods, X, ids, range(12))
    print("  12 splits: 17 people train, 8 calibrate, 10 enrolled with one image each, 5 strangers. Gains are paired against whitened PCA.")
    print("  12 splits cannot rank methods that are this close. Over 80 splits, the whitening step gives +0.12 AUC and the best metrics +0.02 to +0.03 more.")
    for name, r in table.items():
        print(f"    {name:20s} AUC {r['auc']:.3f} ({r['auc_gain']:+.3f} +- {r['auc_gain_se']:.3f})  rank-1 {r['rank1']:.3f}  balanced {r['balanced']:.3f}  at the oracle threshold {r['oracle']:.3f}")
    far = {g: [] for g in (1, 5, 10, 20)}
    shots = {k: [] for k in (1, 2, 3, 5)}
    for seed in range(10):
        s = split_identities(40, 17, 8, 10, 5, seed)
        p = whitened_pca(X[np.isin(ids, s["train"])], 100)
        rest = np.r_[s["calib"], s["enrolled"], s["impostors"]]
        for g in far:
            order = np.random.default_rng(seed).permutation(rest)
            gal = np.array([np.flatnonzero(ids == c)[0] for c in order[:g]])
            q = np.flatnonzero(np.isin(ids, order[g:]))
            pred, _ = open_set_predict(apply_whitening(X[gal], *p), order[:g], apply_whitening(X[q], *p), 0.35)
            far[g].append(float(np.mean(pred != -1)))
        for k in shots:
            r = np.random.default_rng(seed)
            g_idx, g_id, q_idx, q_id = [], [], [], []
            for c in s["enrolled"]:
                idx = r.permutation(np.flatnonzero(ids == c))
                g_idx += idx[:k].tolist()
                g_id += [c] * k
                q_idx += idx[k:].tolist()
                q_id += [c] * (10 - k)
            pred, _ = open_set_predict(apply_whitening(X[g_idx], *p), np.array(g_id), apply_whitening(X[q_idx], *p), -np.inf)
            shots[k].append(float(np.mean(pred == np.array(q_id))))
    print("  false accept rate at a fixed threshold 0.35, by number of enrolled people:", {g: round(float(np.mean(v)), 3) for g, v in far.items()})
    print("  rank-1 accuracy by number of images enrolled per person:", {k: round(float(np.mean(v)), 3) for k, v in shots.items()})
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6):
        rows = []
        for seed in range(6):
            s = split_identities(40, 17, 8, 10, 5, seed)
            p = whitened_pca(X[np.isin(ids, s["train"])], 100)
            test = np.random.default_rng(seed).permutation(np.flatnonzero(np.isin(ids, np.r_[s["enrolled"], s["impostors"]])))
            out = online_gallery(apply_whitening(X[test], *p), tau)
            rows.append((out.max() + 1, adjusted_rand_score(ids[test], out)))
        m = np.mean(rows, axis=0)
        print(f"  online gallery, 15 strangers, tau = {tau}: {m[0]:.1f} identities created, ARI {m[1]:.3f}")


def part_search():
    print("\n== Part 9. Searching faster ==")
    for D in (2, 5, 10, 16, 20):
        r = np.random.default_rng(D)
        X = r.uniform(size=(5000, D))
        tree = KDTree(X, 8)
        visited = [tree.query(q)[2] for q in r.uniform(size=(30, D))]
        print(f"  k-d tree in {D:2d} dimensions: each query visits {np.mean(visited) / len(X):.2%} of 5000 points")
    r = np.random.default_rng(0)
    C = r.normal(size=(200, 64))
    X = C[r.integers(0, 200, 20000)] + 0.3 * r.normal(size=(20000, 64))
    r2 = np.random.default_rng(1)
    Q = X[r2.integers(0, len(X), 100)] + 0.15 * r2.normal(size=(100, 64))
    Xn, _ = unit_rows(X)
    Qn, _ = unit_rows(Q)
    t0 = time.time()
    truth = np.argsort(-(Qn @ Xn.T), axis=1)[:, :10]
    print(f"  exact cosine scan of 20,000 points: {(time.time() - t0) / len(Q) * 1000:.2f} ms per query")
    for tables, bits in ((1, 12), (8, 12), (32, 12)):
        index = build_lsh(X, tables, bits, 0)
        recall, scanned = [], []
        for q, t in zip(Q, truth):
            got, n = query_lsh(index, X, q, 10)
            recall.append(len(set(got.tolist()) & set(t.tolist())) / 10)
            scanned.append(n)
        print(f"  LSH with {tables:2d} tables of {bits} bits: recall@10 {np.mean(recall):.3f}, scanning {np.mean(scanned) / len(X):.2%} of the data")


def main():
    t0 = time.time()
    part_knn()
    part_curse()
    part_kde()
    part_balloon()
    part_regression()
    part_metric()
    part_losses()
    part_faces()
    part_search()
    print(f"\nDone in {time.time() - t0:.0f}s")


main()

