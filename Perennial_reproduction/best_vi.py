#!/usr/bin/env python3
"""Best mean-field variational approximation of the Table S1 model, found by random restarts
and ranked with NumPyro's ELBO (the same scoring idea as the VB bakeoff simpleVI code:
TraceMeanField_ELBO, a fixed-seed Monte-Carlo estimate, positive parameters represented as
exp(Normal) in the guide).

Pipeline (run from the Perennial_reproduction folder):
  1. Rscript best_vi_stan.R            # 50 restarts of Stan ADVI via cmdstanr -> best_vi/stan_restarts_moments.csv
  2. python best_vi.py pymc            # 50 restarts of PyMC ADVI             -> best_vi/pymc_restarts_moments.csv
  3. python best_vi.py score           # NumPyro ELBO of all restarts, picks the best
                                       #   -> best_vi/restart_elbos.csv, TableS1_best_vi.csv
  (compare_plots.R then adds the "Best VI" line when it finds TableS1_best_vi.csv.)

Every restart is a diagonal Normal over the SAME unconstrained coordinates, in this order
(the order PyMC uses; best_vi_stan.R writes Stan's fits in the same order):
  b_Intercept (2: Pr, De) | b (3x2 row-major: rows MAT, P1, P2; cols Pr, De) | log sd (2) |
  z (n_genus x 2 row-major) | log phi (1)
`score` evaluates each restart under ONE NumPyro copy of the model (priors are the assumed brms
defaults, see pymc_reproducing.py), so PyMC and Stan fits are ranked on equal footing.

Requirements: stage `pymc` needs pymc (and the PYTENSOR_FLAGS setting from pymc_reproducing.py if
you needed one); stage `score` needs jax + numpyro (pip install numpyro).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pymc_reproducing as P          # data loading / PyMC model; importing it does not import pymc

OUT_DIR = Path("best_vi")
PYMC_FILE = OUT_DIR / "pymc_restarts_moments.csv"
STAN_FILE = OUT_DIR / "stan_restarts_moments.csv"
ELBO_FILE = OUT_DIR / "restart_elbos.csv"
BEST_TABLE = Path("TableS1_best_vi.csv")
NUTS_FILE = Path("TableS1_reproduced.csv")
Z975 = 1.959963984540054


def n_params(n_genus):
    return 2 + 6 + 2 + 2 * n_genus + 1


def unpack(vec, n_genus):
    """Canonical flat vector -> named pieces (works on numpy or jax arrays)."""
    return dict(b_Intercept=vec[0:2], b=vec[2:8].reshape(3, 2), sd=vec[8:10],
                z=vec[10:10 + 2 * n_genus].reshape(n_genus, 2), phi=vec[10 + 2 * n_genus])


def moments_files(main_file):
    """The main moments CSV plus any per-seed-range part files written by parallel runs
    (e.g. best_vi/stan_restarts_moments_part_001-005.csv), that exist."""
    parts = sorted(main_file.parent.glob(f"{main_file.stem}_part_*.csv"))
    return [p for p in [main_file, *parts] if p.exists()]


def done_seeds(main_file):
    return {int(s) for f in moments_files(main_file) for s in pd.read_csv(f)["seed"]}


def read_moments(paths, n_genus):
    """-> (seeds, means[n_restarts, P], sds[n_restarts, P]) from one or more moments CSVs."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).drop_duplicates("seed").sort_values("seed")
    mcols = sorted([c for c in df.columns if c.startswith("mean_")], key=lambda c: int(c.split("_")[1]))
    scols = sorted([c for c in df.columns if c.startswith("sd_")], key=lambda c: int(c.split("_")[1]))
    if len(mcols) != n_params(n_genus) or len(scols) != n_params(n_genus):
        raise ValueError(f"{paths}: expected {n_params(n_genus)} mean/sd columns, found {len(mcols)}/{len(scols)}")
    return df["seed"].to_numpy(int), df[mcols].to_numpy(float), df[scols].to_numpy(float)


def fixef_table(mean, sd, n_genus, seed, method, elbo, package):
    """Table S1 rows for one mean-field fit. Intercepts and slopes are identity-transformed
    Normals in the guide, so mean, sd and the 95% interval are exact (no draws needed)."""
    u_m, u_s = unpack(np.asarray(mean), n_genus), unpack(np.asarray(sd), n_genus)
    rows = []
    for t in P.TERMS:
        resp, coef = t.split("_", 1)
        j = {"muPrB": 0, "muDeB": 1}[resp]
        if coef == "Intercept":
            m, s = u_m["b_Intercept"][j], u_s["b_Intercept"][j]
        else:
            k = {"MAT": 0, "polyMAP21": 1, "polyMAP22": 2}[coef]
            m, s = u_m["b"][k, j], u_s["b"][k, j]
        rows.append(dict(term=t, Estimate=float(m), **{"Est.Error": float(s)},
                         **{"Q2.5": float(m - Z975 * s), "Q97.5": float(m + Z975 * s)},
                         seed=seed, method=method, package=package, elbo=elbo))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ stage: pymc
def stage_pymc(args):
    import pymc as pm
    OUT_DIR.mkdir(exist_ok=True)
    d = P.load_data()
    G = d["n_genus"]
    model = P.build_model(d)
    names = list(model.initial_point().keys())
    if names != ["b_Intercept", "b", "sd_log__", "z", "phi_log__"]:
        sys.exit(f"Unexpected PyMC parameter order {names}; the canonical layout would not match.")
    done = done_seeds(PYMC_FILE)
    kw = dict(obj_n_mc=args.n_mc, progressbar=False)
    if args.optimizer == "adam":
        kw["obj_optimizer"] = pm.adam()
    # --first/--last run one seed range (for parallel jobs), each writing its own part file
    first, last = (args.first or 1), (args.last or args.n)
    out_file = PYMC_FILE if (args.first is None and args.last is None) else \
        OUT_DIR / f"{PYMC_FILE.stem}_part_{first:03d}-{last:03d}.csv"
    print(f"PyMC {pm.__version__}: seeds {first}..{last}, {args.iters} iterations, obj_n_mc={args.n_mc}, "
          f"optimizer={args.optimizer}; writing {out_file}; already done: {sorted(done)}")
    for seed in range(first, last + 1):
        if seed in done:
            continue
        t0 = time.time()
        try:
            advi = pm.ADVI(model=model, random_seed=seed)
            advi.fit(args.iters, **kw)
            mean = np.asarray(advi.approx.mean.eval(), float)
            sd = np.asarray(advi.approx.std.eval(), float)
        except Exception as e:                                      # noqa: BLE001
            print(f"  seed {seed}: failed ({e})")
            continue
        if mean.size != n_params(G) or not (np.all(np.isfinite(mean)) and np.all(np.isfinite(sd))):
            print(f"  seed {seed}: skipped (wrong size or non-finite values)")
            continue
        row = {"seed": seed, **{f"mean_{i + 1}": v for i, v in enumerate(mean)},
               **{f"sd_{i + 1}": v for i, v in enumerate(sd)}}
        pd.DataFrame([row]).to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
        print(f"  seed {seed}: done in {time.time() - t0:.0f}s", flush=True)
    print("Wrote", out_file)


# ------------------------------------------------------------------------------ stage: score
def score_sites(n_genus):
    """[(site, shape, transform)] of the scoring model in the order of the canonical flat vector;
    transform "exp" marks the sites the guide represents as exp(Normal)."""
    return [("b_Intercept", (2,), "identity"), ("b", (3, 2), "identity"), ("sd", (2,), "exp"),
            ("z", (n_genus, 2), "identity"), ("phi", (), "exp")]


def score_model_and_guide(d):
    """The NumPyro model and fixed mean-field guide used by `score` (also read by
    simpleVI's sampled-KL check, so both score the same model)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import gammaln

    G, n = d["n_genus"], d["n"]
    X, logY, w, g = jnp.asarray(d["X"]), jnp.log(jnp.asarray(d["Y"])), jnp.asarray(d["w"]), jnp.asarray(d["g"])

    def model(mean=None, sd=None):
        del mean, sd
        b0 = numpyro.sample("b_Intercept", dist.StudentT(3.0, 0.0, 2.5).expand([2]).to_event(1))
        b = numpyro.sample("b", dist.ImproperUniform(dist.constraints.real, (), (3, 2)))      # flat prior
        sdv = numpyro.sample("sd", dist.ImproperUniform(dist.constraints.positive, (), (2,)))
        numpyro.factor("sd_prior", dist.StudentT(3.0, 0.0, 2.5).log_prob(sdv).sum() + 2 * jnp.log(2.0))  # half-t
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([G, 2]).to_event(2))
        phi = numpyro.sample("phi", dist.Gamma(0.01, 0.01))
        eta = b0 + X @ b + (z * sdv)[g]
        alpha = jax.nn.softmax(jnp.concatenate([jnp.zeros((n, 1)), eta], axis=1), axis=1) * phi
        ll = gammaln(phi) - gammaln(alpha).sum(1) + ((alpha - 1.0) * logY).sum(1)
        numpyro.factor("loglik", (w * ll).sum())                                               # brms weights()

    def guide(mean, sd):
        m, s = unpack(mean, G), unpack(jnp.maximum(sd, 1e-8), G)
        numpyro.sample("b_Intercept", dist.Normal(m["b_Intercept"], s["b_Intercept"]).to_event(1))
        numpyro.sample("b", dist.Normal(m["b"], s["b"]).to_event(2))
        numpyro.sample("sd", dist.TransformedDistribution(dist.Normal(m["sd"], s["sd"]),
                                                          dist.transforms.ExpTransform()).to_event(1))
        numpyro.sample("z", dist.Normal(m["z"], s["z"]).to_event(2))
        numpyro.sample("phi", dist.TransformedDistribution(dist.Normal(m["phi"], s["phi"]),
                                                           dist.transforms.ExpTransform()))

    return model, guide


def stage_score(args):
    import jax
    jax.config.update("jax_enable_x64", True)        # before any jax array is created
    import jax.numpy as jnp
    from numpyro.infer import TraceMeanField_ELBO

    d = P.load_data()
    G = d["n_genus"]
    model, guide = score_model_and_guide(d)
    elbo = TraceMeanField_ELBO(num_particles=args.chunk)

    @jax.jit
    def chunk_loss(key, mean, sd):
        return elbo.loss(key, {}, model, guide, mean, sd)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_chunks)     # same keys for every restart

    def score(mean, sd):
        losses = [float(chunk_loss(k, jnp.asarray(mean), jnp.asarray(sd))) for k in keys]
        return -float(np.mean(losses))                                         # ELBO: higher is better

    levels_ok = True
    if (OUT_DIR / "genus_levels.csv").exists():
        r_levels = pd.read_csv(OUT_DIR / "genus_levels.csv")["level"].astype(str).tolist()
        py_levels = list(pd.factorize(pd.read_csv(P.DATA_FILE)["Genus"])[1].astype(str))
        levels_ok = r_levels == py_levels
        if not levels_ok:
            sys.exit("Genus order in R (best_vi/genus_levels.csv) differs from the order used here; "
                     "the random-effect coordinates would not line up.")

    all_rows, fits = [], {}
    for package, main_file in [("pymc", PYMC_FILE), ("stan", STAN_FILE)]:
        paths = moments_files(main_file)
        if not paths:
            print(f"skipping {package}: no {main_file} (or part files) found")
            continue
        seeds, means, sds = read_moments(paths, G)
        print(f"\nScoring {len(seeds)} {package} restarts ({args.chunk * args.n_chunks} particles each)...")
        for sd_seed, m, s in zip(seeds, means, sds):
            try:
                e = score(m, s)
            except Exception as ex:                                            # noqa: BLE001
                print(f"  {package} seed {sd_seed}: scoring failed ({ex})")
                e = np.nan
            all_rows.append(dict(package=package, seed=int(sd_seed), elbo=e))
            fits[(package, int(sd_seed))] = (m, s)
    if not all_rows:
        sys.exit("No restart files found. Run best_vi_stan.R and/or `python best_vi.py pymc` first.")

    res = pd.DataFrame(all_rows)
    res.to_csv(ELBO_FILE, index=False)
    ok = res[np.isfinite(res["elbo"])]
    if ok.empty:
        sys.exit("Every restart had a non-finite ELBO.")
    print("\nELBO by package (higher is better):")
    print(ok.groupby("package")["elbo"].describe()[["count", "min", "50%", "max"]].round(2).to_string())
    print("\nTop 5 restarts overall:")
    print(ok.sort_values("elbo", ascending=False).head(5).round(3).to_string(index=False))

    best = ok.sort_values("elbo", ascending=False).iloc[0]
    bm, bs = fits[(best["package"], int(best["seed"]))]
    label = f"Best VI ({best['package']}, seed {int(best['seed'])})"
    tab = fixef_table(bm, bs, G, int(best["seed"]), label, float(best["elbo"]), best["package"])
    tab.to_csv(BEST_TABLE, index=False)
    print(f"\nBest approximation: {label}, ELBO = {best['elbo']:.3f}\nWrote {ELBO_FILE} and {BEST_TABLE}")

    # --- sanity checks: a coordinate mismatch would show up here ---
    per_pkg = ok.groupby("package")["elbo"].max()
    if len(per_pkg) == 2 and abs(per_pkg.iloc[0] - per_pkg.iloc[1]) > 50:
        print(f"WARNING: best ELBO differs by {abs(per_pkg.iloc[0] - per_pkg.iloc[1]):.0f} between packages. "
              "A large gap can be a real difference, but also check the Stan <-> canonical parameter mapping.")
    if NUTS_FILE.exists():
        nuts = pd.read_csv(NUTS_FILE).set_index("term")
        for package in per_pkg.index:
            s_ = ok[ok["package"] == package].sort_values("elbo", ascending=False).iloc[0]
            m_, sd_ = fits[(package, int(s_["seed"]))]
            t_ = fixef_table(m_, sd_, G, int(s_["seed"]), package, float(s_["elbo"]), package).set_index("term")
            off = ((t_["Estimate"] - nuts["Estimate"]) / nuts["Est.Error"]).abs().max()
            print(f"  {package}: largest |mean offset| from R NUTS over the 8 terms = {off:.2f} NUTS SDs"
                  + ("   <-- unusually large, check the mapping" if off > 3 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("pymc", help="run PyMC ADVI random restarts")
    p.add_argument("--n", type=int, default=50, help="number of restarts (seeds 1..n)")
    p.add_argument("--first", type=int, default=None, help="first seed of a range (parallel jobs; writes a part file)")
    p.add_argument("--last", type=int, default=None, help="last seed of a range (parallel jobs; writes a part file)")
    p.add_argument("--iters", type=int, default=50_000, help="ADVI iterations per restart")
    p.add_argument("--n-mc", type=int, default=100, help="Monte-Carlo samples per gradient step (obj_n_mc)")
    p.add_argument("--optimizer", choices=["default", "adam"], default="default")
    s = sub.add_parser("score", help="score all restarts with NumPyro's ELBO and pick the best")
    s.add_argument("--chunk", type=int, default=5000, help="particles per ELBO evaluation")
    s.add_argument("--n-chunks", type=int, default=10, help="evaluations averaged (total particles = chunk * n-chunks)")
    s.add_argument("--seed", type=int, default=20240512, help="fixed ELBO-evaluation seed (same for every restart)")
    args = ap.parse_args()
    {"pymc": stage_pymc, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
