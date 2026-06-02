from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


RIDALL_DEFAULT_SEASON_CODE = "2324"
RIDALL_DEFAULT_LEAGUE_CODE = "E0"
RIDALL_DEFAULT_REFERENCE_TEAM = "Arsenal"
RIDALL_PRIOR_DELTA = 10.0
RIDALL_BIVARIATE_VB_KAPPA = 6.323

RIDALL_REFERENCE_NOTES = {
    "paper": (
        "Ridall, Titman, and Pettitt (2025) model EPL scores with attack, "
        "defense, home-goal advantage, and a Gamma multiplicative match effect."
    ),
    "static_model": (
        "These notebooks implement the static score model using the marginalized "
        "bivariate negative-binomial likelihood from Equation (2.3)."
    ),
    "implementation_choice": (
        "The match-effect parameter kappa is fixed to the paper's Bayes.BV.VB "
        "estimate 6.323 from Table 2 to keep the random-restart comparison "
        "focused on the attack, defense, and home-advantage variational geometry."
    ),
    "identifiability": (
        "The paper fixes beta_1=1. We use Arsenal as the reference team when it "
        "appears in the selected EPL season, matching the paper's convention."
    ),
}

RIDALL_PAPER_COMPARISONS = [
    {
        "source": "Ridall et al. 2025 Table 2",
        "method": "Bayes.BV.VB SSM",
        "quantity": "kappa",
        "value": RIDALL_BIVARIATE_VB_KAPPA,
        "notes": "Dynamic bivariate variational Bayes nuisance-parameter estimate.",
    },
    {
        "source": "Ridall et al. 2025 Table 3",
        "method": "Bayes.BV.VB",
        "quantity": "cumulative_rps_relative_to_bookmakers",
        "value": 17.55,
        "notes": "Best reported dynamic Bayesian SSM predictive score.",
    },
    {
        "source": "Ridall et al. 2025 Table 3",
        "method": "Bayes.UV.VB",
        "quantity": "cumulative_rps_relative_to_bookmakers",
        "value": 18.64,
        "notes": "Reported dynamic univariate Bayesian SSM predictive score.",
    },
    {
        "source": "Ridall et al. 2025 Table 3",
        "method": "Bayes.BV.Ax",
        "quantity": "cumulative_rps_relative_to_bookmakers",
        "value": 19.16,
        "notes": "Reported one-step bivariate Bayesian SSM predictive score.",
    },
]


def ridall_season_label(season_code=RIDALL_DEFAULT_SEASON_CODE):
    season_code = str(season_code)
    start = 2000 + int(season_code[:2])
    end = 2000 + int(season_code[2:])
    return f"{start}-{str(end)[-2:]}"


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _repo_relative_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(_repo_root().resolve()))
    except (OSError, ValueError):
        return str(path)


def _default_data_dir():
    return _repo_root() / "single_MC" / "ridall_football" / "data"


def ridall_data_url(
    season_code=RIDALL_DEFAULT_SEASON_CODE,
    league_code=RIDALL_DEFAULT_LEAGUE_CODE,
):
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"


def ridall_data_path(
    season_code=RIDALL_DEFAULT_SEASON_CODE,
    league_code=RIDALL_DEFAULT_LEAGUE_CODE,
    data_dir=None,
):
    data_dir = _default_data_dir() if data_dir is None else Path(data_dir)
    return data_dir / f"{league_code}_{season_code}.csv"


def download_ridall_football_data(
    season_code=RIDALL_DEFAULT_SEASON_CODE,
    league_code=RIDALL_DEFAULT_LEAGUE_CODE,
    data_dir=None,
):
    path = ridall_data_path(season_code, league_code, data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(ridall_data_url(season_code, league_code), path)
    return path


def _ordered_teams(frame, reference_team=RIDALL_DEFAULT_REFERENCE_TEAM):
    teams = sorted(set(frame["HomeTeam"]).union(set(frame["AwayTeam"])))
    if reference_team in teams:
        teams = [reference_team] + [team for team in teams if team != reference_team]
    return teams


def make_ridall_param_names(teams, unconstrained=True):
    suffix = "_log__" if unconstrained else ""
    attack_names = [f"attack{suffix}[{team}]" for team in teams]
    defense_names = [f"defense{suffix}[{team}]" for team in teams[1:]]
    hga_name = "home_advantage_log__" if unconstrained else "home_advantage"
    return attack_names + defense_names + [hga_name]


def make_ridall_natural_param_names(teams):
    return (
        [f"attack[{team}]" for team in teams]
        + [f"defense[{team}]" for team in teams]
        + ["home_advantage"]
    )


def ridall_parameter_slices(n_teams):
    n_teams = int(n_teams)
    attack = slice(0, n_teams)
    defense_free = slice(n_teams, 2 * n_teams - 1)
    home_advantage = 2 * n_teams - 1
    return {
        "attack": attack,
        "defense_free": defense_free,
        "home_advantage": home_advantage,
        "dim": 2 * n_teams,
    }


def load_ridall_football_data(
    season_code=RIDALL_DEFAULT_SEASON_CODE,
    league_code=RIDALL_DEFAULT_LEAGUE_CODE,
    data_dir=None,
    reference_team=RIDALL_DEFAULT_REFERENCE_TEAM,
    download=True,
):
    path = ridall_data_path(season_code, league_code, data_dir=data_dir)
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Could not find {path}. Set download=True or put the football-data CSV there."
            )
        path = download_ridall_football_data(season_code, league_code, data_dir=data_dir)

    frame = pd.read_csv(path)
    frame.columns = frame.columns.map(str.strip)
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Football data are missing required columns: {sorted(missing)}")

    frame = frame.dropna(subset=list(required)).copy()
    frame["_row"] = np.arange(len(frame))
    frame["parsed_date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    frame = frame.sort_values(["parsed_date", "_row"], na_position="last").reset_index(drop=True)

    teams = _ordered_teams(frame, reference_team=reference_team)
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    frame["home_idx"] = frame["HomeTeam"].map(team_to_idx).astype(int)
    frame["away_idx"] = frame["AwayTeam"].map(team_to_idx).astype(int)

    home_goals = frame["FTHG"].astype(int).to_numpy()
    away_goals = frame["FTAG"].astype(int).to_numpy()
    outcome = np.where(home_goals > away_goals, "H", np.where(home_goals == away_goals, "D", "A"))

    return {
        "path": _repo_relative_path(path),
        "frame": frame,
        "season_code": season_code,
        "season_label": ridall_season_label(season_code),
        "league_code": league_code,
        "teams": teams,
        "team_to_idx": team_to_idx,
        "reference_team": teams[0],
        "free_defense_teams": teams[1:],
        "n_teams": len(teams),
        "home_team": frame["home_idx"].to_numpy(dtype=np.int32),
        "away_team": frame["away_idx"].to_numpy(dtype=np.int32),
        "home_goals": home_goals.astype(np.int32),
        "away_goals": away_goals.astype(np.int32),
        "outcome": outcome,
        "param_names": make_ridall_param_names(teams, unconstrained=True),
        "natural_param_names": make_ridall_natural_param_names(teams),
    }


def default_ridall_plot_dims(param_names, teams):
    wanted = [
        f"attack_log__[{teams[0]}]",
        "attack_log__[Man City]",
        "attack_log__[Liverpool]",
        "defense_log__[Man City]",
        "defense_log__[Luton]",
        "home_advantage_log__",
    ]
    dims = [param_names.index(name) for name in wanted if name in param_names]
    if len(dims) < 4:
        fallback = list(range(min(3, len(param_names)))) + [len(param_names) - 1]
        dims = list(dict.fromkeys(dims + fallback))
    labels = [param_names[idx] for idx in dims]
    return dims, labels


def bivariate_negative_binomial_logpmf_np(home_goals, away_goals, lambda_home, lambda_away, kappa):
    from scipy.special import gammaln

    home_goals = np.asarray(home_goals, dtype=float)
    away_goals = np.asarray(away_goals, dtype=float)
    lambda_home = np.asarray(lambda_home, dtype=float)
    lambda_away = np.asarray(lambda_away, dtype=float)
    kappa = float(kappa)
    total_rate = kappa + lambda_home + lambda_away
    return (
        gammaln(kappa + home_goals + away_goals)
        - gammaln(kappa)
        - gammaln(home_goals + 1.0)
        - gammaln(away_goals + 1.0)
        + home_goals * np.log(lambda_home)
        + away_goals * np.log(lambda_away)
        + kappa * np.log(kappa)
        - (kappa + home_goals + away_goals) * np.log(total_rate)
    )


def build_numpyro_ridall_bivariate_model(
    n_teams,
    kappa=RIDALL_BIVARIATE_VB_KAPPA,
    delta=RIDALL_PRIOR_DELTA,
):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import gammaln

    n_teams = int(n_teams)
    kappa = float(kappa)
    delta = float(delta)

    def model(home_team, away_team, home_goals, away_goals):
        attack = numpyro.sample(
            "attack",
            dist.Gamma(
                jnp.full(n_teams, delta),
                jnp.full(n_teams, delta),
            ).to_event(1),
        )
        defense_free = numpyro.sample(
            "defense_free",
            dist.Gamma(
                jnp.full(n_teams - 1, delta),
                jnp.full(n_teams - 1, delta),
            ).to_event(1),
        )
        home_advantage = numpyro.sample("home_advantage", dist.Gamma(delta, delta))
        defense = jnp.concatenate([jnp.ones(1), defense_free])

        lambda_home = attack[home_team] * defense[away_team] * home_advantage
        lambda_away = attack[away_team] * defense[home_team]
        total_rate = kappa + lambda_home + lambda_away

        home_goals_f = jnp.asarray(home_goals, dtype=jnp.float32)
        away_goals_f = jnp.asarray(away_goals, dtype=jnp.float32)
        logp = (
            gammaln(kappa + home_goals_f + away_goals_f)
            - gammaln(kappa)
            - gammaln(home_goals_f + 1.0)
            - gammaln(away_goals_f + 1.0)
            + home_goals_f * jnp.log(lambda_home)
            + away_goals_f * jnp.log(lambda_away)
            + kappa * jnp.log(kappa)
            - (kappa + home_goals_f + away_goals_f) * jnp.log(total_rate)
        )
        numpyro.factor("obs", jnp.sum(logp))

    return model


def make_tfp_ridall_conditioned_log_prob(
    data,
    kappa=RIDALL_BIVARIATE_VB_KAPPA,
    delta=RIDALL_PRIOR_DELTA,
    dtype=None,
):
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    if dtype is None:
        dtype = tf.float32

    n_teams = int(data["n_teams"])
    home_team = tf.convert_to_tensor(data["home_team"], dtype=tf.int32)
    away_team = tf.convert_to_tensor(data["away_team"], dtype=tf.int32)
    home_goals = tf.convert_to_tensor(data["home_goals"], dtype=dtype)
    away_goals = tf.convert_to_tensor(data["away_goals"], dtype=dtype)
    kappa = tf.convert_to_tensor(float(kappa), dtype=dtype)
    delta = tf.convert_to_tensor(float(delta), dtype=dtype)

    @tf.function(jit_compile=False)
    def log_prob_fn(theta):
        theta = tf.convert_to_tensor(theta, dtype=dtype)
        attack_log = theta[..., :n_teams]
        defense_free_log = theta[..., n_teams : 2 * n_teams - 1]
        hga_log = theta[..., 2 * n_teams - 1]

        attack = tf.exp(attack_log)
        defense_free = tf.exp(defense_free_log)
        home_advantage = tf.exp(hga_log)
        defense = tf.concat([tf.ones_like(defense_free[..., :1]), defense_free], axis=-1)

        attack_home = tf.gather(attack, home_team, axis=-1)
        attack_away = tf.gather(attack, away_team, axis=-1)
        defense_home = tf.gather(defense, home_team, axis=-1)
        defense_away = tf.gather(defense, away_team, axis=-1)

        lambda_home = attack_home * defense_away * home_advantage[..., tf.newaxis]
        lambda_away = attack_away * defense_home
        total_rate = kappa + lambda_home + lambda_away

        logp = (
            tf.math.lgamma(kappa + home_goals + away_goals)
            - tf.math.lgamma(kappa)
            - tf.math.lgamma(home_goals + 1.0)
            - tf.math.lgamma(away_goals + 1.0)
            + home_goals * tf.math.log(lambda_home)
            + away_goals * tf.math.log(lambda_away)
            + kappa * tf.math.log(kappa)
            - (kappa + home_goals + away_goals) * tf.math.log(total_rate)
        )

        gamma_prior = tfd.Gamma(concentration=delta, rate=delta)
        prior = tf.reduce_sum(gamma_prior.log_prob(attack) + attack_log, axis=-1)
        prior += tf.reduce_sum(gamma_prior.log_prob(defense_free) + defense_free_log, axis=-1)
        prior += gamma_prior.log_prob(home_advantage) + hga_log
        return prior + tf.reduce_sum(logp, axis=-1)

    return log_prob_fn


def build_pymc_ridall_bivariate_model(
    data,
    kappa=RIDALL_BIVARIATE_VB_KAPPA,
    delta=RIDALL_PRIOR_DELTA,
):
    import pymc as pm
    import pytensor.tensor as pt

    n_teams = int(data["n_teams"])
    home_team = data["home_team"]
    away_team = data["away_team"]
    home_goals = data["home_goals"].astype(float)
    away_goals = data["away_goals"].astype(float)
    kappa = float(kappa)
    delta = float(delta)

    with pm.Model() as model:
        attack = pm.Gamma("attack", alpha=delta, beta=delta, shape=n_teams)
        defense_free = pm.Gamma("defense_free", alpha=delta, beta=delta, shape=n_teams - 1)
        home_advantage = pm.Gamma("home_advantage", alpha=delta, beta=delta)
        defense = pt.concatenate([pt.ones(1), defense_free])

        lambda_home = attack[home_team] * defense[away_team] * home_advantage
        lambda_away = attack[away_team] * defense[home_team]
        total_rate = kappa + lambda_home + lambda_away

        logp = (
            pt.gammaln(kappa + home_goals + away_goals)
            - pt.gammaln(kappa)
            - pt.gammaln(home_goals + 1.0)
            - pt.gammaln(away_goals + 1.0)
            + home_goals * pt.log(lambda_home)
            + away_goals * pt.log(lambda_away)
            + kappa * np.log(kappa)
            - (kappa + home_goals + away_goals) * pt.log(total_rate)
        )
        pm.Potential("obs_loglike", pt.sum(logp))

    return model


def make_ridall_reference_summary_frame():
    rows = []
    for key, note in RIDALL_REFERENCE_NOTES.items():
        rows.append(
            {
                "source": "implementation_note",
                "method": key,
                "quantity": "note",
                "value": np.nan,
                "notes": note,
            }
        )
    rows.extend(RIDALL_PAPER_COMPARISONS)
    return pd.DataFrame(rows)


def save_ridall_reference_summary(path):
    path = Path(path)
    make_ridall_reference_summary_frame().to_csv(path, index=False)
    return path


def load_ridall_best_references(path="best_reference_values.csv"):
    reference_summary = pd.read_csv("ridall_reference_summary.csv")
    best_reference_path = Path(path)
    if not best_reference_path.exists():
        raise FileNotFoundError(
            f"Best reference values not found at {best_reference_path}. "
            "Run compute_best_reference.py after generating saved restart outputs."
        )
    best_reference = pd.read_csv(best_reference_path)
    best_reference_means = {
        row["parameter"]: [(float(row["mean"]), row["source"])]
        for _, row in best_reference.iterrows()
    }
    best_reference_stds = {
        row["parameter"]: [(float(row["sd"]), row["source"])]
        for _, row in best_reference.iterrows()
    }
    return reference_summary, best_reference, best_reference_means, best_reference_stds


def print_ridall_reference_summary():
    print("Ridall football reference notes:")
    for key, note in RIDALL_REFERENCE_NOTES.items():
        print(f"- {key}: {note}")
    print("\nDigitized paper comparison values:")
    print(pd.DataFrame(RIDALL_PAPER_COMPARISONS))


def exp_positive_moments(loc, scale):
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)
    mean = np.exp(loc + 0.5 * scale**2)
    var = (np.exp(scale**2) - 1.0) * np.exp(2.0 * loc + scale**2)
    return mean, np.sqrt(np.maximum(var, 0.0))


def ridall_natural_moments_from_unconstrained(loc, scale, teams):
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)
    n_teams = len(teams)
    slices = ridall_parameter_slices(n_teams)

    attack_mean, attack_sd = exp_positive_moments(loc[slices["attack"]], scale[slices["attack"]])
    defense_free_mean, defense_free_sd = exp_positive_moments(
        loc[slices["defense_free"]],
        scale[slices["defense_free"]],
    )
    hga_mean, hga_sd = exp_positive_moments(
        loc[slices["home_advantage"]],
        scale[slices["home_advantage"]],
    )

    defense_mean = np.concatenate([[1.0], np.ravel(defense_free_mean)])
    defense_sd = np.concatenate([[0.0], np.ravel(defense_free_sd)])
    mean = np.concatenate([np.ravel(attack_mean), defense_mean, [float(hga_mean)]])
    sd = np.concatenate([np.ravel(attack_sd), defense_sd, [float(hga_sd)]])
    return mean, sd


def ridall_natural_summary_frame(loc, scale, teams, source="variational_approximation"):
    mean, sd = ridall_natural_moments_from_unconstrained(loc, scale, teams)
    return pd.DataFrame(
        {
            "parameter": make_ridall_natural_param_names(teams),
            "mean": mean,
            "sd": sd,
            "source": source,
        }
    )


def plot_ridall_selected_dims(
    single_means,
    single_stds,
    multi_means,
    multi_stds,
    dims,
    labels,
    title_prefix="",
    reference_means=None,
    reference_stds=None,
    iteration_stride=1,
    reference_window=False,
    mean_window_sd=3.0,
    std_window=(0.6, 1.6),
):
    import matplotlib.pyplot as plt

    def _as_ref_list(refs, default_label):
        if isinstance(refs, (int, float, np.floating)):
            return [(float(refs), default_label)]
        return refs

    fig, axs = plt.subplots(len(dims), 2, figsize=(16, 4.0 * len(dims)), squeeze=False)
    iteration_stride = max(1, int(iteration_stride))
    x = (
        np.arange(single_means.shape[1])
        if iteration_stride == 1
        else (np.arange(single_means.shape[1]) + 1) * iteration_stride
    )

    for row, (dim, label) in enumerate(zip(dims, labels)):
        mean_refs, std_refs = [], []
        for values, color, run_label in (
            (single_means[:, :, dim], "blue", "1 MC sample"),
            (multi_means[:, :, dim], "green", "100 MC samples"),
        ):
            mean = values.mean(axis=0)
            sd = values.std(axis=0)
            axs[row, 0].plot(x, mean, color=color, label=run_label)
            axs[row, 0].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2)

        for values, color, run_label in (
            (single_stds[:, :, dim], "blue", "1 MC sample"),
            (multi_stds[:, :, dim], "green", "100 MC samples"),
        ):
            mean = values.mean(axis=0)
            sd = values.std(axis=0)
            axs[row, 1].plot(x, mean, color=color, label=run_label)
            axs[row, 1].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2)

        if reference_means and label in reference_means:
            for ref_idx, (value, ref_label) in enumerate(
                _as_ref_list(reference_means[label], "best variational approximation")
            ):
                mean_refs.append(float(value))
                axs[row, 0].axhline(
                    value,
                    color="red",
                    linestyle="--" if ref_idx == 0 else ":",
                    label=ref_label,
                )

        if reference_stds and label in reference_stds:
            for ref_idx, (value, ref_label) in enumerate(
                _as_ref_list(reference_stds[label], "best variational approximation")
            ):
                std_refs.append(float(value))
                axs[row, 1].axhline(
                    value,
                    color="red",
                    linestyle="--" if ref_idx == 0 else ":",
                    label=ref_label,
                )

        if reference_window and mean_refs and std_refs:
            ref_sd = max(max(std_refs), 1e-12)
            axs[row, 0].set_ylim(
                min(mean_refs) - mean_window_sd * ref_sd,
                max(mean_refs) + mean_window_sd * ref_sd,
            )
        if reference_window and std_refs:
            axs[row, 1].set_ylim(
                max(min(std_refs) * std_window[0], 0.0),
                max(std_refs) * std_window[1],
            )

        axs[row, 0].set_title(f"{title_prefix}{label}: variational mean")
        axs[row, 1].set_title(f"{title_prefix}{label}: variational std")
        axs[row, 0].set_xlabel("Iteration")
        axs[row, 1].set_xlabel("Iteration")
        axs[row, 0].grid()
        axs[row, 1].grid()
        axs[row, 0].legend()
        axs[row, 1].legend()

    plt.tight_layout()
    plt.show()


def plot_ridall_selected_dims_zoomed(*args, **kwargs):
    kwargs.setdefault("reference_window", True)
    return plot_ridall_selected_dims(*args, **kwargs)


def ridall_numpyro_model_args(data):
    import jax.numpy as jnp

    return (
        jnp.asarray(data["home_team"]),
        jnp.asarray(data["away_team"]),
        jnp.asarray(data["home_goals"]),
        jnp.asarray(data["away_goals"]),
    )


def make_ridall_fixed_diagonal_guide(mean, sd, n_teams):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    mean = jnp.asarray(mean)
    sd = jnp.maximum(jnp.asarray(sd), 1e-8)
    slices = ridall_parameter_slices(n_teams)

    def guide(*args, **kwargs):
        del args, kwargs
        numpyro.sample(
            "attack",
            dist.TransformedDistribution(
                dist.Normal(mean[slices["attack"]], sd[slices["attack"]]),
                dist.transforms.ExpTransform(),
            ).to_event(1),
        )
        numpyro.sample(
            "defense_free",
            dist.TransformedDistribution(
                dist.Normal(mean[slices["defense_free"]], sd[slices["defense_free"]]),
                dist.transforms.ExpTransform(),
            ).to_event(1),
        )
        numpyro.sample(
            "home_advantage",
            dist.TransformedDistribution(
                dist.Normal(mean[slices["home_advantage"]], sd[slices["home_advantage"]]),
                dist.transforms.ExpTransform(),
            ),
        )

    return guide
