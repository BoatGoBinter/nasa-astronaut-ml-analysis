"""
model_representations.py

Standalone script version of the modeling portion of
`nasa_astronaut_model_testing.ipynb`.

Reproduces only the KEY MODEL REPRESENTATION plots (regression
coefficients, bootstrap stability, partial-effect curves, PCA, and
LASSO feature-importance bars) needed to answer the research question:

    "Can undergraduate major or military branch predict an astronaut's
    total space-flight hours or number of missions?"

Purely descriptive/EDA charts (branch counts, raw group averages,
STEM-by-branch stacked bars) are intentionally left out — this script
focuses on the fitted-model outputs.

Usage:
    python model_representations.py [path_to_merged_csv]

Default input file: merged_astronauts_data.csv
Output: all figures are saved as PNGs into ./figures/
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "merged_astronauts_data.csv"
OUTPUT_DIR = "figures"

NASA_BLUE = "#0B3D91"
NASA_GREEN = "#00A651"

N_BOOTSTRAP_RUNS = 100
RANDOM_SEED = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------
# 1. Load + clean data
# ---------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)

    # Keep only astronauts who actually flew
    df_analysis = df[df["Space Flight (hr)"] > 0].copy()

    # Drop duplicate/renamed gender + flight-count columns from the merge
    cols_to_drop = [c for c in ["Gender_y", "Flights", "Total Flights"] if c in df_analysis.columns]
    df_analysis = df_analysis.drop(columns=cols_to_drop).rename(columns={"Gender_x": "Gender"})

    return df_analysis


def simplify_branch(b):
    if pd.isna(b):
        return "Civilian"
    s = str(b)
    if "Air Force" in s:
        return "Air Force"
    if "Army" in s:
        return "Army"
    if "Navy" in s:
        return "Navy"
    if "Marine" in s:
        return "Marines"
    if "Coast Guard" in s:
        return "Coast Guard"
    return "Other Military"


def recode_branch_raw(b):
    # Fold Naval Reserve into Navy before simplifying
    if pd.isna(b):
        return b
    s = str(b)
    if "Naval Reserve" in s:
        return "US Navy"
    return s


def is_stem_major(m):
    if pd.isna(m):
        return np.nan
    s = str(m)
    stem_keywords = [
        "Engineering", "Physics", "Math", "Science", "Astronomy", "Chemistry",
        "Biology", "Geology", "Geophysics", "Mechanics", "Astronautics",
        "Aeronautics", "Aerospace", "Computer", "Electrical", "Mechanical",
        "Applied", "Operations Research",
    ]
    return int(any(k in s for k in stem_keywords))


def engineer_features(df_analysis):
    df_feat = df_analysis.copy()
    df_feat["Military Branch"] = df_feat["Military Branch"].apply(recode_branch_raw)
    df_feat["BranchSimple"] = df_feat["Military Branch"].apply(simplify_branch)
    df_feat["UndergradSTEM"] = df_feat["Undergraduate Major"].apply(is_stem_major)
    return df_feat


# ---------------------------------------------------------------------
# 2. Full model: UndergradSTEM + BranchSimple + Gender + Year
# ---------------------------------------------------------------------

def build_full_model_data(df_feat):
    model2 = df_feat[
        ["Space Flight (hr)", "Space Flights", "UndergradSTEM", "BranchSimple", "Gender", "Year"]
    ].dropna()

    X2 = pd.get_dummies(model2[["UndergradSTEM", "BranchSimple", "Gender", "Year"]], drop_first=True)
    y_hr2 = model2["Space Flight (hr)"]
    y_fl2 = model2["Space Flights"]

    return model2, X2, y_hr2, y_fl2


def fig_bootstrap_coef_boxplots(X2, y_hr2, y_fl2):
    """Bootstrap the linear regression 100x and show coefficient stability (excl. Year)."""
    feature_names = list(X2.columns)
    rows = []

    for run in range(N_BOOTSTRAP_RUNS):
        X_train, _, y_hr_train, _, y_fl_train, _ = train_test_split(
            X2, y_hr2, y_fl2, test_size=0.30, random_state=run
        )
        model_hr = LinearRegression().fit(X_train.values, y_hr_train)
        model_fl = LinearRegression().fit(X_train.values, y_fl_train)

        for j, feat in enumerate(feature_names):
            rows.append({"Run": run, "Feature": feat, "Coefficient": model_hr.coef_[j], "Outcome": "Space Flight (hr)"})
            rows.append({"Run": run, "Feature": feat, "Coefficient": model_fl.coef_[j], "Outcome": "Space Flights"})

    coef_runs = pd.DataFrame(rows)

    feature_order = [f for f in X2.columns if f != "Year"]
    outcome_list = ["Space Flight (hr)", "Space Flights"]
    color_list = [NASA_BLUE, NASA_GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for idx, outcome in enumerate(outcome_list):
        ax = axes[idx]
        color = color_list[idx]
        sub = coef_runs[(coef_runs["Outcome"] == outcome) & (coef_runs["Feature"] != "Year")]

        data = [sub[sub["Feature"] == feat]["Coefficient"].values for feat in feature_order]
        box = ax.boxplot(data, vert=True, patch_artist=True, tick_labels=feature_order)
        for b in box["boxes"]:
            b.set_facecolor(color)
            b.set_alpha(0.6)

        ax.axhline(0, color="grey", linewidth=1)
        ax.set_title(outcome)
        ax.set_ylabel("Coefficient")
        ax.set_xticklabels(feature_order, rotation=90)

    savefig("fig4_coefficients_boxplots_no_year.png")

    return coef_runs


def fig_partial_effects(X2, model2, linreg_hr):
    """Partial-effect curves: STEM, Gender, Branch, Year (holding others at their mean)."""
    feature_mean = X2.mean()

    def predict_with(**changes):
        x = feature_mean.copy()
        for k, v in changes.items():
            x[k] = v
        return linreg_hr.predict([x.values])[0]

    # --- STEM effect ---
    vals = [0, 1]
    preds = [predict_with(UndergradSTEM=v) for v in vals]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(vals, preds, marker="o")
    ax.set_xticks(vals)
    ax.set_xticklabels(["Non-STEM", "STEM"])
    ax.set_ylabel("Predicted Space Flight Hours")
    ax.set_xlabel("Undergrad Major Type")
    ax.set_title("UndergradSTEM Effect")
    savefig("fig5_effect_undergradstem.png")

    # --- Gender effect ---
    gender_col = [c for c in X2.columns if c.startswith("Gender_")][0]
    preds = [predict_with(**{gender_col: v}) for v in vals]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(vals, preds, marker="o")
    ax.set_xticks(vals)
    ax.set_xticklabels(["Baseline gender", gender_col.split("_")[1]])
    ax.set_ylabel("Predicted Space Flight Hours")
    ax.set_xlabel("Gender")
    ax.set_title("Gender Effect")
    savefig("fig6_effect_gender.png")

    # --- Branch effect ---
    branch_dummy_cols = [c for c in X2.columns if c.startswith("BranchSimple_")]
    branches = ["BASELINE"] + [c.split("BranchSimple_")[1] for c in branch_dummy_cols]
    preds = []
    for b in branches:
        x = feature_mean.copy()
        for col in branch_dummy_cols:
            x[col] = 0
        if b != "BASELINE":
            x["BranchSimple_" + b] = 1
        preds.append(linreg_hr.predict([x.values])[0])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(len(branches)), preds, marker="o")
    ax.set_xticks(range(len(branches)))
    ax.set_xticklabels(branches, rotation=45, ha="right")
    ax.set_ylabel("Predicted Space Flight Hours")
    ax.set_xlabel("BranchSimple")
    ax.set_title("Branch Effect")
    savefig("fig7_effect_branch.png")

    # --- Year effect ---
    year_min, year_max = int(model2["Year"].min()), int(model2["Year"].max())
    years = np.linspace(year_min, year_max, 20)
    preds = [predict_with(Year=y) for y in years]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(years, preds, marker="o")
    ax.set_xlabel("Astronaut Selection Year")
    ax.set_ylabel("Predicted Space Flight Hours")
    ax.set_title("Year Effect")
    savefig("fig8_effect_year.png")


# ---------------------------------------------------------------------
# 3. PCA: compress background into 2 components
# ---------------------------------------------------------------------

def fig_pca(df_feat):
    cols_for_pca = ["Undergraduate Major", "Graduate Major", "Military Branch"]
    pca_df = df_feat.dropna(subset=cols_for_pca + ["Space Flight (hr)"]).copy()

    X_hd = pd.get_dummies(pca_df[cols_for_pca])
    X_scaled = StandardScaler().fit_transform(X_hd)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pc1 = X_pca[:, 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pc1, pca_df["Space Flight (hr)"], alpha=0.7)
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Space Flight Hours")
    ax.set_title("Background PC1 vs Space Flight Hours")
    savefig("fig9_pca_pc1_vs_hours.png")


# ---------------------------------------------------------------------
# 4. Majors-only model: bootstrap coefficient stability
# ---------------------------------------------------------------------

def fig_majors_bootstrap(df_feat):
    df_major = df_feat.dropna(subset=["Undergraduate Major", "Space Flight (hr)", "Space Flights"]).copy()

    major_counts = df_major["Undergraduate Major"].value_counts()
    keep_majors = major_counts[major_counts >= 5].index.tolist()
    df_major = df_major[df_major["Undergraduate Major"].isin(keep_majors)].copy()

    X_maj = pd.get_dummies(df_major["Undergraduate Major"], drop_first=True)
    y_hr_maj = df_major["Space Flight (hr)"]
    y_fl_maj = df_major["Space Flights"]
    feature_names_maj = list(X_maj.columns)

    rows_maj = []
    for run in range(N_BOOTSTRAP_RUNS):
        X_train, _, y_hr_train, _, y_fl_train, _ = train_test_split(
            X_maj, y_hr_maj, y_fl_maj, test_size=0.30, random_state=run
        )
        model_hr = LinearRegression().fit(X_train.values, y_hr_train)
        model_fl = LinearRegression().fit(X_train.values, y_fl_train)

        for j, feat in enumerate(feature_names_maj):
            rows_maj.append({"Run": run, "Feature": feat, "Coefficient": model_hr.coef_[j], "Outcome": "Space Flight (hr)"})
            rows_maj.append({"Run": run, "Feature": feat, "Coefficient": model_fl.coef_[j], "Outcome": "Space Flights"})

    coef_runs_maj = pd.DataFrame(rows_maj)

    outcome_list = ["Space Flight (hr)", "Space Flights"]
    color_list = [NASA_BLUE, NASA_GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for idx, outcome in enumerate(outcome_list):
        ax = axes[idx]
        color = color_list[idx]
        sub = coef_runs_maj[coef_runs_maj["Outcome"] == outcome]

        data = [sub[sub["Feature"] == feat]["Coefficient"].values for feat in feature_names_maj]
        box = ax.boxplot(data, vert=True, patch_artist=True, tick_labels=feature_names_maj)
        for b in box["boxes"]:
            b.set_facecolor(color)
            b.set_alpha(0.6)

        ax.axhline(0, color="grey", linewidth=1)
        ax.set_title(outcome)
        ax.set_ylabel("Coefficient")
        ax.set_xticklabels(feature_names_maj, rotation=90)

    savefig("fig_majors_coefficients_boxplots_100runs.png")


# ---------------------------------------------------------------------
# 5. LASSO: which detailed majors/branches survive regularization?
# ---------------------------------------------------------------------

def fig_lasso_hours(df_feat):
    """LASSO on the full detailed design matrix (majors + branch + gender + year)."""
    cols = ["Space Flight (hr)", "Space Flights", "Undergraduate Major", "Graduate Major", "Military Branch", "Gender", "Year"]
    df_lasso = df_feat[cols].dropna().copy()

    X_cat = df_lasso[["Undergraduate Major", "Graduate Major", "Military Branch", "Gender"]]
    X_dummies = pd.get_dummies(X_cat, drop_first=True)
    X = X_dummies.copy()
    X["Year"] = df_lasso["Year"].values

    y_hours = df_lasso["Space Flight (hr)"]

    X_scaled = StandardScaler().fit_transform(X)

    lasso_hours = LassoCV(cv=5, random_state=RANDOM_SEED).fit(X_scaled, y_hours)
    print(f"[LASSO hours] best alpha={lasso_hours.alpha_:.4f}, train R^2={lasso_hours.score(X_scaled, y_hours):.3f}")

    feature_names = list(X.columns)
    lasso_coefs = pd.DataFrame({
        "Feature": feature_names,
        "Coef_hours": lasso_hours.coef_,
    })
    lasso_coefs["Abs_hours"] = lasso_coefs["Coef_hours"].abs()

    hours_nonzero = lasso_coefs[(lasso_coefs["Abs_hours"] >= 1.0) & (lasso_coefs["Feature"] != "Year")].copy()
    hours_nonzero = hours_nonzero.sort_values("Abs_hours", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(hours_nonzero["Feature"], hours_nonzero["Coef_hours"], color=NASA_BLUE)
    ax.axvline(0, color="grey", linewidth=1)
    ax.set_xlabel("LASSO Coefficient (Effect on Space Flight Hours)")
    ax.set_title("Important Predictors of Space Flight Hours (LASSO)")
    savefig("lasso_hours_bar_no_year.png")


def fig_lasso_flights(df_feat):
    """LASSO on majors/branches filtered to groups with n >= 5, predicting number of flights."""
    cols_needed = ["Space Flight (hr)", "Space Flights", "Undergraduate Major", "Graduate Major", "Military Branch", "Gender"]
    df_lasso = df_feat[cols_needed].dropna().copy()

    for col in ["Undergraduate Major", "Graduate Major", "Military Branch"]:
        counts = df_lasso[col].value_counts()
        keep = counts[counts >= 5].index.tolist()
        df_lasso = df_lasso[df_lasso[col].isin(keep)].copy()

    X_cat = df_lasso[["Undergraduate Major", "Graduate Major", "Military Branch", "Gender"]]
    X = pd.get_dummies(X_cat, drop_first=True)
    feature_names = list(X.columns)

    y_flights = df_lasso["Space Flights"]

    X_scaled = StandardScaler().fit_transform(X)
    lasso_flights = LassoCV(cv=5, random_state=RANDOM_SEED).fit(X_scaled, y_flights)
    print(f"[LASSO flights] best alpha={lasso_flights.alpha_:.4f}, train R^2={lasso_flights.score(X_scaled, y_flights):.3f}")

    lasso_coefs = pd.DataFrame({
        "Feature": feature_names,
        "Coef_flights": lasso_flights.coef_,
    })
    lasso_coefs["Abs_flights"] = lasso_coefs["Coef_flights"].abs()

    flights_nonzero = lasso_coefs[lasso_coefs["Abs_flights"] >= 0.05].copy()
    print(f"Number of 'meaningful' predictors for flights (n>=5, no Year): {flights_nonzero.shape[0]}")
    flights_nonzero = flights_nonzero.sort_values("Abs_flights", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    if flights_nonzero.shape[0] > 0:
        ax.barh(flights_nonzero["Feature"], flights_nonzero["Coef_flights"], color=NASA_GREEN)
    else:
        ax.text(
            0.5, 0.5,
            "No majors/branches have a meaningful effect\non number of flights in this LASSO model\n(after requiring n >= 5 per group).",
            ha="center", va="center", fontsize=12, transform=ax.transAxes,
        )
        ax.set_yticks([])

    ax.axvline(0, color="grey", linewidth=1)
    ax.set_xlabel("LASSO Coefficient (Effect on Number of Flights)")
    ax.set_title("Majors & Branches for Number of Flights (n >= 5 per group)")
    savefig("lasso_flights_bar_major_branch_n5.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: could not find input file '{INPUT_CSV}'.")
        print("Usage: python model_representations.py [path_to_merged_csv]")
        sys.exit(1)

    print(f"Loading {INPUT_CSV} ...")
    df_analysis = load_data(INPUT_CSV)
    df_feat = engineer_features(df_analysis)

    print("Fitting full model (UndergradSTEM + BranchSimple + Gender + Year) ...")
    model2, X2, y_hr2, y_fl2 = build_full_model_data(df_feat)
    linreg_hr = LinearRegression().fit(X2.values, y_hr2)

    print(f"Bootstrapping coefficients over {N_BOOTSTRAP_RUNS} train/test splits ...")
    fig_bootstrap_coef_boxplots(X2, y_hr2, y_fl2)

    print("Generating partial-effect plots ...")
    fig_partial_effects(X2, model2, linreg_hr)

    print("Running PCA on background variables ...")
    fig_pca(df_feat)

    print("Bootstrapping majors-only model ...")
    fig_majors_bootstrap(df_feat)

    print("Fitting LASSO models ...")
    fig_lasso_hours(df_feat)
    fig_lasso_flights(df_feat)

    print(f"\nDone. All figures saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
