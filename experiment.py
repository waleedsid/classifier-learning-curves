"""DASC 5420 experiment, packaged for reproducible command line execution."""
import argparse
import json
import platform
import sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



def diminishing_returns(sizes, scores, threshold=0.005):
    """Original rule: next two signed gains below threshold, including declines."""
    if len(sizes) != len(scores):
        raise ValueError("Sizes and scores must align")
    gains = np.diff(scores)
    for j in range(len(gains)-1):
        if gains[j] < threshold and gains[j+1] < threshold:
            return int(sizes[j])
    return None


def run(output="results", jobs=1):
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X, y = digits.data, digits.target
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_fracs = np.linspace(0.1, 1.0, 10)

    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0))
        ]),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    rows = []
    for model_name, model in models.items():
        for metric, scoring in [("Accuracy", "accuracy"), ("Macro F1", "f1_macro")]:
            sizes, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_fracs, cv=cv, scoring=scoring,
                shuffle=True, random_state=42, n_jobs=jobs, error_score="raise"
            )
            for i, n in enumerate(sizes):
                rows.append({
                    "Model": model_name, "Metric": metric, "Training Size": int(n),
                    "Training Fraction": float(train_fracs[i]),
                    "Train Mean": train_scores[i].mean(),
                    "Train SD": train_scores[i].std(ddof=1),
                    "Validation Mean": val_scores[i].mean(),
                    "Validation SD": val_scores[i].std(ddof=1),
                    "Generalization Gap": train_scores[i].mean() - val_scores[i].mean()
                })

    results = pd.DataFrame(rows)
    results.to_csv(OUT/"learning_curve_results.csv", index=False)

    # Diminishing returns: first training size after which the next two
    # validation-accuracy gains are each below 0.005 (0.5 percentage points).
    plateau_rows = []
    for model_name in models:
        d = results[(results.Model == model_name) & (results.Metric == "Accuracy")].sort_values("Training Size")
        vals, sizes = d["Validation Mean"].to_numpy(), d["Training Size"].to_numpy()
        gains = np.diff(vals)
        plateau = diminishing_returns(sizes, vals)
        plateau_rows.append({"Model": model_name, "Diminishing Returns Size": plateau})
    pd.DataFrame(plateau_rows).to_csv(OUT/"diminishing_returns.csv", index=False)

    # Core plots
    for metric, fname, ylabel in [
        ("Accuracy", "accuracy_learning_curves.png", "Mean cross-validated accuracy"),
        ("Macro F1", "macro_f1_learning_curves.png", "Mean cross-validated macro F1")
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for model_name in models:
            d = results[(results.Model == model_name) & (results.Metric == metric)].sort_values("Training Size")
            ax.errorbar(d["Training Size"], d["Validation Mean"], yerr=d["Validation SD"],
                        marker="o", capsize=3, label=model_name)
        ax.set_xlabel("Number of training observations")
        ax.set_ylabel(ylabel)
        ax.set_title(metric + " learning curves")
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(OUT/fname, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model_name in models:
        d = results[(results.Model == model_name) & (results.Metric == "Accuracy")].sort_values("Training Size")
        ax.plot(d["Training Size"], d["Generalization Gap"], marker="o", label=model_name)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Number of training observations")
    ax.set_ylabel("Training accuracy - validation accuracy")
    ax.set_title("Generalization gap across training sizes")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT/"generalization_gap.png", dpi=220)
    plt.close(fig)

    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                   "matplotlib": matplotlib.__version__, "seed": 42, "folds": 5}
    (OUT/"environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    print(results[results["Training Size"] == results["Training Size"].max()].to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs == 0:
        parser.error("--jobs cannot be zero")
    run(args.output, args.jobs)


if __name__ == "__main__":
    main()
