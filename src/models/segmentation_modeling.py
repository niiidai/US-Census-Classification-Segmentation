"""Production segmentation modeling for the engineered Census dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def load_segmentation_artifacts(artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature matrix and engineered data."""
    X = pd.read_pickle(artifact_dir / "X.pkl")
    data = pd.read_pickle(artifact_dir / "engineered_data.pkl")
    return X, data


def sample_for_segmentation(
    X: pd.DataFrame,
    data: pd.DataFrame,
    sample_size: int = 20000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample rows for visualization and expensive clustering methods."""
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(X.shape[0], size=min(sample_size, X.shape[0]), replace=False)
    return X.iloc[sample_idx], data.iloc[sample_idx].copy()


def make_svd_projection(
    X_sample: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, TruncatedSVD]:
    """Create an SVD projection for high-dimensional encoded data."""
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    projection = svd.fit_transform(X_sample)
    return projection, svd


def build_plot_df(X_sample: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Create 2D and 20D SVD representations for cluster visualization/comparison."""
    X_2d, svd_2d = make_svd_projection(X_sample, n_components=2)
    X_20d, svd_20 = make_svd_projection(X_sample, n_components=20)
    plot_df = pd.DataFrame({"component_1": X_2d[:, 0], "component_2": X_2d[:, 1]})
    print("2D explained variance ratio:", svd_2d.explained_variance_ratio_)
    print("2D total explained variance:", svd_2d.explained_variance_ratio_.sum())
    print("20D total explained variance:", svd_20.explained_variance_ratio_.sum())
    return plot_df, X_20d


def fit_sample_clusters(
    X_sample: pd.DataFrame,
    X_sample_20d: np.ndarray,
    k: int = 5,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Fit several clustering methods on the sample."""
    labels: dict[str, np.ndarray] = {}
    labels["kmeans"] = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X_sample)
    labels["gmm"] = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=random_state,
    ).fit_predict(X_sample_20d)
    labels["birch"] = Birch(n_clusters=k, threshold=0.5).fit_predict(X_sample_20d)
    labels["agglomerative"] = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_sample_20d)
    return labels


def plot_clusters(plot_df: pd.DataFrame, label_col: str, title: str, output_path: Path | None = None) -> None:
    """Plot cluster assignments in a 2D projection."""
    plt.figure(figsize=(8, 6))
    for segment in sorted(plot_df[label_col].unique()):
        subset = plot_df[plot_df[label_col] == segment]
        plt.scatter(
            subset["component_1"],
            subset["component_2"],
            s=8,
            alpha=0.4,
            label=f"Segment {segment}",
        )
    plt.title(title)
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def compare_cluster_methods(labels: dict[str, np.ndarray], X_sample_20d: np.ndarray) -> pd.DataFrame:
    """Compare clustering methods using silhouette score."""
    rows = []
    for method_name, method_labels in labels.items():
        score = silhouette_score(X_sample_20d, method_labels)
        rows.append({
            "method": method_name,
            "silhouette_score": score,
            "num_clusters": len(np.unique(method_labels)),
        })
    return pd.DataFrame(rows).sort_values("silhouette_score", ascending=False)


def fit_final_kmeans(X: pd.DataFrame, data: pd.DataFrame, k: int = 5, random_state: int = 42) -> tuple[pd.DataFrame, KMeans]:
    """Fit final K-means on the full feature matrix and attach segment labels."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    data = data.copy()
    data["segment"] = model.fit_predict(X)
    return data, model


def save_segment_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Save segment size and high-income-rate plots."""
    segment_counts = data["segment"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(segment_counts.index.astype(str), segment_counts.values, edgecolor="black")
    plt.title("Customer Segment Size")
    plt.xlabel("Segment")
    plt.ylabel("Number of Records")
    for i, value in enumerate(segment_counts.values):
        plt.text(i, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "segment_size.png", dpi=150)
    plt.close()

    if "label" in data.columns:
        segment_income_rate = data.groupby("segment")["label"].mean().sort_index()
        plt.figure(figsize=(8, 5))
        plt.bar(segment_income_rate.index.astype(str), segment_income_rate.values, edgecolor="black")
        plt.title("High-Income Rate by Segment")
        plt.xlabel("Segment")
        plt.ylabel("High-Income Rate")
        for i, value in enumerate(segment_income_rate.values):
            plt.text(i, value, f"{value:.1%}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(output_dir / "segment_income_rate.png", dpi=150)
        plt.close()


def make_segment_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Create a numerical segment profile table."""
    profile_cols = [
        "age",
        "label",
        "has_wage_per_hour",
        "has_capital_gains",
        "has_capital_losses",
        "has_dividends_from_stocks",
        "log_wage_per_hour",
        "log_capital_gains",
        "log_capital_losses",
        "log_dividends_from_stocks",
        "weeks_worked_in_year",
        "worked_none_year",
        "worked_full_year",
    ]
    available_cols = [col for col in profile_cols if col in data.columns]
    return data.groupby("segment")[available_cols].mean().round(3)


def top_category_by_segment(data: pd.DataFrame, segment_col: str, feature_col: str) -> pd.DataFrame:
    """Return the most common feature category in each segment."""
    return (
        data.groupby(segment_col)[feature_col]
        .agg(lambda x: x.value_counts(dropna=False).index[0])
        .reset_index()
        .rename(columns={feature_col: f"top_{feature_col}"})
    )


def make_categorical_profiles(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create top-category profile tables for available categorical columns."""
    categorical_cols = [
        "education",
        "education_grouped",
        "weeks_worked_bucket",
        "class_of_worker",
        "class_of_worker_grouped",
        "major_industry_code",
        "major_occupation_code",
        "marital_stat",
        "marital_stat_grouped",
        "full_or_part_time_employment_stat",
        "full_or_part_time_employment_stat_grouped",
        "member_of_a_labor_union",
    ]
    return {
        col: top_category_by_segment(data, "segment", col)
        for col in categorical_cols
        if col in data.columns
    }


def save_categorical_profiles(profiles: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save categorical profile tables."""
    profile_dir = output_dir / "categorical_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    for col, profile in profiles.items():
        profile.to_csv(profile_dir / f"{col}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Census customer segmentation.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("../artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("../model_artifacts"))
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, data = load_segmentation_artifacts(args.artifact_dir)
    X_sample, _ = sample_for_segmentation(X, data, sample_size=args.sample_size)
    plot_df, X_sample_20d = build_plot_df(X_sample)
    labels = fit_sample_clusters(X_sample, X_sample_20d, k=args.k)

    for method_name, method_labels in labels.items():
        col = f"{method_name}_segment"
        plot_df[col] = method_labels
        plot_clusters(
            plot_df,
            col,
            f"Customer Segments: {method_name}",
            args.output_dir / f"{method_name}_segments.png",
        )

    comparison = compare_cluster_methods(labels, X_sample_20d)
    comparison.to_csv(args.output_dir / "cluster_method_comparison.csv", index=False)

    segmented_data, final_model = fit_final_kmeans(X, data, k=args.k)
    pd.to_pickle(segmented_data, args.output_dir / "segmented_data.pkl")
    joblib.dump(final_model, args.output_dir / "final_kmeans_model.joblib")

    save_segment_plots(segmented_data, args.output_dir)
    make_segment_profile(segmented_data).to_csv(args.output_dir / "segment_profile.csv")
    save_categorical_profiles(make_categorical_profiles(segmented_data), args.output_dir)

    print("Cluster comparison:")
    print(comparison)
    print(f"Segmentation artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
