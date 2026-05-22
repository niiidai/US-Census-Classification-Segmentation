"""Production feature engineering pipeline for the Census income dataset.

This module loads the raw Census Bureau data, applies the feature engineering
steps developed in the notebook, and writes model-ready artifacts to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


MONEY_COLS = [
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
]

NUMERIC_COLS_TO_SCALE = [
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
    "log_wage_per_hour",
    "log_capital_gains",
    "log_capital_losses",
    "log_dividends_from_stocks",
]

COUNTRY_REGION_MAP = {
    "United-States": "north_america",
    "Canada": "north_america",
    "Mexico": "north_america",
    "Outlying-U S (Guam USVI etc)": "north_america",
    "Puerto-Rico": "latin_america",
    "Cuba": "latin_america",
    "Jamaica": "latin_america",
    "Dominican-Republic": "latin_america",
    "Haiti": "latin_america",
    "El-Salvador": "latin_america",
    "Guatemala": "latin_america",
    "Honduras": "latin_america",
    "Nicaragua": "latin_america",
    "Ecuador": "latin_america",
    "Peru": "latin_america",
    "Columbia": "latin_america",
    "Trinadad&Tobago": "latin_america",
    "Panama": "latin_america",
    "England": "europe",
    "Germany": "europe",
    "Greece": "europe",
    "Holand-Netherlands": "europe",
    "Hungary": "europe",
    "Ireland": "europe",
    "Italy": "europe",
    "Poland": "europe",
    "Portugal": "europe",
    "Scotland": "europe",
    "Yugoslavia": "europe",
    "France": "europe",
    "China": "asia",
    "India": "asia",
    "Iran": "asia",
    "Japan": "asia",
    "Cambodia": "asia",
    "Laos": "asia",
    "Philippines": "asia",
    "Taiwan": "asia",
    "Thailand": "asia",
    "Vietnam": "asia",
    "Hong Kong": "asia",
    "South Korea": "asia",
}

EDUCATION_MAP = {
    "Children": "children",
    "High school graduate": "high_school",
    "Bachelors degree(BA AB BS)": "bachelors",
    "Less than 1st grade": "less_than_high_school",
    "1st 2nd 3rd or 4th grade": "less_than_high_school",
    "5th or 6th grade": "less_than_high_school",
    "7th and 8th grade": "less_than_high_school",
    "9th grade": "less_than_high_school",
    "10th grade": "less_than_high_school",
    "11th grade": "less_than_high_school",
    "12th grade no diploma": "less_than_high_school",
    "Some college but no degree": "some_college_or_associate",
    "Associates degree-occup /vocational": "some_college_or_associate",
    "Associates degree-academic program": "some_college_or_associate",
    "Masters degree(MA MS MEng MEd MSW MBA)": "graduate_degree",
    "Prof school degree (MD DDS DVM LLB JD)": "graduate_degree",
    "Doctorate degree(PhD EdD)": "graduate_degree",
}

FAMILY_UNDER_18_MAP = {
    "Not in universe": "not_in_universe",
    "Neither parent present": "no_parent_present",
    "Mother only present": "one_parent_present",
    "Father only present": "one_parent_present",
    "Both parents present": "both_parents_present",
}

CITIZENSHIP_MAP = {
    "Native- Born in the United States": "native_us_born",
    "Foreign born- U S citizen by naturalization": "naturalized_citizen",
    "Foreign born- Not a citizen of U S": "non_citizen",
    "Native- Born abroad of American Parent(s)": "native_born_abroad",
    "Native- Born in Puerto Rico or U S Outlying": "native_us_territory",
}

MIGRATION_REGION_MAP = {
    "?": "unknown",
    "Nonmover": "nonmover",
    "Same county": "local_mover",
    "Different county same state": "local_mover",
    "Different state same division": "long_distance_domestic_mover",
    "Different division same region": "long_distance_domestic_mover",
    "Different region": "long_distance_domestic_mover",
    "Abroad": "international_mover",
    "Not in universe": "not_applicable",
}


def clean_column_name(name: str) -> str:
    """Standardize one column name."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def clean_dummy_column_names(dummy_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dummy column names after one-hot encoding."""
    dummy_df = dummy_df.copy()
    dummy_df.columns = (
        dummy_df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("&", "and", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("+", "plus", regex=False)
    )
    return dummy_df


def load_census_data(data_path: Path, columns_path: Path) -> pd.DataFrame:
    """Load raw Census data with standardized column names."""
    with columns_path.open("r", encoding="utf-8") as f:
        columns = [clean_column_name(line) for line in f.read().splitlines() if line.strip()]

    return pd.read_csv(
        data_path,
        header=None,
        names=columns,
        skipinitialspace=True,
    )


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic row cleaning and binary target mappings."""
    df = df.copy().drop_duplicates(keep="first")
    df["hispanic_origin"] = df["hispanic_origin"].fillna("?")
    df["label"] = df["label"].map(lambda x: 1 if str(x).strip() == "50000+." else 0)
    df["sex"] = df["sex"].map(lambda x: 1 if str(x).strip() == "Male" else 0)
    df["year"] = df["year"].map(lambda x: 1 if str(x).strip() == "94" else 0)
    return df


def add_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot age bucket features."""
    df = df.copy()
    age_bucket = pd.cut(
        df["age"],
        bins=[-1, 17, 24, 34, 44, 54, 64, np.inf],
        labels=["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        include_lowest=True,
    )
    dummies = pd.get_dummies(age_bucket, prefix="age_bucket", dtype=int)
    dummies = clean_dummy_column_names(dummies)
    return pd.concat([df, dummies], axis=1)


def add_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create indicator and log-transformed features for money columns."""
    df = df.copy()
    for col in MONEY_COLS:
        if col in df.columns:
            df[f"has_{col}"] = (df[col] > 0).astype(int)
            df[f"log_{col}"] = np.log1p(df[col])
    return df


def add_weeks_worked_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create weeks-worked indicator and bucket features."""
    df = df.copy()
    weeks_col = "weeks_worked_in_year"
    df["worked_none_year"] = (df[weeks_col] == 0).astype(int)
    df["worked_full_year"] = (df[weeks_col] == 52).astype(int)
    bucket = pd.cut(
        df[weeks_col],
        bins=[0, 13, 26, 39, 51],
        labels=["1-13 weeks", "14-26 weeks", "27-39 weeks", "40-51 weeks"],
    )
    dummies = pd.get_dummies(bucket, prefix="weeks_worked", dtype=int)
    dummies = clean_dummy_column_names(dummies)
    return pd.concat([df, dummies], axis=1)


def add_industry_occupation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert detailed recodes to strings and one-hot encode major groups."""
    df = df.copy()
    for col in ["detailed_industry_recode", "detailed_occupation_recode"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    major_cols = ["major_industry_code", "major_occupation_code"]
    dummies = pd.get_dummies(df[major_cols], prefix=major_cols, dtype=int)
    dummies = clean_dummy_column_names(dummies)
    return pd.concat([df, dummies], axis=1)


def clean_value(val: object) -> str:
    return str(val).strip()


def group_class_of_worker(val: object) -> str:
    val = clean_value(val)
    if val == "Not in universe":
        return "not_applicable"
    if val == "Private":
        return "private"
    if "Self-employed" in val:
        return "self_employed"
    if "government" in val.lower():
        return "government"
    if val in {"Never worked", "Without pay"}:
        return "other_non_working"
    return "other"


def group_marital_status(val: object) -> str:
    val = clean_value(val)
    if "Married" in val:
        return "married"
    if val == "Never married":
        return "never_married"
    if val == "Divorced":
        return "no_spouse_divorced"
    return "no_spouse_other"


def group_race(val: object) -> str:
    return {"White": "white", "Black": "black", "Asian or Pacific Islander": "asian"}.get(
        clean_value(val), "other"
    )


def group_hispanic_origin(val: object) -> str:
    val = clean_value(val)
    if val == "?":
        return "unknown"
    if val == "All other":
        return "not_hispanic"
    return "hispanic"


def group_country_of_birth(val: object) -> str:
    val = clean_value(val)
    if val in {"?", "Not in universe"}:
        return "unknown"
    return COUNTRY_REGION_MAP.get(val, "other")


def group_full_or_part_time_employment(val: object) -> str:
    val = clean_value(val)
    if val == "Not in labor force":
        return "not_in_labor_force"
    if val == "Children or Armed Forces":
        return "children_or_af"
    if val == "Unemployed full-time":
        return "unemployed_full_time"
    if val in {
        "PT for econ reasons usually FT",
        "PT for econ reasons usually PT",
        "PT for non-econ reasons usually FT",
        "Unemployed part- time",
    }:
        return "part_time_or_unemployed_part_time"
    if val == "Full-time schedules":
        return "full_time"
    return "other"


def group_detailed_household_summary(val: object) -> str:
    val = clean_value(val).lower()
    if "spouse" in val:
        return "spouse_of_householder"
    if val == "householder":
        return "householder"
    if "child" in val:
        return "child"
    return "other"


def add_grouped_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create grouped categorical columns."""
    grouping_map: dict[str, Callable[[object], str]] = {
        "class_of_worker": group_class_of_worker,
        "marital_stat": group_marital_status,
        "race": group_race,
        "hispanic_origin": group_hispanic_origin,
        "education": lambda x: EDUCATION_MAP.get(clean_value(x), "other"),
        "family_members_under_18": lambda x: FAMILY_UNDER_18_MAP.get(clean_value(x), "other"),
        "country_of_birth_self": group_country_of_birth,
        "full_or_part_time_employment_stat": group_full_or_part_time_employment,
        "detailed_household_summary_in_household": group_detailed_household_summary,
        "citizenship": lambda x: CITIZENSHIP_MAP.get(clean_value(x), "other"),
        "migration_code_change_in_reg": lambda x: MIGRATION_REGION_MAP.get(clean_value(x), "other"),
    }

    df = df.copy()
    for col, group_func in grouping_map.items():
        if col in df.columns:
            df[f"{col}_grouped"] = df[col].apply(group_func)
    return df


def one_hot_encode_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    drop_original: bool = True,
) -> pd.DataFrame:
    """One-hot encode selected categorical features."""
    df = df.copy()
    existing_cols = [col for col in feature_cols if col in df.columns]
    if not existing_cols:
        return df

    dummies = pd.get_dummies(df[existing_cols].astype(str), prefix=existing_cols, dtype=int)
    dummies = clean_dummy_column_names(dummies)
    df = pd.concat([df, dummies], axis=1)
    if drop_original:
        df = df.drop(columns=existing_cols)
    return df


def add_scaled_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """Add standardized numeric columns for linear models."""
    df = df.copy()
    cols = [col for col in NUMERIC_COLS_TO_SCALE if col in df.columns]
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[cols]),
        columns=[f"scaled_{col}" for col in cols],
        index=df.index,
    )
    return pd.concat([df, scaled], axis=1), scaler, scaled.columns.tolist()


def build_features(raw_df: pd.DataFrame) -> dict[str, object]:
    """Run full feature engineering and return model artifacts."""
    data = basic_cleaning(raw_df)
    weight = data[["weight"]].copy()
    data = data.drop(columns=["weight"])

    data = add_age_bucket(data)
    data = add_money_features(data)
    data = add_weeks_worked_features(data)
    data = add_industry_occupation_features(data)
    data = add_grouped_categorical_features(data)

    grouped_cols = [col for col in data.columns if col.endswith("_grouped")]
    categorical_cols = grouped_cols + [
        "own_business_or_self_employed",
        "veterans_benefits",
        "member_of_a_labor_union",
        "live_in_this_house_1_year_ago",
        "tax_filer_stat",
        "year",
    ]
    data = one_hot_encode_features(data, categorical_cols, drop_original=True)
    data, scaler, scaled_cols = add_scaled_numeric_features(data)

    cols_to_drop = [
        "major_industry_code_not_in_universe_or_children",
        "major_industry_code_armed_forces",
        "log_wage_per_hour",
        "log_capital_gains",
        "log_capital_losses",
        "log_dividends_from_stocks",
    ]
    sensitive_prefixes = ("sex", "race", "hispanic_origin", "country_of_birth_self")

    label_idx = list(data.columns).index("label")
    candidate_features = list(data.columns)[label_idx + 1 :]
    selected_features = [
        col
        for col in candidate_features
        if col not in cols_to_drop and not col.startswith(sensitive_prefixes)
    ]
    selected_features = ["year"] + [col for col in selected_features if col != "year"]
    selected_features = [col for col in selected_features if col in data.columns]

    return {
        "data": data,
        "X": data[selected_features],
        "y": data["label"],
        "weight": weight,
        "selected_features": selected_features,
        "scaler": scaler,
        "scaled_cols": scaled_cols,
    }


def save_feature_artifacts(artifacts: dict[str, object], output_dir: Path) -> None:
    """Save engineered data and feature metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(artifacts["data"], output_dir / "engineered_data.pkl")
    pd.to_pickle(artifacts["X"], output_dir / "X.pkl")
    pd.to_pickle(artifacts["y"], output_dir / "y.pkl")
    pd.to_pickle(artifacts["weight"], output_dir / "weight.pkl")
    joblib.dump(artifacts["selected_features"], output_dir / "selected_features.joblib")
    joblib.dump(artifacts["scaler"], output_dir / "numeric_scaler.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build engineered Census features.")
    parser.add_argument("--data-path", type=Path, default=Path("../data/census-bureau.data"))
    parser.add_argument("--columns-path", type=Path, default=Path("../data/census-bureau.columns"))
    parser.add_argument("--output-dir", type=Path, default=Path("../artifacts/"))
    args = parser.parse_args()

    raw_df = load_census_data(args.data_path, args.columns_path)
    artifacts = build_features(raw_df)
    save_feature_artifacts(artifacts, args.output_dir)

    print(f"Engineered data shape: {artifacts['data'].shape}")
    print(f"Feature matrix shape: {artifacts['X'].shape}")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
