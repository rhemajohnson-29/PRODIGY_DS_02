import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def load_data(path="titanic.csv"):
    """Load Titanic data from a CSV path."""
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    return df


def summary_stats(df):
    print("\n*** Dataset summary ***")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())


def data_cleaning(df):
    # Copy to avoid side effects
    df = df.copy()

    # Drop columns not useful for survival prediction in EDA
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Fill missing values
    if "Age" in df.columns:
        age_median = df["Age"].median()
        df["Age"] = df["Age"].fillna(age_median)
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Map categories
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    # Feature engineering
    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 80], labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"])
    if "Fare" in df.columns:
        df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Mid", "High", "VeryHigh"])
    if "Cabin" not in df.columns and "Pclass" in df.columns:
        # add first class indicator
        df["IsFirstClass"] = (df["Pclass"] == 1).astype(int)

    return df


def univariate_analysis(df, output_dir="eda_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Survival rate
    if "Survived" in df.columns:
        surv_rate = df["Survived"].mean()
        print(f"Survival rate: {surv_rate:.4f} ({surv_rate * 100:.2f}%)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Survived", palette="Set2", ax=ax)
        ax.set_title("Survived distribution")
        ax.set_xlabel("Survived")
        ax.set_ylabel("Count")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2, height), ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "survival_counts.png"))
        plt.close(fig)

    # Numeric distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Survived" in numeric_cols:
        numeric_cols.remove("Survived")

    for col in numeric_cols:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax[0], color="teal")
        ax[0].set_title(f"{col} distribution")

        sns.boxplot(x=df[col], ax=ax[1], color="skyblue")
        ax[1].set_title(f"{col} boxplot")

        fig.tight_layout()
        fpath = os.path.join(output_dir, f"{col}_dist_box.png")
        fig.savefig(fpath)
        plt.close(fig)

    # Categorical distributions
    categorical_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    categorical_cols += [c for c in ["Pclass", "Sex", "Embarked", "AgeGroup", "FareBin", "IsFirstClass"] if c in df.columns and c not in categorical_cols]
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette="pastel", ax=ax)
        ax.set_title(f"{col} category counts")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{col}_counts.png"))
        plt.close(fig)


def bivariate_analysis(df, output_dir="eda_plots"):
    os.makedirs(output_dir, exist_ok=True)

    if "Survived" not in df.columns:
        return

    # Survival by categorical features
    cat_vars = ["Pclass", "Sex", "Embarked", "AgeGroup", "FareBin", "IsFirstClass"]
    cat_vars = [c for c in cat_vars if c in df.columns]

    for col in cat_vars:
        fig, ax = plt.subplots(figsize=(7, 5))
        cross = (df[[col, "Survived"]].groupby(col).mean() * 100).sort_values("Survived", ascending=False)
        cross["Survived"].plot(kind="bar", color="coral", ax=ax)
        ax.set_ylabel("Survival rate (%)")
        ax.set_title(f"Survival rate by {col}")
        ax.yaxis.set_major_formatter(PercentFormatter())
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"survival_by_{col}.png"))
        plt.close(fig)

    # Pair plot for numeric features with samples
    numeric_cols = [c for c in ["Age", "Fare", "SibSp", "Parch"] if c in df.columns]
    if numeric_cols:
        sample = df[["Survived"] + numeric_cols].dropna().sample(min(500, len(df)), random_state=42)
        pairplot = sns.pairplot(sample, hue="Survived", palette="Set1", vars=numeric_cols, plot_kws={"alpha": 0.5})
        pairplot.fig.suptitle("Pairplot for numeric features colored by Survived", y=1.02)
        pairplot.savefig(os.path.join(output_dir, "pairplot_survival.png"))
        plt.close("all")


def correlation_analysis(df, output_dir="eda_plots"):
    os.makedirs(output_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return

    corr = numeric_df.corr()
    print("\n*** Numeric correlation matrix ***")
    print(corr)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close(fig)


def main():
    # Set path to your Titanic CSV file here
    csv_path = "titanic.csv"

    try:
        df = load_data(csv_path)
    except FileNotFoundError:
        print(f"Data file not found at {csv_path}. Please place titanic.csv in this folder or update the path.")
        return

    summary_stats(df)
    df = data_cleaning(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    correlation_analysis(df)

    print("\nEDA completed. Check generated plots in the 'eda_plots' directory.")


if __name__ == "__main__":
    main()
