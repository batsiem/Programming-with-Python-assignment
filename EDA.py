"""
Exploratory Data Analysis (EDA) on CDC COVID-19 Case Surveillance Public Use Data with Geography dataset
Loads "limit" number of rows from CDC Socrata API data source and performs:
    - Dataset overview
    - Missing value analysis
    - Data cleaning
    - Numerical and categorical summaries*
    - Descriptive statistics
    - Key Dataset visualizations
"""

# read data directly from CDC website
from config import CONFIG
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


class Covid_EDA:
    """
    Reusable class for exploratory analysis of the CDC COVID-19
    Case Surveillance dataset.
    """
    def __init__(self, url: str, limit: int):  
        """
        Parameters
        ----------
        url   : CDC Socrata API endpoint (from CONFIG).
        limit : Maximum number of rows to load (from CONFIG).
        """

        # build full API request from URL using url and limit from CONFIG        
        self.url = f"{url}?$limit={limit}"
        self.df = None
        self._is_clean = False
        self.unknown_vals = [
            "Missing", "Unknown", "NA", 
            "missing", "unknown", "N/A"
            ]
    
    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataset has not been loaded.")
        return self.df

    # Loading the dataset into csv file
    def load_data(self):
        try:
            self.df = pd.read_csv(self.url)
            print("Dataset successfully loaded.")
            print(f"Loaded {len(self.df):,} rows x {self.df.shape[1]} columns")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}") from e

        return self

    # Overview of dataset, displaying basic information about the dataset
    def dataset_overview(self):
        df = self._require_df()
        print(f"Dataset contains the following data types:\n{df.dtypes}\n")
        # display first 5 number of rows in the dataset
        print(f"First 5 rows:\n{df.head()}\n")
        return self
    
    # Display missing value count and percentage for each column in the dataset.
    def missing_values(self):
        df = self._require_df()
        missing_count = df.isnull().sum()
        missing_percent = (
            missing_count / len(df)
            ) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percent.round(2)
        })
        print(missing_df[missing_df['Missing Count'] > 0]
              .sort_values(by='Missing Count', ascending=True))
        return self


    # Cleaning the dataset
    def clean_data(self):
        """
        Fix data types, remove CDC placeholder strings, and strip
        leading/trailing whitespace from text columns.
        """
        df = self._require_df()
        # Convert case_month to datetime
        df["case_month"] = pd.to_datetime(
            df["case_month"],
            errors="coerce"
        )

        # Replace CDC placeholder strings with missing values
        df.replace(
            self.unknown_vals,
            pd.NA,
            inplace=True
        )

        # Strip whitespace from all text columns
        str_cols = df.select_dtypes(
            include=["object", "string"]
        ).columns

        for col in str_cols:
            df[col] = df[col].str.strip()

        # Convert integer-like columns to pandas nullable Int64
        integer_columns = [
            "county_fips_code",
            "state_fips_code",
        ]

        for col in integer_columns:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(
                        df[col],
                        errors="coerce"
                    )
                    .astype("Int64")
                )

        # Convert interval columns to numeric
        for col in (
            "case_positive_specimen_interval",
            "case_onset_interval",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        self._is_clean = True
        print("Data cleaned.\n")

        return self

    # Summaries

    def numerical_summary(self):
        df = self._require_df()
        numeric_cols = df.select_dtypes(
            include=["number"]
        )
        print("\nNUMERICAL SUMMARY")
        print(numeric_cols.describe())
        return self

    def categorical_summary(self):
        """
        Print value counts (top 10) for every categorical column.
        """
        df = self._require_df()
        cat_cols = df.select_dtypes(
            include=["object", "string"]
        )
        print("\nCATEGORICAL SUMMARY")
        for col in cat_cols.columns:
            print(f"\n{col}")
            print(df[col].value_counts(dropna=False).head(10))
        return self



    def descriptive_statistics(self):
        df = self._require_df()
        """Print summary statistics for all columns."""
        print(df.describe(include= "all" ))
        return self


    # Visual depiction of the data
    # Plot number of cases reported per month.
    def plot_cases_per_month(self):
        df = self._require_df()
        monthly_cases = (
            df["case_month"]
            .value_counts()
            .sort_index()
        )
        plt.figure(figsize=(12, 6))
        monthly_cases.plot(kind="line", marker="o")
        plt.title("COVID-19 Cases Per Month")
        plt.xlabel("Month")
        plt.ylabel("Number of Cases")
        plt.tight_layout()
        plt.show()
        return self


    # Plot number of cases by state.
    def plot_cases_by_state(self):
        df = self._require_df()
        state_cases = (
            df["res_state"]
            .value_counts()
            .sort_values(ascending=False)
        )
        plt.figure(figsize=(14, 6))
        state_cases.plot(kind="bar")
        plt.title("COVID-19 Cases by State")
        plt.xlabel("State")
        plt.ylabel("Number of Cases")
        plt.tight_layout()
        plt.show()
        return self


    # Plot the counties with the highest number of COVID-19 cases.
    def plot_cases_per_county(self):
        df = self._require_df()
        county_cases = (
            df["res_county"]
            .value_counts()
            .head(CONFIG["top_counties"])
        )
        plt.figure(figsize=(12, 6))
        county_cases.plot(
            kind="bar",
            rot=45
        )
        plt.title(
        f"Top {CONFIG['top_counties']} Counties by Number of Cases"
        )
        plt.xlabel("County")
        plt.ylabel("Number of Cases")
        plt.tight_layout()
        plt.show()
        return self 
    
    
    def _yes_rate_by_group(
        self,
        group_col : str,
        value_col : str,
    ) -> pd.Series:
        """
        Compute the percentage of rows where `value_col` == 'Yes',
        grouped by `group_col`.
        """
        df = self._require_df()
        return (
            (df[value_col] == "Yes")           # bool Series — O(n) vectorised
            .groupby(df[group_col])            # group without materialising sub-frames
            .mean()                            # mean of bools = proportion
            .mul(100)                          # → percentage
            .sort_index()
        )
 
    def plot_hospitalization_rate_by_age_group(self):
        """Bar chart: hospitalisation rate (%) by age group."""
        hosp_rate = self._yes_rate_by_group("age_group", "hosp_yn")
 
        plt.figure(figsize=(10, 6))
        hosp_rate.plot(kind="bar")
        plt.title("Hospitalisation Rate by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Hospitalisation Rate (%)")
        plt.tight_layout()
        plt.show()
        return self
 
    def plot_death_rate_by_age_group(self):
        """Bar chart: death rate (%) by age group."""
        death_rate = self._yes_rate_by_group("age_group", "death_yn")
 
        plt.figure(figsize=(10, 6))
        death_rate.plot(kind="bar")
        plt.title("Death Rate by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Death Rate (%)")
        plt.tight_layout()
        plt.show()
        return self
 
    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
 
    def run_covid_eda_all(self):
        """Run the entire EDA pipeline in one call."""
        (self
            .load_data()
            .dataset_overview()
            .missing_values()
            .clean_data()
            .numerical_summary()
            .categorical_summary()
            .descriptive_statistics()
            .plot_cases_per_month()
            .plot_cases_by_state()
            .plot_cases_per_county()
            .plot_hospitalization_rate_by_age_group()
            .plot_death_rate_by_age_group()
        )