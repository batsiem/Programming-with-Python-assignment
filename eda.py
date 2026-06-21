"""
In this step a class is created for the Exploratory Data Analysis by loading and reading the first 50 000 lines from CDC COVID-19 Case Surveillance Public Use Data with Geography database
to understand dataset's main characteristics, discover patterns, spot anomalies, and check statistical assumptions.
"""

# read data directly from CDC website
from config import CONFIG
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class Covid_EDA:
    """
    Reusable Class that conducts Exploratory Data Analysis on a CDC COVID-19 Case Surveillance Public Use Data with Geography dataset
    Performs:
        - Dataset overview
        - Missing value analysis
        - Data cleaning
        - Numerical summaries*
        - Categorical summaries*
        - Descriptive statistics
        - Dataset visualizations
    """
    def __init__(self, url, limit):  
        """
        url = CDC API point, passed in from CONFIG
        limit = number of rows to be loaded passed in from CONFIG
        """  
        # build full API request from URL using url and limit from CONFIG        
        self.url = f"{url}?$limit={limit}"
        self.df = None
        self.unknown_vals = [
            "Missing", "Unknown", "NA", 
            "missing", "unknown", "N/A"
            ]
       
    def load_data(self):
        """Loading the data from CDC APIL url into a dataframe."""
        try:
            self.df = pd.read_csv(self.url)
            print(f"Dataset successfully loaded.")
            print(f"Loaded {len(self.df):,} rows x {self.df.shape[1]} columns")  
        except Exception as e:
            print(f"Error loading dataset: {e}")
        return self

    def dataset_overview(self):
        """Display basic information about the dataset"""
        # datatypes contained in the dataset
        print(f"Dataset contains the following data types:\n{self.df.dtypes}\n")
        # display first 5 number of rows in the dataset
        print(f"First 5 rowss:\n{self.df.head()}\n")
        return self
    
    def missing_values(self):
        """Display missing value count and percentage for each column in the dataset."""
        missing_count = self.df.isnull().sum()
        missing_percent = (
            missing_count / len(self.df)
            ) * 100

        missing_df = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percent.round(2)
        })
        print(missing_df[missing_df['Missing Count'] > 0]
              .sort_values(by='Missing Count', ascending=True))
        return self

    def clean_data(self):
        """Fix data types, remove CDC placeholder strings, strip whitespace."""
        # Convert case_month to datetime
        self.df["case_month"] = pd.to_datetime(
            self.df["case_month"], 
            errors="coerce"
            ) 

        # Replace CDC placeholders strings      
        self.df.replace(
            self.unknown_vals, 
            pd.NA, 
            inplace=True
            )

        # Select all text columns and remove leading or trailing spaces from text columns
        str_cols = self.df.select_dtypes(
            include=["object", "string"]).columns
        self.df[str_cols] = self.df[str_cols].apply(
            lambda col : col.str.strip()
            )
        
        # convert columns that should be integers back to nullable integer type.
        # pandas used pd.NA -compatible Int64 instead of numpy int64
        # so that integer columns can hold Nan without being cast to float

        # Cast integer-like columns to pandas nullable Int64 so that they can hold
        # NaN without being silently upcast to float        
        integer_columns = [
            "county_fips_code",
            "state_fips_code",
        ]

        for col in ("county_fips_code", "state_fips_code"):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col], 
                    errors="coerce"
                    ).astype("Int64")
        
        for col in ("case_positive_specimen_interval", "case_onset_interval"):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self._is_clean =True        
        print("Data cleaned \n")
        return self

    def numerical_summary(self):
        numeric_cols = self.df.select_dtypes(
            include=["number"]
        )
        print("\nNUMERICAL SUMMARY")
        print(numeric_cols.describe())
        return self

    def categorical_summary(self):
        cat_cols = self.df.select_dtypes(
            include=["object", "string"]
    )
        print("\nCATEGORICAL SUMMARY")
        for col in cat_cols.columns:
            print(f"\n{col}")
            print(
                self.df[col]
                .value_counts(dropna=False)
                .head(10)
            )
        return self
    
    def descriptive_statistics(self):
        """Print summary statistics for all columns."""
        print(self.df.describe(include= "all" ))
        return self

    """
    Visual depiction of the data
    """
    def plot_cases_per_month(self):
        """
        Plot number of cases reported per month.
        """
        monthly_cases = (
            self.df["case_month"]
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

    def plot_cases_by_state(self):
        """
        Plot number of cases by state.
        """
        state_cases = (
            self.df["res_state"]
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

    def plot_cases_per_county(self):
        """
        Plot the counties with the highest number of COVID-19 cases.
        """
        county_cases = (
            self.df["res_county"]
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
    
    def plot_hospitalization_rate_by_age_group(self):
        """
        Plot hospitalization rate (%) by age group.
        """
        hospitalization_rate = (
            self.df.groupby("age_group")["hosp_yn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_index()
        )
        plt.figure(figsize=(10, 6))
        hospitalization_rate.plot(kind="bar")
        plt.title("Hospitalization Rate by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Hospitalization Rate (%)")
        plt.tight_layout()
        plt.show()
        return self

    def plot_death_rate_by_age_group(self):
        """
        Plot death rate (%) by age group.
        """

        death_rate = (
            self.df.groupby("age_group")["death_yn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_index()
        )
        plt.figure(figsize=(10, 6))
        death_rate.plot(kind="bar")
        plt.title("Death Rate by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Death Rate (%)")
        plt.tight_layout()
        plt.show()
        return self

    def run_covid_eda_all(self):
        """
        Run the entire pipeline in one call
        """
        (self
        .load_data()
        .dataset_overview()            # shows raw datatypes from API
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


