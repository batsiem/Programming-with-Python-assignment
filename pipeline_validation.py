from config import CONFIG
from eda import Covid_EDA
from schema_validation import SchemaValidation
import pandas as pd
from statsmodels.tsa.seasonal import STL
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')

class PipelineValidation(Covid_EDA):
    """
    Child class that inherits from parent class Covid_EDA.
    Enforces business logic rules on the dataset ti ensure that data is not only schematically correct 
    but actually makes logical sense.
    This class checks that data values are logically consistent e.d death date cannot be before case report date.
    All thresholds are passed in fron CONFIG at the top of the script
    """
    
    def __init__(self, url, limit, duplicate_threshold, missing_threshold):
        """
        url = CDC API point, passed in from CONFIG
        limit = number of rows to be loaded passed in from CONFIG
        duplicate_threshold = max number of times that a demographic combination can appear
        missing_threshold = max percentage of missing values allowed per column
        """
        super().__init__(url, limit)
        self.duplicate_threshold = duplicate_threshold
        self.missing_threshold = missing_threshold
        self.validation_results = []            # stores pass or fail results for each rule

    # NOT SURE WTF IS HAPPENING HERE
    def _record(self, rule, passed, detail="", warning=False):
        """
        INTERNAL HELPER THAT RECORDS THE RESULT OF EACH VALIDATION RULE.
        USES _ PREFIX TO SIGNAL THIS METHOD FOR INTERNAL USE ONLY.
        APPENDS A RESULT TO A DICTIONARY FOR seld.validation_results and prints a one line summary for each rule
        There are 3 possible statuses:
        PASSED - rule satisfied, data looks correct
        WARNING - rule violate but plausible, flag for review
        FAILED - rule broken and likely indicates a data error
        """
        if passed:
            status = "PASSED"
        elif warning:
            status = "WARNING"          # soft violation, suspicious but not necessarily wrong
        else:
            status = "FAILED"           # hard violation, data is likely incorrect

        self.validation_results.append({
            "Rule" : rule,
            "Status" : status,
            "Detail" : detail
        })
        print(f"[{status}] {rule}" + (f" - {detail}" if detail else""))

    
    # Validation rule 1
    def check_date_range(self):
        """
        Enforce that all case_month values fall within CDC reporting window i.e from January 2020 - 1 July 2024 (based on CDC dataset window of relevance)
        and not in the future
        This rule violation would be a hard fail
        """
        earliest = pd.Timestamp("2020-01-01")
        latest = pd.Timestamp("2024-07-05")         

        # Identify rows where case_month is outside the valid range
        out_of_range = self.df[
            (self.df["case_month"] < earliest) |
            (self.df["case_month"] > latest)
        ]
        passed = len(out_of_range) == 0
        self._record(
            rule = "Date range check (2020-01-01 to 2024-07-05)",
            passed = passed, 
            detail = f"{len(out_of_range):,} rows outside the valid range",
            warning = False
            )
        return self
    
    # Validation rule 2 
    def check_icu_against_hospitalization(self):
        """
        Enforces the rule that icu_yn == "Yes" only when hosp_yn =="Yes". A patient can't be in ICU without being hosipitalized
        This rule violation would be a hard fail, as it is clinically impossible
        """
        invalid = self.df[
            (self.df["icu_yn"] == "Yes") &
            (self.df["hosp_yn"] != "Yes")
        ]
        passed = len(invalid) == 0
        self._record(
            rule = "ICU requires hospitalization",
            passed = passed,
            detail = f"{len(invalid):,} ICU cases recorded without hospitalization.",
            warning = False
        )
        return self
    
    # Validation rule 3
    def hospitalization_requires_symptoms(self):
        """
        Flags cases where "hosp_yn" == "Yes" but sympton_status != "Symptomatic".
        This is a warning because symptom_status is typically missing in surveillance data 
        and an absense of symptom record does not necessarily mean that the patient is asymptomatic
        """
        flagged = self.df[
            (self.df["hosp_yn"] == "Yes") &
            (self.df["symptom_status"] != "Symptomatic")
        ]

        warning = len(flagged) > 0
        self._record(
            rule = "Hospitalization without record of symptoms, should be reviewed",
            passed = not warning, 
            detail = (
                f"{len(flagged):,} patients hospitalized without symptom record," 
            "may reflect missing data rather than asymptomatic cases"
            if warning 
            else
            "All hospitalised patients have a symptom record"
            ),
            warning = warning
        )
        return self
    
    # Validation rule 4
    def check_death_sans_hospitalization(self):
        """
        Flag cases where death_yn ="Yes" but hosp_yn != "Yes".
        This rule violation is a warning, because it is plausible that patients died without
        being admitted to a hospital.
        This rule violation is flagged for human review
        """
        flagged = self.df[
            (self.df["death_yn"] == "Yes") &
            (self.df["hosp_yn"] != "Yes")
        ]
    
        warning = len(flagged) > 0
        self._record(
            rule = "Deaths without hospitalization; review recommended",
            passed = not warning, 
            detail = (
                f"{len(flagged):,} deaths recorded without hospitalization,"
                "plausible but flagged for review"
                if warning 
                else
                "All deaths match with a hospitalization record"
                ),
                warning = warning
            )
        return self

    # Validation rule 5
    def checking_missing_threshold(self):
        """
        Flag any column where more than threshold percent of values are missing, because columns with excessive
        missingness are unreliable for analysis. 
        Default value is 80%
        TRY AND REVERSE CODE WITH LIMIT OF 20% MISSINGNESS
        """
        missing_pct = self.df.isnull().mean()*100
        high_missing = missing_pct[missing_pct> self.missing_threshold]

        passed = len(high_missing) == 0
        self._record(
            rule = f"Missing valur threshold (<{self.missing_threshold}% per column)",
            passed = passed,
            detail = (
                " ,".join(
                    f"{col}: {pct:.1f}%"
                    for col, pct in high_missing.items()
                ) if not passed else ""
            )
        )
        return self 


    # validation rule 6
    def check_repeated_combinations(self):
        """
        CDC Dataset does not contain unique identifiers (they were stripped for privacy), so fully duplicated rows can be expected
        as many patients with share similar race, age range, state and month tested.
        This rule, checks for excessively repeated combinations such as case_month + res_state + age_group + sex +  race. 
        This could be indicative of data feed error or repeated API ingestion, rather than actual case volume.
        """
        # Define columns that combined together form a logical grouping key
        # These are the fields that are most likely to be identical across real cases
        duplicate_threshold = self.duplicate_threshold
        group_cols = [
            "case_month", "res_state", "age_group",
            "sex", "race", "death_yn", "hosp_yn"
        ]
        # Using columns that strictly exist in the dataset
        available_cols = [
            col for col in group_cols if col in self.df.columns
        ]
        
        # Count how many times each combo appears 
        group_counts = (
            self.df 
            .groupby(available_cols, dropna=False)
            .size()
            .reset_index(name="count")
        )

        suspicious = group_counts[group_counts["count"]> duplicate_threshold]
        warning = len(suspicious) > 0

        self._record(
            rule = f"Suspicious duplicate combinations (threshold: > {duplicate_threshold} rows)",
            passed = not warning,
            detail= (
                f"{len(suspicious):,} combinations exceed threshold - possible repeated data"
                if warning 
                else
                "No suspiciously repeated combinations found" 
            ),
            warning = warning, 
        )
    
        if warning:
            print("\nTop suspicious combinations")
            print(suspicious.sort_values("count", ascending=False)
                  .head()
                  .to_string(index=False)
                  )
        return self

    


    def validation_summary(self):
        """
        Print a summary table of all validation results showing how many rules passed,
        produced warnings, and failed.
        """
        results_df = pd.DataFrame(self.validation_results)
        total = len(results_df)
        passed = (results_df["Status"] == "PASSED").sum()
        warnings = (results_df["Status"] == "WARNING").sum()
        failed = (results_df["Status"] == "FAILED").sum()

        print("\n"+ "="*55)
        print("PIPELINE VALIDATION SUMMARY")
        print("="*55)
        print(results_df.to_string(index=False))
        print("="*55)
        print(f"Total: {total} | Passed: {passed} | Warnings: {warnings} | Failed: {failed}")
        print("="*55)
        return self 
    
    def run_pipeline_validation(self):
        """Run all pipeline validation rules in one call"""
        (self
        .check_date_range()
        .check_icu_against_hospitalization()
        .hospitalization_requires_symptoms()
        .check_death_sans_hospitalization()        
        .checking_missing_threshold()
        .check_repeated_combinations()
        .validation_summary()
        )
        
