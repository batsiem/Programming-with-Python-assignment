from config import (
    CONFIG, 
    CDC_SYMPTOM_STATUS_VALUES, 
    CDC_AGE_GROUP_VALUES
    )
from Anomaly_Detection import AnomalyDetection
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class Pipelinelogic_Validation(AnomalyDetection):
    """
    Child class that inherits from parent class SchemaValidation.
    Enforces business logic rules on the dataset to ensure that data is not only schematically correct 
    but actually makes logical sense.
    This class checks that data values are logically consistent e.g death date cannot be before case report date.
    All thresholds are passed in fron CONFIG at the top of the script
*****
    Child class that inherits from AnomalyDetection, which itself inherits
    from SchemaValidation -> Covid_EDA. This puts run_anomaly_detection()
    in the same MRO as run_businessrule_validation(), so both are
    reachable in a single fluent chain on one pipeline object.
    Enforces business logic rules on the dataset ti ensure that data is not only schematically correct 
    but actually makes logical sense.
    This class checks that data values are logically consistent e.d death date cannot be before case report date.
    All thresholds are passed in fron CONFIG at the top of the script
    """  
    def __init__(self, 
                 url                    : str, 
                 limit                  : int, 
                 duplicate_threshold    : int, 
                 missing_threshold      : float
                ):
        """
        url = CDC API point, passed in from CONFIG
        limit = number of rows to be loaded passed in from CONFIG
        duplicate_threshold = max number of times that a demographic combination can appear
        missing_threshold = max percentage of missing values allowed per column
        """
        super().__init__(url, limit)
        self.duplicate_threshold = duplicate_threshold
        self.missing_threshold = missing_threshold
        self.validation_results: list [dict] = []            # stores pass or fail results for each rule


    # Internal helper ***
    def _record(self, rule: str, passed: bool, detail: str= "", warning: bool =False):
        """
        INTERNAL HELPER THAT RECORDS THE RESULT OF EACH VALIDATION RULE.
        USES _ PREFIX TO SIGNAL THIS METHOD FOR INTERNAL USE ONLY.
        Records results of a validation rule and prints a one line summary for each rule
        There are 3 possible statuses:
        PASSED - rule satisfied, data looks correct
        WARNING - rule violated but plausible, flag for review
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
        df = self._require_df()
        earliest = pd.Timestamp(CONFIG["date_min"])
        latest = pd.Timestamp(CONFIG["date_max"])       

        # Identify rows where case_month is outside the valid range
        out_of_range = df[
            (df["case_month"] < earliest) |
            (df["case_month"] > latest)
        ]
        passed = len(out_of_range) == 0
        self._record(
            rule = f"Date range check ({CONFIG['date_min']} to {CONFIG['date_max']})", #####
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
        df = self._require_df()
        invalid = df[
            (df["icu_yn"] == "Yes") &
            (df["hosp_yn"] != "Yes")
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
        df = self._require_df()
        flagged = df[
            (df["hosp_yn"] == "Yes") &
            (df["symptom_status"] != "Symptomatic")
        ]

        warning = len(flagged) > 0
        self._record(
            rule = "Hospitalization without record of symptoms, should be reviewed",
            passed = not warning, 
            detail = (
                f"{len(flagged):,} patients hospitalized without symptom record; " 
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
        df = self._require_df()
        flagged = df[
            (df["death_yn"] == "Yes") &
            (df["hosp_yn"] != "Yes")
        ]
    
        warning = len(flagged) > 0
        self._record(
            rule = "Deaths without hospitalization; review recommended ",
            passed = not warning, 
            detail = (
                f"{len(flagged):,} deaths recorded without hospitalization; "
                "plausible but flagged for review"
                if warning 
                else
                "All deaths match with a hospitalization record "
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
        df = self._require_df()
        missing_pct = df.isnull().mean()*100
        high_missing = missing_pct[missing_pct> self.missing_threshold]

        passed = len(high_missing) == 0
        self._record(
            rule = f"Missing value threshold (<{self.missing_threshold}% per column)",
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
        df = self._require_df()
        duplicate_threshold = self.duplicate_threshold
        group_cols = [
            "case_month", "res_state", "age_group",
            "sex", "race", "death_yn", "hosp_yn",  "current_status"
        ]
        available_cols = [
            col for col in group_cols if col in df.columns
        ]

        # NEW: guard against no usable columns
        if not available_cols:
            self._record(
                rule="Suspicious duplicate combinations",
                passed=False,
                detail="No grouping columns available in dataset — check column names",
                warning=True
            )
            return self

        # rest of the method continues unchanged...
        group_counts = (
            df
            .groupby(available_cols, dropna=False)
            .size()
            .reset_index(name="count")
        )

        suspicious = group_counts[group_counts["count"] > duplicate_threshold]
        warning = len(suspicious) > 0

        self._record(
            rule=f"Suspicious duplicate combinations (threshold: > {duplicate_threshold} rows)",
            passed=not warning,
            detail=(
                f"{len(suspicious):,} combinations exceed threshold - possible repeated data"
                if warning
                else "No suspiciously repeated combinations found"
            ),
            warning=warning,
        )

        if warning:
            print("\nTop suspicious combinations")
            print(suspicious.sort_values("count", ascending=False)
                .head()
                .to_string(index=False))
        return self
    

    # validation rule 7
    def check_age_group_values(self):
        """
        Verify that age_group contains only known CDC age bins
        An API change can introduce a new bin or rename an exisiting one,
        which would break alll age-stratified analyses without this check.
        """
        df = self._require_df()
        if "age_group" not in df.columns:
            self._record(
                rule = "Age group enumeration check",
                passed = False,
                detail = "Column 'age_group' not found in dataset",
                warning= False
            )
            return self
        
        unknown_ages = df[
            (~df["age_group"].isin(CDC_AGE_GROUP_VALUES)) &
            df["age_group"].notna()
        ]
        passed = len(unknown_ages) == 0
        self._record(
            rule = "Age group enumeration check",
            passed = passed, 
            detail = (
                f"{len(unknown_ages):,} rows contain unrecognised age_group values:"
                f"{unknown_ages['age_group'].unique().tolist()}"  ####
                if not passed
                else 
                "All age group value match known CDC age bins"
            ), warning = False
        )
        return self
    

    # Rule 8 — NEW (Layer 2 recommendation)
    def check_binary_field_distributions(self):
        """
        Sanity-check marginal Yes-rates for exposure_yn and
        underlying_conditions_yn.
 
        A Yes-rate outside [1%, 99%] may indicate a coding change or a
        batch data error rather than a true epidemiological signal.
        """
        df = self._require_df()
        df = self._require_df()
        fields_to_check = {
            "exposure_yn"              : (0.01, 0.99),
            "underlying_conditions_yn" : (0.01, 0.99),
        }
 
        for col, (low, high) in fields_to_check.items():
            if col not in df.columns:
                continue                                  # column absent — skip silently
 
            non_null = df[col].notna().sum()              # ← inside the loop (was outside)
 
            if non_null == 0:                             # ← None guard is its own branch
                self._record(
                    rule    = f"Distribution sanity check — {col}",
                    passed  = False,
                    detail  = f"All values in '{col}' are null; cannot evaluate distribution",
                    warning = True,
                )
                continue                                  # nothing more to check for this col
 
            # yes_rate is always float here — the None branch above already handled 0 rows
            yes_rate = (df[col] == "Yes").sum() / non_null
 
            if yes_rate < low or yes_rate > high:
                self._record(
                    rule    = f"Distribution sanity check — {col}",
                    passed  = False,
                    detail  = (
                        f"'{col}' Yes-rate is {yes_rate:.1%}, outside expected "
                        f"range [{low:.0%}, {high:.0%}] — possible encoding shift"
                    ),
                    warning = True,
                )
            else:
                self._record(
                    rule   = f"Distribution sanity check — {col}",
                    passed = True,
                    detail = f"Yes-rate = {yes_rate:.1%} (within expected range)",
                )
 
        return self

   

    def validation_summary(self):
        """
        Print a summary table of all validation results showing how many rules passed,
        produced warnings, and failed.
        """
        df = self._require_df()
        results_df = pd.DataFrame(self.validation_results)
        total = len(results_df)
        passed = (results_df["Status"] == "PASSED").sum()
        warning_count = (results_df["Status"] == "WARNING").sum()
        failed = (results_df["Status"] == "FAILED").sum()

        print("\n"+ "="*55)
        print("PIPELINE VALIDATION SUMMARY")
        print("="*55)
        print(results_df.to_string(index=False))
        print("="*55)
        print(f"Total: {total} | Passed: {passed} | Warnings: {warning_count} | Failed: {failed}")
        print("="*55)
        return self 
    
    def run_businessrule_validation(self):
        """Run all pipeline validation rules in one call"""
        if self.df is None or not self._is_clean:
             raise RuntimeError(
        "Run load_data().clean_data() first."
        )     
        (self
        .check_date_range()
        .check_icu_against_hospitalization()
        .hospitalization_requires_symptoms()
        .check_death_sans_hospitalization()        
        .checking_missing_threshold()
        .check_repeated_combinations()
        .check_age_group_values()
        .check_binary_field_distributions()
        .validation_summary()
           )
        return self

        
