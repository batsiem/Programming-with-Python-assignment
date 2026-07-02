from config import CONFIG
from Schema_Validation import SchemaValidation
import pandas as pd
from statsmodels.tsa.seasonal import STL
import ruptures as rpt


 # ********
class AnomalyDetection(SchemaValidation):
    """
    Advanced data quality checks using statistical methods.
    """
    def __init__(self, url, limit):
        super().__init__(url, limit) 
        
        # Store results from all anomaly detection checks 
        self.anomaly_results = []

    @property
    def _monthly_cases(self) -> pd.Series:
        """
        Aggregate monthly case counts, sorted chronologically.
        Requires load_data().clean_data() to have run first (case_month
        must already be datetime — clean_data() handles that).
        """
        self._assert_clean("_monthly_cases")
        df = self._require_df()
        return (
            df["case_month"]
            .value_counts()
            .sort_index()
        ) 

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

        self.anomaly_results.append({
            "Rule" : rule,
            "Status" : status,
            "Detail" : detail
        })
        print(f"[{status}] {rule}" + (f" - {detail}" if detail else""))

 
    def check_monthly_case_anomalies(self):
        """
        Detect unusual spikes and drops in monthly case counts
        using STL decomposition and MAD thresholding.
        """
        monthly_cases = self._monthly_cases
        if len(monthly_cases) < CONFIG["minimum_months_for_stl"]:
            self._record(
                rule="Monthly case anomaly detection",
                passed=False,
                detail="Not enough monthly observations for STL analysis",
                warning=True
                )
            return self

        stl = STL(monthly_cases, period=12)
        result = stl.fit()

        residuals = result.resid

        median = residuals.median()
        mad = (residuals - median).abs().median()

        threshold = 3 * mad

        anomalies = residuals[
        (residuals - median).abs() > threshold
            ]

        warning = len(anomalies) > 0

        self._record(
        rule="Monthly case anomaly detection",
            passed=not warning,
            detail=(
                f"{len(anomalies)} anomalous months detected"
                if warning
                else "No anomalous months detected"
                ),
                warning=warning
            )

        if warning:
            print("\nAnomalous Months:")
            print(anomalies)

        return self
        
    def check_change_points(self):
        """
        Detect structural changes in reporting patterns.
        """
        monthly_cases = self._monthly_cases
        if len(monthly_cases) < CONFIG["minimum_months_for_stl"]:
            self._record(
                rule="Change point detection",
                passed=False,
                detail="Not enough observations",
                warning=True
            )
            return self

        signal = monthly_cases.values

        algo = rpt.Pelt(model="rbf")
        algo.fit(signal)

        change_points = algo.predict(pen=CONFIG["change_point_penalty"])

        detected = len(change_points[:-1])

        warning = detected > 0

        self._record(
            rule="Change point detection",
            passed=not warning,
            detail=(
                f"{detected} change points detected"
                if warning
                else "No change points detected"
                ),
                warning=warning
            )

        if warning:
            print("\nDetected Change Points:")

            for cp in change_points[:-1]:
                print(
                    monthly_cases.index[cp - 1]
                )

        return self
        
    def check_age_distribution_drift(self):
        """
        Detect suspicious age-group concentration.
        """
        self._assert_clean("check_age_distribution_drift")
        df = self._require_df()
        distribution = (
                df["age_group"]
                .value_counts(normalize=True)
        )
        dominant_group = distribution.max()

        warning = dominant_group > CONFIG["max_age_group_percentage"]

        self._record(
            rule="Age-group distribution sanity check",
            passed=not warning,
            detail=(
                f"One age group represents {dominant_group:.1%} of all cases"
                if warning
                else "Age distribution appears reasonable"
                ),
            warning=warning
            )
        return self
        
    def check_future_dates(self):
        """
        Ensure case_month is not in the future.
        """
        self._assert_clean("check_future_dates")
        df = self._require_df()

        today = pd.Timestamp.today().normalize()

        future_rows = df[
            df["case_month"] > today
        ]

        warning = len(future_rows) > 0

        self._record(
            rule="Future date check",
            passed=not warning,
            detail=f"{len(future_rows):,} future-dated records found",
            warning=warning
        )

        if warning:
            print("\nFuture-dated records:")
            print(future_rows)

        return self

    # Calling full pipeline
    def run_anomaly_detection(self):
        """
        Run all anomaly detection checks in one call.
        Requires load_data().clean_data() to be called first.
        """
        if self.df is None or not self._is_clean:
            raise RuntimeError(
                "Run load_data().clean_data() first"
            )
        (
         self
            .check_monthly_case_anomalies()
            .check_change_points()
            .check_age_distribution_drift()
            .check_future_dates   
        )
        return self