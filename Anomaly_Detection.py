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

        # populated by capture lineage(); empty until method runs
        self.lineage: dict = {}

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


    def check_interval_consistency(self):
        """
        Temporal validation using case_onset_interval / case_positive_specimen_interval.
 
        CDC suppresses raw dates in this dataset for privacy (e.g. date of
        first positive specimen is an "indirect identifier"). Both interval
        columns are offsets, in the same units, from one shared hidden
        reference date — so their relationship to each other is checkable
        even though the real dates aren't visible:
 
          1. Neither interval can be negative. Since both are measured
             forward from the same reference point, a negative value is
             impossible under the data's own definition — a hard fail.
          2. A very large gap between the two intervals is unusual but not
             impossible (e.g. pre-symptomatic testing, or a late-reported
             onset) — flagged as a warning, not a hard fail.
        """
        self._assert_clean("check_interval_consistency")
        df = self._require_df()
 
        interval_cols = ("case_onset_interval", "case_positive_specimen_interval")
        missing_cols = [c for c in interval_cols if c not in df.columns]
        if missing_cols:
            self._record(
                rule="Interval consistency check",
                passed=False,
                detail=f"Column(s) not found in dataset: {missing_cols}",
                warning=True,
            )
            return self
 
        onset = df["case_onset_interval"]
        specimen = df["case_positive_specimen_interval"]
 
        # ---- Rule 1: negative intervals (hard fail) ------------------
        # notna() first so pd.NA rows are excluded rather than producing
        # an ambiguous boolean mask.
        negative = df[
            (onset.notna() & (onset < 0)) |
            (specimen.notna() & (specimen < 0))
        ]
        passed_negative = len(negative) == 0
        self._record(
            rule="Interval consistency — no negative intervals",
            passed=passed_negative,
            detail=(
                f"{len(negative):,} rows have a negative onset or specimen interval"
                if not passed_negative
                else "No negative intervals found"
            ),
            warning=False,
        )
 
        # ---- Rule 2: implausibly large gap between the two (warning) --
        max_gap = CONFIG["max_interval_gap_weeks"]
        both_present = onset.notna() & specimen.notna()
        gap = (onset - specimen).abs()
 
        large_gap = df[both_present & (gap > max_gap)]
        warning = len(large_gap) > 0
        self._record(
            rule=f"Interval consistency — onset/specimen gap (> {max_gap} weeks)",
            passed=not warning,
            detail=(
                f"{len(large_gap):,} rows have an onset/specimen interval gap "
                f"exceeding {max_gap} weeks; plausible (e.g. pre-symptomatic "
                "testing) but flagged for review"
                if warning
                else f"All onset/specimen interval gaps are within {max_gap} weeks"
            ),
            warning=warning,
        )
 
        return self
 
    def capture_lineage(self):
        """
        Lightweight data lineage / provenance snapshot.
 
        Captures where the data came from, when this snapshot was taken,
        how large the dataset currently is, its overall completeness, and
        a rollup of every check recorded so far in self.anomaly_results.
 
        LIMITATION: this timestamps "when capture_lineage() ran," not "when
        load_data() ran" — Covid_EDA.load_data()/clean_data() don't
        currently record their own timestamps or row counts. For a fuller
        before/after picture (e.g. rows or values lost during cleaning),
        load_data() and clean_data() would need to stamp
        self.rows_loaded / self.rows_after_cleaning themselves. This method
        works with what's available today without modifying those methods.
        """
        self._assert_clean("capture_lineage")
        df = self._require_df()
 
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isnull().sum().sum())
        completeness_pct = (
            100 * (1 - missing_cells / total_cells) if total_cells else 0.0
        )
 
        status_counts = {"PASSED": 0, "WARNING": 0, "FAILED": 0}
        for entry in self.anomaly_results:
            status_counts[entry["Status"]] = status_counts.get(entry["Status"], 0) + 1
 
        self.lineage = {
            "captured_at": pd.Timestamp.now().isoformat(),
            "source_url": self.url,
            "row_count": df.shape[0],
            "column_count": df.shape[1],
            "completeness_pct": round(completeness_pct, 2),
            "anomaly_checks_passed": status_counts["PASSED"],
            "anomaly_checks_warning": status_counts["WARNING"],
            "anomaly_checks_failed": status_counts["FAILED"],
        }
 
        print("\n" + "=" * 55)
        print("DATA LINEAGE SNAPSHOT")
        print("=" * 55)
        for key, value in self.lineage.items():
            print(f"  {key:<24}: {value}")
        print("=" * 55)
 
        return self
 
    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
 
    def run_anomaly_detection(self):
        """
        Run all anomaly detection checks in one call.
        Requires load_data().clean_data() to have been called first.
        """
        if self.df is None or not self._is_clean:
            raise RuntimeError(
                "Run load_data().clean_data() first."
            )
        (self
            .check_monthly_case_anomalies()
            .check_change_points()
            .check_age_distribution_drift()
            .check_future_dates()
            .check_interval_consistency()
            .capture_lineage()
        )
        return self   

