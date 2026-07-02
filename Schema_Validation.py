"""
Schema validation layer for the CDC COVID-19 pipeline.
 
Validates the cleaned DataFrame against a predefined pandera schema,
checking column names, data types, nullability, and enumeration values.
Must be called after load_data().clean_data().
"""
 
import warnings
warnings.filterwarnings("ignore")
 
import pandera.pandas as pa
import pandera.errors as pa_errors
 
from config import (
    CONFIG,
    CDC_AGE_GROUP_VALUES,
    CDC_BINARY_VALUES,
    CDC_CURRENT_STATUS_VALUES,
    CDC_PROCESS_VALUES,
    CDC_SEX_VALUES,
    CDC_SYMPTOM_STATUS_VALUES,
)
from EDA import Covid_EDA
 
 
class SchemaValidation(Covid_EDA):
    """
    Child of Covid_EDA.
 
    Validates the loaded DataFrame against a predefined pandera schema,
    checking column names, data types, nullability, and allowed values.
    Must be called after load_data().clean_data().
    """
 
    def __init__(self, url: str, limit: int):
        """
        Parameters
        ----------
        url   : CDC Socrata API endpoint (from CONFIG).
        limit : Maximum number of rows to load (from CONFIG).
        """
        super().__init__(url, limit)
 
        # Build the schema once at construction time rather than
        # rebuilding it on every validate_schema() call.
        # _schema is None until _build_schema() is called after load_data(),
        # because column presence depends on self.df.
        self._schema: pa.DataFrameSchema | None = None
 
    # ------------------------------------------------------------------
    # Order guard
    # ------------------------------------------------------------------
 
    def _assert_clean(self, method_name: str) -> None:
        """
        Raise RuntimeError if clean_data() has not been called yet.
 
        BUG FIX: original concatenated two string literals without a
        separating space, producing:
            "...requires clean data.Call load_data()..."
        Fixed with an explicit space before 'Call'.
        """
        if not self._is_clean:
            raise RuntimeError(
                f"{method_name}() requires clean data. "   # ← space added here
                "Call load_data().clean_data() before running schema validation."
            )
 
    # ------------------------------------------------------------------
    # Schema construction (separated from validation for cacheability)
    # ------------------------------------------------------------------
 
    def _build_schema(self) -> pa.DataFrameSchema:
        """
        Build and return the pandera DataFrameSchema for self.df.
 
        Separated from validate_schema() so the schema object can be
        constructed once and reused across multiple validate() calls
        (e.g. in tests or batch pipelines) rather than being rebuilt
        from scratch every time.
 
        SCALABILITY FIX: coerce=True was set in the original, which forces
        pandera to cast every column before validating — effectively doubling
        memory usage on large DataFrames. Since clean_data() already casts
        all types correctly, coerce is set to False here.
        """
        df = self._require_df()
        schema_dict: dict = {}
 
        # ── Datetime ──────────────────────────────────────────────────
        schema_dict["case_month"] = pa.Column(pa.DateTime, nullable=True)
 
        # ── Nullable integer (FIPS codes) ─────────────────────────────
        for col in ("county_fips_code", "state_fips_code"):
            if col in df.columns:
                schema_dict[col] = pa.Column("Int64", nullable=True)
 
        # ── Float (interval columns) ──────────────────────────────────
        for col in ("case_positive_specimen_interval", "case_onset_interval"):
            if col in df.columns:
                schema_dict[col] = pa.Column(pa.Float64, nullable=True)
 
        # ── Binary indicator columns (Yes / No only) ──────────────────
        for col in (
            "exposure_yn", "hosp_yn", "icu_yn",
            "death_yn", "underlying_conditions_yn",
        ):
            if col in df.columns:
                schema_dict[col] = pa.Column(
                    pa.String,
                    nullable=True,
                    checks=pa.Check.isin(CDC_BINARY_VALUES),
                )
 
        # ── Categorical columns with known CDC enumerations ───────────
        categorical_checks = {
            "sex"            : CDC_SEX_VALUES,
            "current_status" : CDC_CURRENT_STATUS_VALUES,
            "symptom_status" : CDC_SYMPTOM_STATUS_VALUES,
            "age_group"      : CDC_AGE_GROUP_VALUES,
            "process"        : CDC_PROCESS_VALUES,
        }
        for col, allowed in categorical_checks.items():
            if col in df.columns:
                schema_dict[col] = pa.Column(
                    pa.String,
                    nullable=True,
                    checks=pa.Check.isin(allowed),
                )
 
        # ── Free-text columns ─────────────────────────────────────────
        for col in ("res_state", "res_county", "race", "ethnicity"):
            if col in df.columns:
                schema_dict[col] = pa.Column(pa.String, nullable=True)
 
        return pa.DataFrameSchema(
            schema_dict,
            strict="filter",  # ignore columns not listed above
            coerce=False,     # types already fixed by clean_data() — no re-cast needed
        )
 
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
 
    def validate_schema(self):
        """
        Validate self.df against the predefined pandera schema.
        Must be called after clean_data().
 
        On failure, prints a structured summary of which columns failed
        and what values violated the schema, rather than a raw exception
        wall-of-text.
        """
        self._assert_clean("validate_schema")
 
        # Build the schema once; reuse it on subsequent calls.
        # The assert below is a type narrowing hint for Pylance/pyright:
        # the annotation `pa.DataFrameSchema | None` means the checker
        # can't prove _schema is non-None after the if-block alone, because
        # _build_schema() is an opaque call it can't inspect. The assert
        # gives the type checker a guarantee it can verify statically, and
        # costs nothing at runtime since _build_schema() always returns a
        # DataFrameSchema (never None).
        if self._schema is None:
            self._schema = self._build_schema()
        assert self._schema is not None   # narrows type: DataFrameSchema | None → DataFrameSchema
 
        df = self._require_df()
 
        try:
            self._schema.validate(df, lazy=True)
            print("\nSchema validation passed.")
 
        except pa_errors.SchemaErrors as exc:
            # BUG FIX: original caught bare Exception and called print(e),
            # which dumps a wall of unformatted text.
            # pa_errors.SchemaErrors exposes structured attributes:
            #   .failure_cases  → DataFrame of rows/values that failed
            #   .schema_errors  → list of dicts with column + check detail
            print("\nSchema validation failed.")
            print(f"  {len(exc.schema_errors)} error(s) found:\n")
 
            failure_summary = (
                exc.failure_cases
                [["schema_context", "column", "check", "failure_case"]]
                .drop_duplicates()
                .head(20)      # cap output — full detail in exc.failure_cases
            )
            print(failure_summary.to_string(index=False))
 
            if len(exc.failure_cases) > 20:
                print(
                    f"\n  ... and {len(exc.failure_cases) - 20} more. "
                    "Inspect exc.failure_cases for the full list."
                )
 
        return self
 
    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
 
    def run_schema(self):
        """
        Run schema validation in one call.
        Requires load_data().clean_data() to have been called first.
        """
        return self.validate_schema()