
from config import CONFIG
from EDA import Covid_EDA
from Schema_Validation import SchemaValidation
from BusinessRule_Validation import Pipelinelogic_Validation
from Anomaly_Detection import AnomalyDetection

"""
Entry point that will run the full CDC COVID-19 validation pipeline.
""" 
 
def main():
    pipeline = Pipelinelogic_Validation(
        url                 = CONFIG["url"],
        limit               = CONFIG["limit"],
        duplicate_threshold = CONFIG["duplicate_threshold"],
        missing_threshold   = CONFIG["missing_threshold"],
    )
 
    (
    pipeline
        .load_data()
        .clean_data()
        # ── EDA ───────────────────────────────────────────────────────
        .dataset_overview()
        .missing_values()
        .numerical_summary()
        .categorical_summary()
        # ── Schema ────────────────────────────────────────────────────
        .validate_schema()
        # ── Business-logic validation ─────────────────────────────────
        .run_businessrule_validation()
        # ── Anomaly detection ─────────────────────────────────────────
        .run_anomaly_detection()
    )
 
 
if __name__ == "__main__":
    main()
 