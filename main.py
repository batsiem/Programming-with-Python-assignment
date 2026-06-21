from config import CONFIG
from eda import Covid_EDA
from schema_validation import SchemaValidation
from pipeline_validation import PipelineValidation

if __name__ == "__main__":
    # --- EDA ---
    eda = Covid_EDA(url=CONFIG["url"], limit=CONFIG["limit"])
    eda.run_covid_eda_all()
 
    # --- Schema validation ---
    # FIX: SchemaValidation does not run EDA automatically; load + clean first,
    #      then validate.  Original called run_covid_eda_all() on a
    #      SchemaValidation instance which works via inheritance but is misleading.
    schema_validator = SchemaValidation(url=CONFIG["url"], limit=CONFIG["limit"])
    schema_validator.load_data().clean_data().run_schema()
 
    # --- Pipeline / business-logic validation ---
    pipeline = PipelineValidation(
        url=CONFIG["url"],
        limit=CONFIG["limit"],
        duplicate_threshold=CONFIG["duplicate_threshold"],
        missing_threshold=CONFIG["missing_threshold"],
    )
    pipeline.load_data()\
        .clean_data()\
        .run_pipeline_validation()