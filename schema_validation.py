from config import CONFIG
from eda import Covid_EDA
import pandera.pandas as pa
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SchemaValidation(Covid_EDA):
    """
    Child class that inherits methods and attributes from parent class Covid_EDA
    *SchemaValidation class checks runtime data against a predefined blueprint (schema) to ensure it satisfies specific data types, keys, and constraints. 
    *It acts as a safety contract for your data, ensuring that external inputs like API responses, configuration files, or user forms are structured correctly before your code processes them.
    Validates the loaded DataFrame against a predefined schema using pandera, checking column names, data types and nullability
    """
    def __init__(self, url, limit):
        super().__init__(url, limit)            # passed url and limit up to Covid_EDA

    def validate_schema(self):
        # Define the expected schema for the dataset
        # A schema acts like a blueprint that describes
        # what columns should exist and what data types
        # they should contain
        """Validate the dataframe against predefined pandera schema"""

        # validating schema in date column
        schema_dict = {
            "case_month": pa.Column(
                pa.DateTime,
                nullable=True
            )
        }

        # columns expected to have integer value data
        num_columns = [
            "county_fips_code",
            "state_fips_code",
        ]
        for col in num_columns:
            if col in self.df.columns:
                schema_dict[col] = pa.Column(
                    pa.Int64,
                    nullable=True
            )
        
        # columns expected to contain float values
        float_columns = [
            "case_positive_specimen",
            "case_onset_interval",
        ]
        for col in float_columns:
            if col in self.df.columns:
                schema_dict[col] = pa.Column(
                    float,
                    nullable=True
            )

        # columns expected to contain text values 
        text_columns = [
            "res_state", "res_county", "age_group", "sex", "race", 
            "ethnicity", "process", "exposure_yn", "current_status", 
            "symptom_status", "hosp_yn", "icu_yn", "death_yn", "underlying_conditions_yn"
        ]
        for col in text_columns:
            if col in self.df.columns:
                schema_dict[col] = pa.Column(
                    str,
                    nullable=True
            )

        schema = pa.DataFrameSchema(schema_dict)

        try:
            schema.validate(
                self.df,
                lazy=True
                )
            print(
            "\n Schema validation passed."
            )
        except Exception as e:
            print(
            "\n Schema validation failed."   
            )
            print(e)
        return self
        
    def run_schema(self):
        """Run the entire pipeline in one call"""
        self.validate_schema()


