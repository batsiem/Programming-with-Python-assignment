# Constants and Configuration
# set all pipeline thresholds and settings 

CONFIG = {
    "url"                   : "https://data.cdc.gov/resource/n8mc-b4w4.csv",        # CDC API endpoint
    "limit"                 : 5000,                                              # number of rows to load from the CDC API
    "duplicate_threshold"   : 5,                                  # max number of times that a demographic combination can appear
    "missing_threshold"     : 80.0,                                       # max percentage of missing values allowed per column
    "top_counties"          : 10,
    "data_min"              : "2020-01-01",
    "date_max"              : "2024-07-05"
}

# Known CDC enumeration values used by schema and domain checks
CDC_BINARY_VALUES = ["Yes", "No"]

CDC_SEX_VALUES = ["Female", "Male", "Other", "Unknown"]

CDC_CURRENT_STATUS_VALUES = ["Laboratory-confirmed case", "Probable Case"]

CDC_SYMPTOM_STATUS_VALUES = ["Symptomatic", "Asymptomatic"]

CDC_AGE_GROUP_VALUES = [
    "0 -17 years",
    "18 - 49 years",
    "50 - 64 years",
    "65+ years",
    "Unknown"
]

CDC_PROCESS_VALUES = [
    "Clinical evaluation",
    "Surveillance testing",
    "Lab testing only",
]
