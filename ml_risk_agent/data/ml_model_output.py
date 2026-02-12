"""
Data Source 2: ML Model Output — risk predictions + SHAP values.

The model has ~150 features. We store:
  - predicted probability
  - top SHAP contributors (positive = pushes toward risk, negative = protective)
  - feature values for each SHAP feature

KEY SCENARIO for M1042:
  The dashboard shows nothing alarming. But the model picks up on:
    1. Rapid acceleration in opioid MME (morphine milligram equivalents) over 60 days
    2. Multiple prescribers for pain meds (doctor shopping signal)
    3. CRP (C-reactive protein) lab values trending sharply upward
    4. New prior auth request for lumbar MRI
    5. 3 specialist visits in 45 days (pain mgmt, ortho, rheumatology)
    6. Gap in PCP visits (hasn't seen PCP in 9 months despite new symptoms)
  None of these are visible on the dashboard individually — they show up
  as "3 specialist visits" and "14 rx claims" which look benign.
"""

# Full SHAP output for member M1042
# Sorted by |SHAP value| descending — top drivers first
SHAP_VALUES_M1042 = [
    # --- TOP POSITIVE DRIVERS (pushing toward emerging risk) ---
    {"feature": "opioid_mme_acceleration_60d",      "shap": +0.142, "value": 3.8,   "unit": "x increase",  "description": "Rate of increase in opioid MME over last 60 days"},
    {"feature": "unique_opioid_prescribers_90d",     "shap": +0.128, "value": 3,     "unit": "prescribers", "description": "Number of distinct providers prescribing opioids in 90 days"},
    {"feature": "crp_trend_slope_90d",               "shap": +0.105, "value": 2.1,   "unit": "mg/L per month", "description": "Rate of CRP increase over 90 days (inflammatory marker)"},
    {"feature": "specialist_visit_acceleration_45d",  "shap": +0.098, "value": 3,     "unit": "visits",     "description": "Specialist visits in last 45 days (vs 0 in prior 45 days)"},
    {"feature": "new_prior_auth_imaging_flag",        "shap": +0.091, "value": 1,     "unit": "flag",       "description": "New prior auth for advanced imaging (MRI/CT) submitted"},
    {"feature": "pcp_visit_gap_days",                 "shap": +0.078, "value": 274,   "unit": "days",       "description": "Days since last PCP visit"},
    {"feature": "pain_dx_new_onset_flag",             "shap": +0.072, "value": 1,     "unit": "flag",       "description": "New pain-related diagnosis code appeared in last 90 days"},
    {"feature": "rx_fills_acceleration_60d",          "shap": +0.065, "value": 2.4,   "unit": "x increase", "description": "Rate of increase in total Rx fills over 60 days"},
    {"feature": "distinct_pharmacies_90d",            "shap": +0.058, "value": 3,     "unit": "pharmacies", "description": "Number of distinct pharmacies used in 90 days"},
    {"feature": "age",                                "shap": +0.045, "value": 47,    "unit": "years",      "description": "Member age"},
    {"feature": "multi_specialist_types_90d",         "shap": +0.042, "value": 3,     "unit": "types",      "description": "Distinct specialist types seen in 90 days"},
    {"feature": "out_of_network_claims_pct_90d",      "shap": +0.038, "value": 0.18,  "unit": "pct",        "description": "Percent of claims out-of-network in last 90 days"},
    {"feature": "er_visit_for_pain_flag",             "shap": +0.035, "value": 1,     "unit": "flag",       "description": "ER visit with pain-related primary diagnosis"},
    {"feature": "muscle_relaxant_new_fill_flag",      "shap": +0.031, "value": 1,     "unit": "flag",       "description": "New muscle relaxant prescription filled"},
    {"feature": "nsaid_concurrent_opioid_flag",       "shap": +0.028, "value": 1,     "unit": "flag",       "description": "Concurrent NSAID + opioid prescriptions active"},

    # --- MODERATE POSITIVE DRIVERS ---
    {"feature": "total_allowed_30d",                  "shap": +0.022, "value": 4200,  "unit": "dollars",    "description": "Total allowed amount in last 30 days"},
    {"feature": "imaging_claims_90d",                 "shap": +0.019, "value": 2,     "unit": "claims",     "description": "Number of imaging claims in 90 days"},
    {"feature": "lab_orders_acceleration_60d",        "shap": +0.017, "value": 1.8,   "unit": "x increase", "description": "Rate of increase in lab orders over 60 days"},
    {"feature": "gabapentin_flag",                    "shap": +0.015, "value": 1,     "unit": "flag",       "description": "Gabapentin (nerve pain med) prescription active"},
    {"feature": "sed_rate_latest",                    "shap": +0.014, "value": 38,    "unit": "mm/hr",      "description": "Latest ESR (sed rate) lab value — elevated"},
    {"feature": "dependent_count",                    "shap": +0.010, "value": 3,     "unit": "dependents", "description": "Number of covered dependents"},

    # --- NEGATIVE DRIVERS (protective factors) ---
    {"feature": "ip_admits_12m",                      "shap": -0.045, "value": 0,     "unit": "admits",     "description": "Zero inpatient admissions in 12 months"},
    {"feature": "cancer_dx_flag",                     "shap": -0.038, "value": 0,     "unit": "flag",       "description": "No cancer diagnosis"},
    {"feature": "cardiac_dx_flag",                    "shap": -0.032, "value": 0,     "unit": "flag",       "description": "No cardiac diagnosis"},
    {"feature": "diabetes_dx_flag",                   "shap": -0.028, "value": 0,     "unit": "flag",       "description": "No diabetes diagnosis"},
    {"feature": "total_paid_12m",                     "shap": -0.025, "value": 14210, "unit": "dollars",    "description": "12-month total paid is moderate"},
    {"feature": "hcc_risk_score",                     "shap": -0.022, "value": 0.92,  "unit": "score",      "description": "HCC risk score near average"},
    {"feature": "mental_health_dx_flag",              "shap": -0.018, "value": 0,     "unit": "flag",       "description": "No mental health diagnosis on record"},
    {"feature": "bmi_latest",                         "shap": -0.012, "value": 26.1,  "unit": "kg/m2",      "description": "BMI slightly overweight but not obese"},
    {"feature": "smoking_flag",                       "shap": -0.010, "value": 0,     "unit": "flag",       "description": "Non-smoker"},
    {"feature": "in_network_pct_12m",                 "shap": -0.008, "value": 0.94,  "unit": "pct",        "description": "High in-network utilization"},
]

MODEL_OUTPUT = {
    "M1042": {
        "member_id": "M1042",
        "predicted_probability": 0.83,
        "risk_label": "EMERGING_RISK",
        "model_version": "v3.2.1",
        "prediction_date": "2025-01-15",
        "baseline_probability": 0.12,  # population baseline
        "shap_values": SHAP_VALUES_M1042,
    },
    "M2201": {
        "member_id": "M2201",
        "predicted_probability": 0.91,
        "risk_label": "EMERGING_RISK",
        "model_version": "v3.2.1",
        "prediction_date": "2025-01-15",
        "baseline_probability": 0.12,
        "shap_values": [
            # M2201 is obvious — high cost, cardiac + diabetes, IP admits
            {"feature": "total_paid_12m",            "shap": +0.18, "value": 72300, "unit": "dollars",  "description": "Very high 12-month paid claims"},
            {"feature": "ip_admits_12m",             "shap": +0.15, "value": 2,     "unit": "admits",   "description": "2 inpatient admissions"},
            {"feature": "cardiac_dx_flag",           "shap": +0.13, "value": 1,     "unit": "flag",     "description": "Active cardiac diagnosis"},
            {"feature": "diabetes_dx_flag",          "shap": +0.11, "value": 1,     "unit": "flag",     "description": "Active diabetes diagnosis"},
            {"feature": "hcc_risk_score",            "shap": +0.10, "value": 2.45,  "unit": "score",    "description": "HCC risk score well above average"},
            {"feature": "pmpm_change_pct",           "shap": +0.09, "value": 46.9,  "unit": "pct",      "description": "46.9% PMPM increase year over year"},
            {"feature": "er_visits_12m",             "shap": +0.07, "value": 5,     "unit": "visits",   "description": "5 ER visits in 12 months"},
            {"feature": "specialty_rx_flag",         "shap": +0.06, "value": 1,     "unit": "flag",     "description": "On specialty medications"},
        ],
    },
}


def get_model_output(member_id: str) -> dict | None:
    return MODEL_OUTPUT.get(member_id)


def get_top_shap_drivers(member_id: str, n: int = 10, direction: str = "positive") -> list[dict]:
    """Return top N SHAP drivers. direction: 'positive', 'negative', or 'all'."""
    output = MODEL_OUTPUT.get(member_id)
    if not output:
        return []
    shaps = output["shap_values"]
    if direction == "positive":
        filtered = [s for s in shaps if s["shap"] > 0]
    elif direction == "negative":
        filtered = [s for s in shaps if s["shap"] < 0]
    else:
        filtered = shaps
    return sorted(filtered, key=lambda x: abs(x["shap"]), reverse=True)[:n]


def get_shap_for_feature(member_id: str, feature_keyword: str) -> list[dict]:
    """Search SHAP values by keyword in feature name or description."""
    output = MODEL_OUTPUT.get(member_id)
    if not output:
        return []
    keyword = feature_keyword.lower()
    return [s for s in output["shap_values"]
            if keyword in s["feature"].lower() or keyword in s["description"].lower()]
