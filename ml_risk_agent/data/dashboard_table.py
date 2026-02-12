"""
Data Source 1: Dashboard Table
~50 data points per member — this is what UWs see in Tableau today.
These are AGGREGATED / SUMMARY metrics. They intentionally hide granular detail.

Scenario: Member M1042 looks unremarkable on the dashboard.
"""

import pandas as pd
from datetime import date

DASHBOARD_DATA = pd.DataFrame([
    {
        # --- Member Demographics ---
        "member_id": "M1042",
        "member_name": "John Rivera",
        "age": 47,
        "gender": "M",
        "state": "TX",
        "zip": "75201",
        "group_id": "GRP-8821",
        "group_name": "Lone Star Manufacturing",
        "plan_type": "PPO",
        "effective_date": "2023-01-01",
        "term_date": None,
        "relationship": "Subscriber",
        "dependent_count": 3,

        # --- Utilization Summary (rolling 12 months) ---
        "total_claims_12m": 34,
        "total_allowed_12m": 18_420.00,
        "total_paid_12m": 14_210.00,
        "member_oop_12m": 4_210.00,
        "ip_admits_12m": 0,
        "ip_days_12m": 0,
        "er_visits_12m": 1,
        "office_visits_12m": 8,
        "specialist_visits_12m": 3,
        "urgent_care_visits_12m": 1,
        "telehealth_visits_12m": 2,

        # --- Pharmacy Summary ---
        "rx_claims_12m": 14,
        "rx_total_paid_12m": 3_850.00,
        "unique_rx_count_12m": 5,
        "specialty_rx_flag": "N",
        "generic_fill_rate": 0.78,

        # --- Clinical Flags (binary) ---
        "chronic_condition_flag": "N",
        "mental_health_flag": "N",
        "substance_abuse_flag": "N",
        "maternity_flag": "N",
        "cancer_flag": "N",
        "diabetes_flag": "N",
        "cardiac_flag": "N",
        "musculoskeletal_flag": "N",

        # --- Risk Scores ---
        "hcc_risk_score": 0.92,  # near average (1.0 = population avg)
        "prior_year_risk_score": 0.88,
        "risk_score_trend": "stable",

        # --- Cost Trends ---
        "pmpm_current": 1_535.00,
        "pmpm_prior_year": 1_410.00,
        "pmpm_change_pct": 8.9,
        "large_claimant_flag": "N",  # threshold: $50k
        "catastrophic_flag": "N",    # threshold: $250k

        # --- Network ---
        "in_network_pct": 0.94,
        "pcp_assigned": "Y",
        "pcp_name": "Dr. Sarah Chen",

        # --- Care Management ---
        "care_mgmt_enrolled": "N",
        "disease_mgmt_enrolled": "N",
        "high_risk_flag": "N",       # <--- Dashboard says NOT high risk

        # --- ML Model Output (summary only) ---
        "ml_emerging_risk_flag": "Y",       # <--- BUT the model says YES
        "ml_risk_probability": 0.83,        # 83% probability
        "ml_risk_rank_in_group": 4,         # 4th highest in their group
    },
    # --- A second member for contrast (clearly high risk on dashboard) ---
    {
        "member_id": "M2201",
        "member_name": "Patricia Walsh",
        "age": 62,
        "gender": "F",
        "state": "TX",
        "zip": "75202",
        "group_id": "GRP-8821",
        "group_name": "Lone Star Manufacturing",
        "plan_type": "PPO",
        "effective_date": "2022-01-01",
        "term_date": None,
        "relationship": "Subscriber",
        "dependent_count": 1,
        "total_claims_12m": 142,
        "total_allowed_12m": 87_600.00,
        "total_paid_12m": 72_300.00,
        "member_oop_12m": 6_500.00,
        "ip_admits_12m": 2,
        "ip_days_12m": 8,
        "er_visits_12m": 5,
        "office_visits_12m": 22,
        "specialist_visits_12m": 14,
        "urgent_care_visits_12m": 3,
        "telehealth_visits_12m": 4,
        "rx_claims_12m": 48,
        "rx_total_paid_12m": 14_200.00,
        "unique_rx_count_12m": 12,
        "specialty_rx_flag": "Y",
        "generic_fill_rate": 0.52,
        "chronic_condition_flag": "Y",
        "mental_health_flag": "N",
        "substance_abuse_flag": "N",
        "maternity_flag": "N",
        "cancer_flag": "N",
        "diabetes_flag": "Y",
        "cardiac_flag": "Y",
        "musculoskeletal_flag": "N",
        "hcc_risk_score": 2.45,
        "prior_year_risk_score": 1.98,
        "risk_score_trend": "increasing",
        "pmpm_current": 6_025.00,
        "pmpm_prior_year": 4_100.00,
        "pmpm_change_pct": 46.9,
        "large_claimant_flag": "Y",
        "catastrophic_flag": "N",
        "in_network_pct": 0.88,
        "pcp_assigned": "Y",
        "pcp_name": "Dr. James Okafor",
        "care_mgmt_enrolled": "Y",
        "disease_mgmt_enrolled": "Y",
        "high_risk_flag": "Y",
        "ml_emerging_risk_flag": "Y",
        "ml_risk_probability": 0.91,
        "ml_risk_rank_in_group": 1,
    },
])


def get_member_dashboard(member_id: str) -> dict | None:
    """Return dashboard row as a dict, or None if not found."""
    row = DASHBOARD_DATA[DASHBOARD_DATA["member_id"] == member_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def list_flagged_members() -> pd.DataFrame:
    """Return all members flagged as emerging risk by the ML model."""
    return DASHBOARD_DATA[DASHBOARD_DATA["ml_emerging_risk_flag"] == "Y"][
        ["member_id", "member_name", "ml_risk_probability", "ml_risk_rank_in_group",
         "hcc_risk_score", "total_paid_12m", "high_risk_flag"]
    ]
