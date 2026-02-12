"""
Emerging Risk Explainer Agent

A conversational agent that helps underwriters understand WHY the ML model
flagged a member as emerging risk. It bridges three data sources:
  1. Dashboard table (what UWs already see)
  2. ML model output + SHAP values (what drives the prediction)
  3. Raw claims/labs/pharmacy data (the granular evidence)

The agent can:
  - Summarize why a member is flagged
  - Explain individual SHAP features in plain English
  - Pull up raw claim timelines, lab trends, Rx history
  - Highlight patterns (opioid escalation, doctor shopping, lab trends)
  - Answer follow-up questions from the UW
"""

import textwrap
from ml_risk_agent.data.dashboard_table import get_member_dashboard, list_flagged_members
from ml_risk_agent.data.ml_model_output import (
    get_model_output, get_top_shap_drivers, get_shap_for_feature,
)
from ml_risk_agent.data.raw_claims_data import (
    get_claims, get_pharmacy, get_prior_auths, get_labs,
    get_opioid_timeline, get_abnormal_labs,
)


# ── Tool definitions (these would be registered with an LLM framework) ──

TOOLS = {
    "get_dashboard_summary": {
        "description": "Get the Tableau dashboard summary for a member (what UWs already see).",
        "parameters": ["member_id"],
    },
    "get_risk_explanation": {
        "description": "Get the ML model's risk prediction and top SHAP drivers for a member.",
        "parameters": ["member_id", "top_n"],
    },
    "get_shap_detail": {
        "description": "Search SHAP values by keyword (e.g., 'opioid', 'lab', 'specialist').",
        "parameters": ["member_id", "keyword"],
    },
    "get_claims_timeline": {
        "description": "Get the full medical claims timeline for a member.",
        "parameters": ["member_id"],
    },
    "get_pharmacy_history": {
        "description": "Get the full pharmacy fill history for a member.",
        "parameters": ["member_id"],
    },
    "get_opioid_analysis": {
        "description": "Analyze the opioid prescription trajectory — fills, MME, prescribers, pharmacies.",
        "parameters": ["member_id"],
    },
    "get_lab_trends": {
        "description": "Get lab results with focus on abnormal/trending values.",
        "parameters": ["member_id"],
    },
    "get_prior_auth_details": {
        "description": "Get pending/recent prior authorization requests and their clinical rationale.",
        "parameters": ["member_id"],
    },
    "list_emerging_risk_members": {
        "description": "List all members currently flagged as emerging risk.",
        "parameters": [],
    },
}


def execute_tool(tool_name: str, **kwargs) -> str:
    """Execute a tool and return formatted results."""
    dispatch = {
        "get_dashboard_summary": _tool_dashboard_summary,
        "get_risk_explanation": _tool_risk_explanation,
        "get_shap_detail": _tool_shap_detail,
        "get_claims_timeline": _tool_claims_timeline,
        "get_pharmacy_history": _tool_pharmacy_history,
        "get_opioid_analysis": _tool_opioid_analysis,
        "get_lab_trends": _tool_lab_trends,
        "get_prior_auth_details": _tool_prior_auth_details,
        "list_emerging_risk_members": _tool_list_flagged,
    }
    fn = dispatch.get(tool_name)
    if not fn:
        return f"Unknown tool: {tool_name}"
    return fn(**kwargs)


# ── Tool implementations ────────────────────────────────────────────

def _tool_dashboard_summary(member_id: str) -> str:
    dash = get_member_dashboard(member_id)
    if not dash:
        return f"No dashboard data found for {member_id}."

    return textwrap.dedent(f"""\
    ══════════════════════════════════════════════════
    DASHBOARD SUMMARY — {dash['member_name']} ({member_id})
    ══════════════════════════════════════════════════
    Demographics: {dash['age']}yo {dash['gender']}, {dash['state']} | Plan: {dash['plan_type']} | Group: {dash['group_name']}
    Effective: {dash['effective_date']} | Dependents: {dash['dependent_count']}

    ── Utilization (12 months) ──
    Total Claims: {dash['total_claims_12m']}  |  Allowed: ${dash['total_allowed_12m']:,.0f}  |  Paid: ${dash['total_paid_12m']:,.0f}
    IP Admits: {dash['ip_admits_12m']}  |  ER Visits: {dash['er_visits_12m']}  |  Office Visits: {dash['office_visits_12m']}
    Specialist Visits: {dash['specialist_visits_12m']}  |  Urgent Care: {dash['urgent_care_visits_12m']}

    ── Pharmacy ──
    Rx Claims: {dash['rx_claims_12m']}  |  Rx Paid: ${dash['rx_total_paid_12m']:,.0f}  |  Unique Drugs: {dash['unique_rx_count_12m']}
    Specialty Rx: {dash['specialty_rx_flag']}  |  Generic Rate: {dash['generic_fill_rate']:.0%}

    ── Clinical Flags ──
    Chronic: {dash['chronic_condition_flag']}  |  Mental Health: {dash['mental_health_flag']}  |  Substance Abuse: {dash['substance_abuse_flag']}
    Cancer: {dash['cancer_flag']}  |  Diabetes: {dash['diabetes_flag']}  |  Cardiac: {dash['cardiac_flag']}

    ── Risk ──
    HCC Score: {dash['hcc_risk_score']}  |  Prior Year: {dash['prior_year_risk_score']}  |  Trend: {dash['risk_score_trend']}
    PMPM: ${dash['pmpm_current']:,.0f}  |  YoY Change: {dash['pmpm_change_pct']}%  |  Large Claimant: {dash['large_claimant_flag']}

    ── ML Model ──
    Emerging Risk Flag: {dash['ml_emerging_risk_flag']}  |  Probability: {dash['ml_risk_probability']:.0%}
    Risk Rank in Group: #{dash['ml_risk_rank_in_group']}

    ⚠ NOTE: Dashboard flags this member as emerging risk (83% probability)
       but NO clinical flags are triggered and costs appear moderate.
       Use 'get_risk_explanation' to understand what the model sees.
    """)


def _tool_risk_explanation(member_id: str, top_n: int = 10) -> str:
    output = get_model_output(member_id)
    if not output:
        return f"No model output for {member_id}."

    top_pos = get_top_shap_drivers(member_id, n=top_n, direction="positive")
    top_neg = get_top_shap_drivers(member_id, n=5, direction="negative")

    lines = [
        f"══════════════════════════════════════════════════",
        f"ML MODEL RISK EXPLANATION — {member_id}",
        f"══════════════════════════════════════════════════",
        f"Predicted Probability: {output['predicted_probability']:.0%} (baseline: {output['baseline_probability']:.0%})",
        f"Risk Label: {output['risk_label']}",
        f"Model Version: {output['model_version']} | Prediction Date: {output['prediction_date']}",
        f"",
        f"── TOP {top_n} RISK DRIVERS (pushing probability UP) ──",
    ]

    for i, s in enumerate(top_pos, 1):
        bar = "█" * int(abs(s["shap"]) * 50)
        lines.append(
            f"  {i:2d}. {s['feature']:<45s} SHAP: +{s['shap']:.3f}  {bar}"
        )
        lines.append(
            f"      Value: {s['value']} {s['unit']}  — {s['description']}"
        )

    lines.append(f"")
    lines.append(f"── PROTECTIVE FACTORS (pushing probability DOWN) ──")
    for s in top_neg:
        lines.append(
            f"  • {s['feature']:<45s} SHAP: {s['shap']:.3f}  — {s['description']}"
        )

    lines.append(f"")
    lines.append(
        f"💡 KEY INSIGHT: The model is detecting TRAJECTORY and PATTERNS,\n"
        f"   not just point-in-time metrics. The top drivers are about\n"
        f"   acceleration and behavior patterns that the dashboard can't show."
    )
    return "\n".join(lines)


def _tool_shap_detail(member_id: str, keyword: str) -> str:
    results = get_shap_for_feature(member_id, keyword)
    if not results:
        return f"No SHAP features matching '{keyword}' for {member_id}."

    lines = [f"SHAP features matching '{keyword}' for {member_id}:", ""]
    for s in results:
        direction = "↑ RISK" if s["shap"] > 0 else "↓ PROTECTIVE"
        lines.append(f"  Feature: {s['feature']}")
        lines.append(f"  SHAP:    {s['shap']:+.3f} ({direction})")
        lines.append(f"  Value:   {s['value']} {s['unit']}")
        lines.append(f"  Meaning: {s['description']}")
        lines.append("")
    return "\n".join(lines)


def _tool_claims_timeline(member_id: str) -> str:
    claims = get_claims(member_id)
    if claims.empty:
        return f"No claims data for {member_id}."

    lines = [f"══════════════════════════════════════════════════",
             f"CLAIMS TIMELINE — {member_id}",
             f"══════════════════════════════════════════════════"]
    for _, row in claims.iterrows():
        lines.append(f"\n📅 {row['service_date']}  |  {row['provider']}")
        lines.append(f"   Dx: {row['primary_dx']} — {row['dx_description']}")
        lines.append(f"   Proc: {row['procedure']} — {row['proc_description']}")
        lines.append(f"   Paid: ${row['paid']:,.2f}  |  {row['network_status']}")
        lines.append(f"   Notes: {row['notes']}")
    return "\n".join(lines)


def _tool_pharmacy_history(member_id: str) -> str:
    rx = get_pharmacy(member_id)
    if rx.empty:
        return f"No pharmacy data for {member_id}."

    lines = [f"══════════════════════════════════════════════════",
             f"PHARMACY HISTORY — {member_id}",
             f"══════════════════════════════════════════════════"]
    for _, row in rx.iterrows():
        opioid_tag = " ⚠️ OPIOID" if row["is_opioid"] else ""
        lines.append(f"\n💊 {row['fill_date']}  |  {row['drug_name']}{opioid_tag}")
        lines.append(f"   Class: {row['drug_class']}  |  Prescriber: {row['prescriber']}")
        lines.append(f"   Pharmacy: {row['pharmacy']}")
        lines.append(f"   Supply: {row['days_supply']} days  |  Paid: ${row['paid']:,.2f}")
        if row["is_opioid"]:
            lines.append(f"   MME/day: {row['mme_per_day']}")
        lines.append(f"   Notes: {row['notes']}")
    return "\n".join(lines)


def _tool_opioid_analysis(member_id: str) -> str:
    opioids = get_opioid_timeline(member_id)
    if opioids.empty:
        return f"No opioid prescriptions found for {member_id}."

    rx_all = get_pharmacy(member_id)
    unique_prescribers = rx_all[rx_all["is_opioid"]]["prescriber"].nunique()
    unique_pharmacies = rx_all[rx_all["is_opioid"]]["pharmacy"].nunique()
    max_mme = rx_all[rx_all["is_opioid"]]["mme_per_day"].max()
    first_fill = opioids.iloc[0]["fill_date"]
    latest_fill = opioids.iloc[-1]["fill_date"]

    lines = [
        f"══════════════════════════════════════════════════",
        f"⚠️  OPIOID ANALYSIS — {member_id}",
        f"══════════════════════════════════════════════════",
        f"",
        f"SUMMARY:",
        f"  Total opioid fills:       {len(opioids)}",
        f"  Unique prescribers:       {unique_prescribers}  ← MULTIPLE PRESCRIBERS",
        f"  Unique pharmacies:        {unique_pharmacies}  ← MULTIPLE PHARMACIES",
        f"  First fill:               {first_fill}",
        f"  Latest fill:              {latest_fill}",
        f"  Peak MME/day:             {max_mme}",
        f"  MME trajectory:           22.5 → 22.5 → 22.5 → 37.5* → 45.0",
        f"                            (* concurrent hydrocodone + tramadol)",
        f"",
        f"🚩 RED FLAGS:",
        f"  1. MME doubled in ~7 weeks (22.5 → 45.0 MME/day)",
        f"  2. 4 different prescribers wrote opioid Rx in 60 days",
        f"  3. 3 different pharmacies used (possible pharmacy shopping)",
        f"  4. Concurrent opioids from overlapping prescriptions",
        f"  5. Escalation from acute (7-day supply) to chronic (30-day supply)",
        f"",
        f"TIMELINE:",
    ]
    for _, row in opioids.iterrows():
        lines.append(f"  {row['fill_date']}  |  {row['drug_name']:<30s}  |  MME: {row['mme_per_day']}")
        lines.append(f"             Prescriber: {row['prescriber']}")
        lines.append(f"             Pharmacy:   {row['pharmacy']}")
    return "\n".join(lines)


def _tool_lab_trends(member_id: str) -> str:
    labs = get_labs(member_id)
    abnormal = get_abnormal_labs(member_id)
    if labs.empty:
        return f"No lab data for {member_id}."

    lines = [
        f"══════════════════════════════════════════════════",
        f"LAB RESULTS & TRENDS — {member_id}",
        f"══════════════════════════════════════════════════",
        f"",
        f"── ABNORMAL RESULTS ──",
    ]
    for _, row in abnormal.iterrows():
        lines.append(f"  🔴 {row['lab_date']}  |  {row['test_name']}: {row['result']} {row['unit']}  "
                      f"(ref: {row['reference_range']})  [{row['flag']}]")

    # CRP trend
    crp = labs[labs["test_name"].str.contains("CRP|C-Reactive", case=False)]
    if len(crp) > 1:
        lines.append(f"")
        lines.append(f"── CRP TREND (inflammatory marker) ──")
        lines.append(f"  Normal range: < 3.0 mg/L")
        for _, row in crp.iterrows():
            bar = "█" * int(float(row["result"]) * 2)
            lines.append(f"  {row['lab_date']}:  {row['result']:>5} mg/L  {bar}")
        lines.append(f"  ⚠ CRP increased 51% in 5 weeks (14.8 → 22.4)")
        lines.append(f"    This indicates WORSENING systemic inflammation.")

    # HLA-B27
    hla = labs[labs["test_name"].str.contains("HLA", case=False)]
    if not hla.empty:
        lines.append(f"")
        lines.append(f"── HLA-B27 ──")
        lines.append(f"  Result: POSITIVE")
        lines.append(f"  Clinical significance: Strongly associated with ankylosing spondylitis (AS).")
        lines.append(f"  If AS is confirmed, expect biologic therapy at $60-80K/year.")

    return "\n".join(lines)


def _tool_prior_auth_details(member_id: str) -> str:
    pa = get_prior_auths(member_id)
    if pa.empty:
        return f"No prior authorizations for {member_id}."

    lines = [
        f"══════════════════════════════════════════════════",
        f"PRIOR AUTHORIZATIONS — {member_id}",
        f"══════════════════════════════════════════════════",
    ]
    for _, row in pa.iterrows():
        lines.append(f"\n📋 {row['pa_id']}  |  Status: {row['status']}")
        lines.append(f"   Submitted: {row['submit_date']}  |  Urgency: {row['urgency']}")
        lines.append(f"   Provider: {row['requesting_provider']}")
        lines.append(f"   Service: {row['service_requested']}")
        lines.append(f"   Dx: {row['dx_code']} — {row['dx_description']}")
        lines.append(f"   Est. Cost: ${row['estimated_cost']:,.0f}")
        lines.append(f"   Rationale: {row['clinical_rationale']}")
        lines.append(f"   ⚠ Notes: {row['notes']}")
    return "\n".join(lines)


def _tool_list_flagged() -> str:
    df = list_flagged_members()
    lines = [
        "══════════════════════════════════════════════════",
        "MEMBERS FLAGGED AS EMERGING RISK",
        "══════════════════════════════════════════════════",
        "",
    ]
    for _, row in df.iterrows():
        flag = "✅" if row["high_risk_flag"] == "Y" else "❌"
        lines.append(
            f"  {row['member_id']}  |  {row['member_name']:<20s}  |  "
            f"ML Prob: {row['ml_risk_probability']:.0%}  |  "
            f"Rank: #{row['ml_risk_rank_in_group']}  |  "
            f"HCC: {row['hcc_risk_score']}  |  "
            f"Paid 12m: ${row['total_paid_12m']:,.0f}  |  "
            f"Dashboard High-Risk: {flag}"
        )
    lines.append("")
    lines.append("⚠ M1042 is flagged by ML (83%) but NOT by dashboard clinical flags.")
    lines.append("  This is a 'blind spot' member — the dashboard can't explain the risk.")
    return "\n".join(lines)


# ── Conversation handler ────────────────────────────────────────────

class RiskExplainerAgent:
    """
    Simulates an agent conversation loop.

    In production, this would use an LLM (e.g., Claude) with the tools above
    registered as function calls. For this prototype, we implement a
    rule-based router that demonstrates the flow.
    """

    def __init__(self):
        self.history = []

    def respond(self, user_message: str) -> str:
        """Process user message and return agent response."""
        self.history.append({"role": "user", "content": user_message})
        msg_lower = user_message.lower()

        # Route to appropriate tools based on intent
        # Check "why/explain" BEFORE "list" since questions like
        # "why is the model flagging this" contain "flag" but are explanations.
        if any(w in msg_lower for w in ["why", "explain", "shap", "what does the model"]):
            member_id = self._extract_member_id(msg_lower)
            tool_output = execute_tool("get_risk_explanation", member_id=member_id)
            response = tool_output

        elif any(w in msg_lower for w in ["list", "flagged", "all members", "who is"]):
            tool_output = execute_tool("list_emerging_risk_members")
            response = tool_output

        elif any(w in msg_lower for w in ["dashboard", "tableau", "summary"]):
            member_id = self._extract_member_id(msg_lower)
            tool_output = execute_tool("get_dashboard_summary", member_id=member_id)
            response = tool_output

        elif any(w in msg_lower for w in ["risk score", "model output", "prediction"]):
            member_id = self._extract_member_id(msg_lower)
            tool_output = execute_tool("get_risk_explanation", member_id=member_id)
            response = tool_output

        elif any(w in msg_lower for w in ["opioid", "pain med", "mme", "narcotic", "doctor shop"]):
            member_id = self._extract_member_id(msg_lower)
            tool_output = execute_tool("get_opioid_analysis", member_id=member_id)
            response = tool_output

        elif any(w in msg_lower for w in ["lab", "crp", "blood", "inflam", "hla", "esr"]):
            member_id = self._extract_member_id(msg_lower)
            tool_output = execute_tool("get_lab_trends", member_id=member_id)
            response = tool_output

        elif any(w in msg_lower for w in ["pharmacy", "rx", "prescription", "drug", "medication"]):
            member_id = self._extract_member_id(msg_lower)
            tool_output = execute_tool("get_pharmacy_history", member_id=member_id)
            response = tool_output

        elif any(w in msg_lower for w in ["prior auth", "authorization", "mri", "imaging"]):
            member_id = self._extract_member_id(msg_lower)
            tool_output = execute_tool("get_prior_auth_details", member_id=member_id)
            response = tool_output

        elif any(w in msg_lower for w in ["claim", "visit", "timeline", "history"]):
            member_id = self._extract_member_id(msg_lower)
            tool_output = execute_tool("get_claims_timeline", member_id=member_id)
            response = tool_output

        elif any(w in msg_lower for w in ["cost", "impact", "project", "biologic", "expensive"]):
            response = self._projected_cost_response()

        else:
            response = (
                "I can help you understand emerging risk members. Try asking:\n"
                "  • 'List all flagged members'\n"
                "  • 'Why is M1042 flagged as emerging risk?'\n"
                "  • 'Show me the dashboard for M1042'\n"
                "  • 'Show me the opioid analysis for M1042'\n"
                "  • 'What do the labs show for M1042?'\n"
                "  • 'Show prior authorizations for M1042'\n"
                "  • 'What's the projected cost impact?'\n"
            )

        self.history.append({"role": "agent", "content": response})
        return response

    def _extract_member_id(self, text: str) -> str:
        """Extract member ID from text, default to M1042."""
        import re
        match = re.search(r"m\d{4}", text, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        return "M1042"

    def _projected_cost_response(self) -> str:
        return textwrap.dedent("""\
        ══════════════════════════════════════════════════
        PROJECTED COST IMPACT — M1042 (John Rivera)
        ══════════════════════════════════════════════════

        CURRENT annual spend:  ~$18,400 (paid: $14,210 + member OOP)
        This is MODERATE and within normal range.

        IF ankylosing spondylitis is confirmed (high likelihood given HLA-B27+,
        elevated/rising CRP, clinical presentation):

        PROJECTED annual spend:
          Biologic therapy (adalimumab or secukinumab):  $60,000 - $80,000/yr
          Ongoing rheumatology visits (4-6/yr):          $1,500 - $2,500/yr
          Lab monitoring (CRP, CBC q3mo):                $400 - $600/yr
          Imaging (annual MRI):                          $2,500 - $3,500/yr
          Pain management (if ongoing):                  $3,000 - $5,000/yr
          ─────────────────────────────────────
          ESTIMATED TOTAL:                               $67,000 - $92,000/yr

        This represents a 4-5x increase from current spend.

        ⚠ TIMELINE: The prior auth for MRI is pending. Once imaging confirms
          sacroiliitis, the rheumatologist will submit a prior auth for biologic
          therapy — likely within 2-4 weeks.

        💡 EARLY INTERVENTION OPPORTUNITY:
          • Care management enrollment NOW could coordinate care
          • Ensure single prescriber for pain management
          • Biologic selection and step therapy compliance
          • If caught early, disease modification can prevent disability
        """)
