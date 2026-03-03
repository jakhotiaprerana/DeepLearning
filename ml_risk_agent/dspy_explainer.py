"""
DSPy-Based Emerging Risk Explanation Generator
================================================
Generates underwriter-friendly explanations for why the ML model flagged a
member as emerging risk.

INPUTS per member (loaded from CSV or the existing synthetic data module):
  1. raw_features      — the ~150 features fed directly to the ML model
  2. raw_shap_values   — per-feature SHAP values (what moves the prediction)
  3. dashboard_features— the aggregated metrics the UW sees on their dashboard
  4. aggregated_shap   — top grouped/aggregated SHAP scores (5 groups)
  5. is_emerging_risk  — flag: True/False

OUTPUTS:
  • explanation        — 3-4 paragraph narrative in plain underwriting language
  • key_risk_factors   — bullet-pointed top 3-5 drivers with dashboard values
  • suggested_actions  — 2-4 specific actions for the UW to take

HOW AGGREGATED SHAP IS BUILT:
  Raw SHAP values are grouped by clinical domain (see SHAP_GROUPS below).
  The aggregated score for a group = sum of raw |SHAP| values within that group.
  This reduces ~150 sparse features into 5 meaningful business concepts.

HOW DASHBOARD FEATURES RELATE TO RAW FEATURES:
  Dashboard metrics are rolling-window aggregations of raw claims/Rx/lab events.
  E.g., "rx_claims_12m" on the dashboard = count of all Rx claims in 12 months,
  but the raw features capture the *rate of change* and *pattern* within that
  window (acceleration, multi-prescriber, etc.) which the dashboard cannot show.

ABOUT DSPy OPTIMIZATION WITHOUT GOLD LABELS:
  DSPy can absolutely be used here without a labeled output dataset.
  Three options (implemented below):

  Option A — No optimization, just structured prompting (USE THIS FIRST):
    Use dspy.ChainOfThought(Signature) directly. DSPy gives you typed I/O,
    automatic few-shot formatting, and clean module composition — all without
    any labeled data.

  Option B — LLM-as-judge optimization (PREFERRED when you have ≥5 members):
    Define a quality metric that uses a separate LLM to score the explanation
    on clarity, accuracy, actionability, and groundedness.
    MIPROv2 then hill-climbs the prompt instructions using this metric.
    No gold outputs needed — the judge replaces them.

  Option C — Manual bootstrap (if you have domain experts write 2-3 examples):
    Write 2-3 ideal explanations by hand, then use BootstrapFewShot.
    This is the fastest path to high quality if a subject-matter expert can
    spend 30 minutes writing example outputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import dspy

# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATION DEFINITIONS
# These define how raw SHAP features map to the 5 aggregated groups the UW sees.
# Adjust these to match your actual feature engineering logic.
# ──────────────────────────────────────────────────────────────────────────────

#: Maps aggregated group name → list of raw feature names that compose it.
#: Aggregated SHAP score = sum(|raw_shap|) for all features in the group.
SHAP_GROUPS: dict[str, list[str]] = {
    "Opioid & Pain Management Risk": [
        "opioid_mme_acceleration_60d",
        "unique_opioid_prescribers_90d",
        "distinct_pharmacies_90d",
        "er_visit_for_pain_flag",
        "muscle_relaxant_new_fill_flag",
        "nsaid_concurrent_opioid_flag",
        "gabapentin_flag",
        "pain_dx_new_onset_flag",
        "rx_fills_acceleration_60d",
    ],
    "Inflammatory / Diagnostic Trajectory": [
        "crp_trend_slope_90d",
        "sed_rate_latest",
        "new_prior_auth_imaging_flag",
        "imaging_claims_90d",
        "lab_orders_acceleration_60d",
    ],
    "Specialist Utilization Surge": [
        "specialist_visit_acceleration_45d",
        "multi_specialist_types_90d",
        "out_of_network_claims_pct_90d",
    ],
    "Care Coordination Gaps": [
        "pcp_visit_gap_days",
    ],
    "Recent Cost Acceleration": [
        "total_allowed_30d",
        "age",
        "dependent_count",
    ],
}

#: Maps dashboard feature name → raw features that feed into it.
#: Used to explain to the UW which raw signals sit behind a dashboard number.
DASHBOARD_TO_RAW: dict[str, list[str]] = {
    "rx_claims_12m": [
        "rx_fills_acceleration_60d",
        "distinct_pharmacies_90d",
        "unique_opioid_prescribers_90d",
    ],
    "specialist_visits_12m": [
        "specialist_visit_acceleration_45d",
        "multi_specialist_types_90d",
    ],
    "er_visits_12m": [
        "er_visit_for_pain_flag",
    ],
    "unique_rx_count_12m": [
        "gabapentin_flag",
        "muscle_relaxant_new_fill_flag",
        "nsaid_concurrent_opioid_flag",
        "opioid_mme_acceleration_60d",
    ],
    "total_allowed_12m": [
        "total_allowed_30d",
    ],
    "total_paid_12m": [
        "total_allowed_30d",
    ],
    "office_visits_12m": [
        "pcp_visit_gap_days",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemberRiskData:
    """All inputs for a single member's risk explanation."""
    member_id: str
    is_emerging_risk: bool

    # What the ML model received
    raw_features: dict[str, float | int | str]
    raw_shap_values: dict[str, float]           # feature → SHAP value

    # What the UW sees on their dashboard
    dashboard_features: dict[str, float | int | str]

    # Top 5 aggregated SHAP groups (pre-computed, or computed by build_aggregated_shap)
    # Each entry: {"group": str, "aggregated_shap": float, "rank": int}
    aggregated_shap: list[dict]

    # Customizable mappings (defaults to module-level constants)
    shap_groups: dict[str, list[str]] = field(default_factory=lambda: SHAP_GROUPS)
    dashboard_to_raw: dict[str, list[str]] = field(default_factory=lambda: DASHBOARD_TO_RAW)


def build_aggregated_shap(
    raw_shap_values: dict[str, float],
    shap_groups: dict[str, list[str]] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Compute aggregated SHAP groups from raw SHAP values.

    Aggregation rule: sum of absolute raw SHAP values within each group.
    This is the most common approach — it measures total influence regardless
    of direction (positive/negative contributions within a group can cancel,
    but the group's overall "pull" is captured by the absolute sum).

    Any raw SHAP feature not covered by the configured groups is collected into
    an automatic "Other Risk Factors" group so no signal is lost.

    Returns top_n groups sorted by aggregated score descending.
    """
    groups = shap_groups or SHAP_GROUPS
    scores: list[dict] = []
    assigned_features: set[str] = set()

    for group_name, group_features in groups.items():
        group_score = sum(
            abs(raw_shap_values.get(feat, 0.0))
            for feat in group_features
        )
        contributors = {
            feat: round(raw_shap_values.get(feat, 0.0), 4)
            for feat in group_features
            if abs(raw_shap_values.get(feat, 0.0)) > 0.001
        }
        assigned_features.update(group_features)
        scores.append({
            "group": group_name,
            "aggregated_shap": round(group_score, 4),
            "contributors": contributors,
        })

    # Catch-all: any feature not in any group that has non-trivial SHAP
    unassigned = {
        feat: round(shap, 4)
        for feat, shap in raw_shap_values.items()
        if feat not in assigned_features and abs(shap) > 0.001
    }
    if unassigned:
        scores.append({
            "group": "Other Risk Factors",
            "aggregated_shap": round(sum(abs(v) for v in unassigned.values()), 4),
            "contributors": unassigned,
        })

    # Sort by aggregated score, assign ranks, return top_n
    scores.sort(key=lambda x: x["aggregated_shap"], reverse=True)
    for rank, entry in enumerate(scores[:top_n], start=1):
        entry["rank"] = rank

    return scores[:top_n]


def build_dashboard_context(
    dashboard_features: dict,
    raw_features: dict,
    raw_shap_values: dict,
    dashboard_to_raw: dict[str, list[str]] | None = None,
) -> dict:
    """
    Build a context dict showing how each dashboard feature is derived from
    raw features, and which raw SHAP values sit behind it.

    This is passed to the LLM so it can explain the gap between what the UW
    sees (aggregated dashboard numbers) and what the model actually detected
    (granular acceleration/pattern features).
    """
    mapping = dashboard_to_raw or DASHBOARD_TO_RAW
    context: dict[str, dict] = {}

    for dash_feat, dash_val in dashboard_features.items():
        raw_feats = mapping.get(dash_feat, [])
        context[dash_feat] = {
            "dashboard_value": dash_val,
            "underlying_raw_features": {
                rf: {
                    "raw_value": raw_features.get(rf),
                    "shap": round(raw_shap_values.get(rf, 0.0), 4),
                }
                for rf in raw_feats
                if rf in raw_features or rf in raw_shap_values
            },
        }

    return context


# ──────────────────────────────────────────────────────────────────────────────
# DSPy SIGNATURES
# ──────────────────────────────────────────────────────────────────────────────

class SummarizeRiskSignals(dspy.Signature):
    """
    You are a senior health insurance actuary.

    You are given structured data for a member flagged as 'emerging risk' by a
    predictive ML model. Your task is to synthesize this data into a coherent
    clinical and actuarial narrative — the 'story' behind the numbers.

    The aggregated SHAP groups tell you WHAT is driving the risk. The dashboard
    context tells you WHY the dashboard alone cannot explain it (the dashboard
    shows totals; the model detected TRAJECTORY and PATTERNS within those totals).

    Write 2-3 short paragraphs that connect the risk signals into a coherent story.
    Focus on what is CHANGING, not just what exists. Use plain language.
    Do not use the words 'SHAP', 'model', 'feature', or 'algorithm'.
    """

    member_id: str = dspy.InputField(
        desc="Member identifier"
    )
    top_5_risk_groups: str = dspy.InputField(
        desc=(
            "JSON of the 5 highest-scoring aggregated risk groups. Each group has: "
            "group name, aggregated_shap score, rank, and the specific raw features "
            "that drove that group's score (with their raw values and individual SHAP)."
        )
    )
    dashboard_context: str = dspy.InputField(
        desc=(
            "JSON showing each dashboard feature (what the UW sees), its value, "
            "and the underlying raw features that feed into it — including those "
            "raw features' values and SHAP scores. This reveals the 'hidden signals' "
            "behind benign-looking dashboard numbers."
        )
    )

    risk_narrative: str = dspy.OutputField(
        desc=(
            "2-3 paragraph narrative explaining the key risk signals in plain language. "
            "Focus on change, trajectory, and patterns over time. "
            "Explicitly note where the dashboard number looks normal but the "
            "underlying pattern is concerning."
        )
    )


class GenerateUWExplanation(dspy.Signature):
    """
    You are a senior actuary writing a risk briefing for an underwriter.

    The underwriter can ONLY see the dashboard features and the top 5 aggregated
    risk groups. They cannot see the raw model features. Your explanation must:

    1. Be entirely grounded in the dashboard values and risk group names they can verify.
    2. Lead with the #1 risk driver (highest aggregated SHAP score).
    3. Explain each risk group in clinical/actuarial terms — never say 'SHAP score',
       'model feature', or 'algorithm'. Instead say things like 'the model is detecting',
       'this pattern indicates', 'the combination of X and Y suggests'.
    4. Use concrete dashboard values (e.g., 'this member had 14 Rx fills, but the
       pattern within those fills indicates...').
    5. Be written at the level of a junior-to-mid underwriter who understands
       insurance concepts but is not a clinician.
    6. Be 3-4 paragraphs. Concise and direct.
    """

    member_id: str = dspy.InputField(desc="Member identifier")
    ml_probability: str = dspy.InputField(
        desc="ML model's predicted probability of emerging risk (e.g., '83%')"
    )
    dashboard_features: str = dspy.InputField(
        desc="JSON of dashboard features and their values — exactly what the UW sees"
    )
    top_5_risk_groups: str = dspy.InputField(
        desc=(
            "JSON of the top 5 aggregated risk groups with their group names and scores. "
            "Ordered by importance (rank 1 = most important). "
            "Each group also includes the specific raw signals driving it."
        )
    )
    risk_narrative: str = dspy.InputField(
        desc="Background narrative connecting the risk signals into a coherent story"
    )

    explanation: str = dspy.OutputField(
        desc=(
            "3-4 paragraph underwriter briefing. "
            "No ML jargon. Reference dashboard values. Lead with the top driver. "
            "Explain what is unusual or concerning about this specific member."
        )
    )
    key_risk_factors: str = dspy.OutputField(
        desc=(
            "Bulleted list of the top 3-5 risk factors in plain English. "
            "Format: '• [Factor]: [specific dashboard value or observation]'. "
            "Example: '• Opioid escalation: 14 Rx fills with multiple prescribers'"
        )
    )
    suggested_actions: str = dspy.OutputField(
        desc=(
            "Bulleted list of 2-4 specific, actionable steps for the underwriter. "
            "Examples: request clinical records, refer to care management, "
            "apply a rating factor, flag for medical review, set a watch period. "
            "Be specific to this member's situation."
        )
    )


class ScoreExplanationQuality(dspy.Signature):
    """
    You are a senior underwriting manager reviewing an AI-generated risk briefing.
    Score it on four dimensions (each 1–5) and provide a single line of feedback.

    Scoring rubric:
      clarity (1–5):       Is it written in plain English an underwriter can act on?
                           5 = no jargon, clear sentences, logical flow
      accuracy (1–5):      Does it correctly reflect the top risk drivers?
                           5 = all top-3 risk groups are addressed and correctly described
      actionability (1–5): Are the suggested actions specific to this member?
                           5 = each action is concrete (not generic) and feasible
      groundedness (1–5):  Does every claim reference a verifiable dashboard value?
                           5 = every risk statement cites a specific number the UW can see
    """

    explanation: str = dspy.InputField(desc="The explanation text to evaluate")
    key_risk_factors: str = dspy.InputField(desc="The key risk factors bullet list")
    suggested_actions: str = dspy.InputField(desc="The suggested actions bullet list")
    top_5_risk_groups: str = dspy.InputField(
        desc="The actual top 5 risk groups (ground truth for accuracy scoring)"
    )
    dashboard_features: str = dspy.InputField(
        desc="The dashboard features available to the UW (ground truth for groundedness)"
    )

    clarity_score: float = dspy.OutputField(desc="Score 1–5 for clarity")
    accuracy_score: float = dspy.OutputField(desc="Score 1–5 for accuracy")
    actionability_score: float = dspy.OutputField(desc="Score 1–5 for actionability")
    groundedness_score: float = dspy.OutputField(desc="Score 1–5 for groundedness")
    feedback: str = dspy.OutputField(desc="One sentence of constructive feedback")


# ──────────────────────────────────────────────────────────────────────────────
# DSPy MODULES
# ──────────────────────────────────────────────────────────────────────────────

class RiskExplainerModule(dspy.Module):
    """
    Two-step DSPy pipeline:
      Step 1 (SummarizeRiskSignals): Build a clinical narrative from raw signals.
      Step 2 (GenerateUWExplanation): Turn that narrative into a UW briefing.

    Using ChainOfThought on both steps forces the LLM to reason through the
    data before writing the output, which significantly improves coherence.
    """

    def __init__(self) -> None:
        super().__init__()
        self.summarizer = dspy.ChainOfThought(SummarizeRiskSignals)
        self.explainer = dspy.ChainOfThought(GenerateUWExplanation)

    def forward(self, member: MemberRiskData) -> dspy.Prediction:
        if not member.is_emerging_risk:
            return dspy.Prediction(
                explanation=f"Member {member.member_id} is not flagged as emerging risk.",
                key_risk_factors="N/A — member is not emerging risk.",
                suggested_actions="No action required at this time.",
            )

        # Build enriched context from raw data
        top_5 = member.aggregated_shap  # already computed and ranked
        dash_context = build_dashboard_context(
            member.dashboard_features,
            member.raw_features,
            member.raw_shap_values,
            member.dashboard_to_raw,
        )

        # Pull predicted probability from raw features if available
        ml_prob = member.raw_features.get("predicted_probability", "N/A")
        ml_prob_str = f"{float(ml_prob):.0%}" if ml_prob != "N/A" else "N/A"

        # Step 1: Build risk narrative (bridges raw signals → aggregated view)
        summary = self.summarizer(
            member_id=member.member_id,
            top_5_risk_groups=json.dumps(top_5, indent=2),
            dashboard_context=json.dumps(dash_context, indent=2),
        )

        # Step 2: Generate underwriter explanation
        result = self.explainer(
            member_id=member.member_id,
            ml_probability=ml_prob_str,
            dashboard_features=json.dumps(member.dashboard_features, indent=2),
            top_5_risk_groups=json.dumps(top_5, indent=2),
            risk_narrative=summary.risk_narrative,
        )

        return result


# ──────────────────────────────────────────────────────────────────────────────
# LLM-AS-JUDGE METRIC  (enables DSPy optimization without gold labels)
# ──────────────────────────────────────────────────────────────────────────────

def explanation_quality_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    LLM-as-judge metric for DSPy's MIPROv2 optimizer.

    No gold-label outputs needed. A separate LLM call scores the explanation
    on four underwriting-relevant dimensions and returns a 0–1 composite score.

    'example' must have fields: top_5_risk_groups, dashboard_features
    (see build_trainset() below for how these are set).
    """
    judge = dspy.ChainOfThought(ScoreExplanationQuality)

    try:
        scores = judge(
            explanation=pred.explanation,
            key_risk_factors=pred.key_risk_factors,
            suggested_actions=pred.suggested_actions,
            top_5_risk_groups=example.top_5_risk_groups,
            dashboard_features=example.dashboard_features,
        )
        composite = (
            float(scores.clarity_score)
            + float(scores.accuracy_score)
            + float(scores.actionability_score)
            + float(scores.groundedness_score)
        ) / 4.0
        return composite / 5.0      # normalize to [0, 1]

    except Exception as exc:
        print(f"[judge] scoring failed: {exc}")
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZATION (Option B — LLM-as-judge, no gold labels needed)
# ──────────────────────────────────────────────────────────────────────────────

def build_trainset(members: dict[str, MemberRiskData]) -> list[dspy.Example]:
    """
    Build a DSPy training set from MemberRiskData objects.
    Examples contain ONLY inputs (no gold outputs).
    The judge metric supplies the signal that replaces gold labels.
    """
    trainset = []
    for member in members.values():
        if not member.is_emerging_risk:
            continue
        example = dspy.Example(
            member=member,
            # Fields used by the judge metric for scoring
            top_5_risk_groups=json.dumps(member.aggregated_shap, indent=2),
            dashboard_features=json.dumps(member.dashboard_features, indent=2),
        ).with_inputs("member")
        trainset.append(example)
    return trainset


def optimize_explainer(
    members: dict[str, MemberRiskData],
    save_path: Optional[str] = None,
    num_trials: int = 15,
) -> RiskExplainerModule:
    """
    Optimize the RiskExplainerModule using MIPROv2 + LLM-as-judge.

    No gold labels required. MIPROv2 will:
      1. Auto-generate candidate instruction variants for each signature.
      2. Score them using explanation_quality_metric (the judge LLM).
      3. Select the instruction set that maximizes average judge score.

    Requires ≥ 5 emerging-risk members for meaningful optimization.
    With fewer members, Option A (no optimization) or Option C (manual examples)
    is recommended instead.

    Args:
        members:     Dict of member_id → MemberRiskData.
        save_path:   If provided, save the compiled module as JSON.
        num_trials:  Number of prompt variants to evaluate. More = better but slower.
                     Use 10–20 for dev, 30–50 for production.
    """
    from dspy.teleprompt import MIPROv2

    trainset = build_trainset(members)
    if len(trainset) < 5:
        print(
            f"[optimize] Only {len(trainset)} training examples found. "
            "Returning uncompiled module. Add more members or use Option C "
            "(manual examples with BootstrapFewShot)."
        )
        return RiskExplainerModule()

    optimizer = MIPROv2(
        metric=explanation_quality_metric,
        auto="light",       # "light" ≈ 10–15 min; "medium" ≈ 30–60 min
        verbose=True,
    )

    compiled = optimizer.compile(
        RiskExplainerModule(),
        trainset=trainset,
        num_trials=num_trials,
        max_labeled_demos=0,    # zero because we have no gold outputs
    )

    if save_path:
        compiled.save(save_path)
        print(f"[optimize] Compiled module saved → {save_path}")

    return compiled


def load_optimized_explainer(load_path: str) -> RiskExplainerModule:
    """Load a previously optimized module (skips re-optimization)."""
    module = RiskExplainerModule()
    module.load(load_path)
    return module


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_explanations(
    members: dict[str, MemberRiskData],
    explainer: Optional[RiskExplainerModule] = None,
) -> dict[str, dict]:
    """
    Generate explanations for all emerging-risk members.

    Args:
        members:   Dict of member_id → MemberRiskData (only emerging-risk ones
                   are processed; others get a 'not flagged' stub).
        explainer: A (possibly optimized) RiskExplainerModule. If None, a fresh
                   uncompiled module is used (Option A — no optimization).

    Returns:
        Dict of member_id → {explanation, key_risk_factors, suggested_actions}.
    """
    if explainer is None:
        explainer = RiskExplainerModule()

    results: dict[str, dict] = {}
    for member_id, member in members.items():
        print(f"  Generating explanation for {member_id} ...", end=" ", flush=True)
        try:
            pred = explainer(member=member)
            results[member_id] = {
                "member_id": member_id,
                "explanation": pred.explanation,
                "key_risk_factors": pred.key_risk_factors,
                "suggested_actions": pred.suggested_actions,
            }
            print("done")
        except Exception as exc:
            print(f"ERROR: {exc}")
            results[member_id] = {"member_id": member_id, "error": str(exc)}

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CSV-BASED LOADER  (for production use with real data files)
# ──────────────────────────────────────────────────────────────────────────────

def load_members_from_csv(
    raw_features_csv: str,
    shap_values_csv: str,
    dashboard_features_csv: str,
    aggregated_shap_csv: str,
    emerging_risk_csv: str,
    member_id_col: str = "member_id",
    shap_groups: Optional[dict[str, list[str]]] = None,
    dashboard_to_raw: Optional[dict[str, list[str]]] = None,
) -> dict[str, MemberRiskData]:
    """
    Load member data from CSV files.

    Expected formats:
      raw_features_csv, shap_values_csv, dashboard_features_csv:
        Wide format — one row per member, columns = feature names.
        Must include a 'member_id' column.

      aggregated_shap_csv:
        Long format — columns: member_id, group, aggregated_shap, rank.
        (This is the output of build_aggregated_shap() serialized to CSV.)

      emerging_risk_csv:
        Columns: member_id, is_emerging_risk (1/0 or True/False).
    """
    import pandas as pd

    raw_df = pd.read_csv(raw_features_csv).set_index(member_id_col)
    shap_df = pd.read_csv(shap_values_csv).set_index(member_id_col)
    dash_df = pd.read_csv(dashboard_features_csv).set_index(member_id_col)
    agg_shap_df = pd.read_csv(aggregated_shap_csv)
    risk_df = pd.read_csv(emerging_risk_csv).set_index(member_id_col)

    members: dict[str, MemberRiskData] = {}
    for member_id in risk_df.index:
        mid = str(member_id)
        is_risk = bool(risk_df.at[member_id, "is_emerging_risk"])

        member_agg = (
            agg_shap_df[agg_shap_df[member_id_col] == mid]
            .sort_values("rank")
            .to_dict("records")
        )

        members[mid] = MemberRiskData(
            member_id=mid,
            is_emerging_risk=is_risk,
            raw_features=raw_df.loc[mid].to_dict() if mid in raw_df.index else {},
            raw_shap_values=shap_df.loc[mid].to_dict() if mid in shap_df.index else {},
            dashboard_features=dash_df.loc[mid].to_dict() if mid in dash_df.index else {},
            aggregated_shap=member_agg,
            shap_groups=shap_groups or SHAP_GROUPS,
            dashboard_to_raw=dashboard_to_raw or DASHBOARD_TO_RAW,
        )

    return members


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: build MemberRiskData from existing synthetic data module
# ──────────────────────────────────────────────────────────────────────────────

def build_from_synthetic_data() -> dict[str, MemberRiskData]:
    """
    Build MemberRiskData objects from the existing synthetic data module
    (ml_risk_agent/data/ml_model_output.py and dashboard_table.py).
    Used for development / demos without needing real CSV files.
    """
    from ml_risk_agent.data.ml_model_output import MODEL_OUTPUT
    from ml_risk_agent.data.dashboard_table import DASHBOARD_DATA

    # DASHBOARD_DATA is a DataFrame; convert to {member_id: row_dict}
    dash_lookup: dict[str, dict] = {
        row["member_id"]: row
        for row in DASHBOARD_DATA.to_dict("records")
    }

    members: dict[str, MemberRiskData] = {}

    for member_id, model_out in MODEL_OUTPUT.items():
        # Raw SHAP values: {feature_name: shap_score}
        raw_shap: dict[str, float] = {
            s["feature"]: s["shap"] for s in model_out["shap_values"]
        }

        # Raw features: {feature_name: value}
        raw_features: dict[str, float | int | str] = {
            s["feature"]: s["value"] for s in model_out["shap_values"]
        }
        # Add model probability so pipeline can surface it in the explanation
        raw_features["predicted_probability"] = model_out["predicted_probability"]

        # Dashboard features — the ~15 key metrics the UW sees
        dash = dash_lookup.get(member_id, {})
        _dash_keys = [
            "total_claims_12m", "total_paid_12m", "ip_admits_12m",
            "er_visits_12m", "specialist_visits_12m", "rx_claims_12m",
            "unique_rx_count_12m", "hcc_risk_score", "pmpm_change_pct",
            "chronic_condition_flag", "mental_health_flag", "specialty_rx_flag",
            "office_visits_12m", "large_claimant_flag", "high_risk_flag",
        ]
        dashboard_features: dict[str, float | int | str] = {
            k: dash[k] for k in _dash_keys if k in dash and dash[k] is not None
        }
        dashboard_features["ml_risk_probability"] = model_out["predicted_probability"]

        # Compute aggregated SHAP from raw SHAP values
        aggregated_shap = build_aggregated_shap(raw_shap)

        members[member_id] = MemberRiskData(
            member_id=member_id,
            is_emerging_risk=(model_out["risk_label"] == "EMERGING_RISK"),
            raw_features=raw_features,
            raw_shap_values=raw_shap,
            dashboard_features=dashboard_features,
            aggregated_shap=aggregated_shap,
        )

    return members
