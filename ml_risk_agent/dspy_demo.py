"""
DSPy Explanation Demo
=====================
Demonstrates the three DSPy usage modes for generating underwriter explanations.

Run:
  # Requires OPENAI_API_KEY (or configure another LM below)

  # Option A — no optimization (instant, good quality)
  python -m ml_risk_agent.dspy_demo

  # Option B — LLM-as-judge optimization (better quality, ~10–15 min on light mode)
  python -m ml_risk_agent.dspy_demo --optimize

  # Load a previously compiled module
  python -m ml_risk_agent.dspy_demo --load models/risk_explainer.json
"""

import argparse
import os
import sys
import textwrap

import dspy

from ml_risk_agent.dspy_explainer import (
    RiskExplainerModule,
    build_from_synthetic_data,
    load_optimized_explainer,
    optimize_explainer,
    run_explanations,
)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _divider(title: str = "", width: int = 68) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(
        textwrap.fill(line, width=90, initial_indent=prefix, subsequent_indent=prefix)
        if line.strip() else ""
        for line in text.splitlines()
    )


def print_result(result: dict) -> None:
    mid = result["member_id"]

    print(f"\n{'═' * 68}")
    print(f"  MEMBER: {mid}")
    print(f"{'═' * 68}")

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    _divider("EXPLANATION")
    print(_wrap(result["explanation"]))

    _divider("KEY RISK FACTORS")
    print(_wrap(result["key_risk_factors"]))

    _divider("SUGGESTED ACTIONS")
    print(_wrap(result["suggested_actions"]))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DSPy Emerging Risk Explanation Demo")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run MIPROv2 LLM-as-judge optimization before generating explanations",
    )
    parser.add_argument(
        "--load",
        metavar="PATH",
        default=None,
        help="Load a previously compiled module from PATH (skips optimization)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default="models/risk_explainer.json",
        help="Where to save the compiled module after optimization (default: models/risk_explainer.json)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="LiteLLM model string, e.g. openai/gpt-4o or anthropic/claude-opus-4-6",
    )
    args = parser.parse_args()

    # ── Configure language model ──────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "[error] Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.\n"
            "        Example: export OPENAI_API_KEY=sk-...\n"
            "        Or use --model anthropic/claude-opus-4-6 with ANTHROPIC_API_KEY."
        )
        sys.exit(1)

    lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)
    print(f"[config] Using model: {args.model}")

    # ── Load synthetic data ───────────────────────────────────────────────────
    print("[data]   Building member data from synthetic dataset ...")
    members = build_from_synthetic_data()
    emerging = {mid: m for mid, m in members.items() if m.is_emerging_risk}
    print(f"[data]   {len(members)} members total, {len(emerging)} flagged as emerging risk")

    # ── Print aggregated SHAP to show how it's computed ───────────────────────
    print()
    _divider("HOW AGGREGATED SHAP IS COMPUTED (example: M1042)")
    m = members.get("M1042")
    if m:
        print("  Raw SHAP → grouped → summed by absolute value → ranked\n")
        for group in m.aggregated_shap:
            contrib_str = ", ".join(
                f"{feat}={shap:+.3f}"
                for feat, shap in list(group.get("contributors", {}).items())[:3]
            )
            if len(group.get("contributors", {})) > 3:
                contrib_str += f" ... +{len(group['contributors']) - 3} more"
            print(f"  Rank {group['rank']}: {group['group']}")
            print(f"    Aggregated SHAP = {group['aggregated_shap']:.4f}")
            print(f"    Driven by: {contrib_str}")
            print()

    # ── Build/load the explainer ──────────────────────────────────────────────
    print()
    if args.load:
        print(f"[module] Loading compiled module from {args.load} ...")
        explainer = load_optimized_explainer(args.load)

    elif args.optimize:
        print("[module] Running MIPROv2 optimization (LLM-as-judge, no gold labels needed) ...")
        print("         This will take ~10–15 minutes on 'light' mode.")
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        explainer = optimize_explainer(members, save_path=args.save)

    else:
        print("[module] Using uncompiled module (Option A — no optimization).")
        print("         Pass --optimize to run LLM-as-judge optimization.")
        explainer = RiskExplainerModule()

    # ── Generate explanations ────────────────────────────────────────────────
    print()
    _divider("GENERATING UNDERWRITER EXPLANATIONS")
    results = run_explanations(members, explainer)

    # ── Print results ────────────────────────────────────────────────────────
    print()
    _divider("RESULTS")
    for result in results.values():
        print_result(result)

    print()
    print("═" * 68)
    print("  DEMO COMPLETE")
    print("═" * 68)
    print()
    print("  What these explanations do:")
    print("  • Bridge ML signals → underwriter-visible dashboard numbers")
    print("  • Explain WHY aggregated SHAP groups indicate emerging risk")
    print("  • Give concrete next steps grounded in dashboard data")
    print()
    print("  To improve quality further:")
    print("  • Option B: python -m ml_risk_agent.dspy_demo --optimize")
    print("  • Option C: Write 3–5 ideal example outputs; use BootstrapFewShot")
    print()


if __name__ == "__main__":
    main()
