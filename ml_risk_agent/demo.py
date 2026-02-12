"""
Demo: The "Aha Moment" — Why You Need an Agent, Not Just a Dashboard

This script simulates a conversation between an underwriter (UW) and the
Emerging Risk Explainer Agent. It demonstrates the exact scenario where
the dashboard CANNOT explain a risk flag, but the agent CAN.

Run: python -m ml_risk_agent.demo
"""

from ml_risk_agent.agent import RiskExplainerAgent


def print_separator():
    print("\n" + "─" * 70 + "\n")


def print_uw(msg: str):
    print(f"🧑‍💼 UNDERWRITER:  {msg}")


def print_agent(msg: str):
    print(f"🤖 AGENT:\n{msg}")


def run_demo():
    agent = RiskExplainerAgent()

    print("=" * 70)
    print("  EMERGING RISK EXPLAINER AGENT — DEMO")
    print("  Scenario: UW reviews ML-flagged members for group GRP-8821")
    print("=" * 70)

    # ── Step 1: UW sees the flagged list ──
    print_separator()
    print_uw("Show me all members flagged as emerging risk.")
    response = agent.respond("List all flagged members")
    print_agent(response)

    # ── Step 2: UW looks at M1042 dashboard — nothing alarming ──
    print_separator()
    print_uw("M1042 is interesting — the ML model says 83% risk but the dashboard "
             "doesn't show any clinical flags. Let me see the dashboard.")
    response = agent.respond("Show me the dashboard summary for M1042")
    print_agent(response)

    # ── Step 3: UW asks THE key question ──
    print_separator()
    print_uw("I don't get it. No chronic conditions, no inpatient stays, "
             "moderate costs, HCC score near average... Why is the model "
             "flagging this member at 83%?")
    response = agent.respond("Why is M1042 flagged as emerging risk?")
    print_agent(response)

    # ── Step 4: UW dives into the opioid pattern ──
    print_separator()
    print_uw("Wait — opioid MME acceleration is the #1 driver? And multiple "
             "prescribers? Show me the opioid analysis.")
    response = agent.respond("Show me the opioid analysis for M1042")
    print_agent(response)

    # ── Step 5: UW checks the labs ──
    print_separator()
    print_uw("And the CRP trend is the #3 driver... what do the labs show?")
    response = agent.respond("Show me the lab trends for M1042")
    print_agent(response)

    # ── Step 6: UW checks prior auths ──
    print_separator()
    print_uw("HLA-B27 positive with worsening CRP... that could be ankylosing "
             "spondylitis. Are there any prior auths pending?")
    response = agent.respond("Show prior authorizations for M1042")
    print_agent(response)

    # ── Step 7: UW asks about cost impact ──
    print_separator()
    print_uw("If this is confirmed AS and they need biologics... what's the "
             "projected cost impact?")
    response = agent.respond("What is the projected cost impact?")
    print_agent(response)

    # ── Conclusion ──
    print_separator()
    print("=" * 70)
    print("  DEMO COMPLETE — THE AHA MOMENT")
    print("=" * 70)
    print("""
    What the DASHBOARD showed:
      ✓ 47yo male, moderate costs ($14K paid)
      ✓ No chronic condition flags
      ✓ HCC risk score near average (0.92)
      ✓ 3 specialist visits, 14 Rx fills, 1 ER visit
      → UW conclusion: "Looks normal. Why is the model flagging this?"

    What the AGENT revealed:
      ⚠ Opioid MME doubled in 7 weeks across 4 prescribers & 3 pharmacies
      ⚠ CRP rising 51% in 5 weeks (14.8 → 22.4) — worsening inflammation
      ⚠ HLA-B27 positive — genetic marker for ankylosing spondylitis
      ⚠ Rheumatologist suspects AS, pending MRI confirmation
      ⚠ If confirmed: biologic therapy at $60-80K/year (4-5x current spend)
      ⚠ Prior auths already submitted for advanced imaging

    The dashboard AGGREGATES hide the TRAJECTORY.
    The agent connects ML signals → raw clinical evidence → actionable insight.

    Without the agent, this member looks like a false positive.
    With the agent, the UW can see a potential $80K/year cost driver
    and intervene EARLY with care management.
    """)


def run_interactive():
    """Run an interactive session where the UW can ask questions."""
    agent = RiskExplainerAgent()
    print("=" * 70)
    print("  EMERGING RISK EXPLAINER AGENT — INTERACTIVE MODE")
    print("  Type your questions. Type 'quit' to exit.")
    print("=" * 70)
    print()

    while True:
        user_input = input("🧑‍💼 You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Session ended.")
            break
        if not user_input:
            continue
        response = agent.respond(user_input)
        print(f"\n🤖 Agent:\n{response}\n")


if __name__ == "__main__":
    import sys
    if "--interactive" in sys.argv:
        run_interactive()
    else:
        run_demo()
