"""
Data Source 3: Raw Claims / Clinical Data
This is the granular, line-level data that UWs do NOT see in the dashboard.
It's what the ML model was trained on and what the agent uses to explain "why".

For M1042 this tells the REAL story:
  - Oct: Routine checkup, everything fine
  - Nov: ER visit for acute lower back pain → opioid Rx #1
  - Nov-Dec: Sees PCP, gets NSAID, referred to orthopedist
  - Dec: Orthopedist visit → second opioid Rx, muscle relaxant, orders labs
  - Dec: Labs show elevated CRP and ESR (inflammation markers)
  - Jan: Pain management specialist → third opioid Rx (higher dose), gabapentin
  - Jan: Prior auth submitted for lumbar MRI
  - Jan: Rheumatology consult → more labs ordered
  - Jan: Filling Rx at 3 different pharmacies

  The dashboard just shows: 3 specialist visits, 14 Rx fills, 1 ER visit.
  It CANNOT show the trajectory, the escalation, or the doctor-shopping pattern.
"""

import pandas as pd

# ── CLAIMS (medical) ────────────────────────────────────────────────
CLAIMS_M1042 = pd.DataFrame([
    {"claim_id": "CLM-90001", "service_date": "2024-10-08", "provider": "Dr. Sarah Chen (PCP)",
     "provider_type": "PCP", "place_of_service": "Office",
     "primary_dx": "Z00.00", "dx_description": "Annual wellness visit",
     "procedure": "99395", "proc_description": "Preventive visit, 40-64 yrs",
     "allowed": 285.00, "paid": 228.00, "network_status": "In-Network",
     "notes": "Routine annual physical. No concerns noted. Labs ordered (lipid panel, CBC, CMP)."},

    {"claim_id": "CLM-90002", "service_date": "2024-10-08", "provider": "Quest Diagnostics",
     "provider_type": "Lab", "place_of_service": "Lab",
     "primary_dx": "Z00.00", "dx_description": "Encounter for general exam",
     "procedure": "80061", "proc_description": "Lipid panel",
     "allowed": 45.00, "paid": 36.00, "network_status": "In-Network",
     "notes": "Routine labs. All within normal limits."},

    {"claim_id": "CLM-90003", "service_date": "2024-11-12", "provider": "Baylor ER - Dallas",
     "provider_type": "ER", "place_of_service": "Emergency Room",
     "primary_dx": "M54.5", "dx_description": "Low back pain",
     "procedure": "99284", "proc_description": "ER visit, high severity",
     "allowed": 1_850.00, "paid": 1_480.00, "network_status": "In-Network",
     "notes": "Acute onset lower back pain. Unable to stand. X-ray negative for fracture. Prescribed hydrocodone/APAP 5/325, ibuprofen. Advised follow-up with PCP."},

    {"claim_id": "CLM-90004", "service_date": "2024-11-19", "provider": "Dr. Sarah Chen (PCP)",
     "provider_type": "PCP", "place_of_service": "Office",
     "primary_dx": "M54.5", "dx_description": "Low back pain",
     "procedure": "99214", "proc_description": "Office visit, moderate complexity",
     "allowed": 195.00, "paid": 156.00, "network_status": "In-Network",
     "notes": "Follow-up for back pain. Not improving with NSAIDs. Referred to orthopedics. Continued hydrocodone."},

    {"claim_id": "CLM-90005", "service_date": "2024-12-03", "provider": "Dr. Mark Torres (Orthopedics)",
     "provider_type": "Specialist - Orthopedics", "place_of_service": "Office",
     "primary_dx": "M54.5", "dx_description": "Low back pain",
     "procedure": "99204", "proc_description": "New patient office visit, moderate-high",
     "allowed": 340.00, "paid": 272.00, "network_status": "In-Network",
     "notes": "New patient eval. Suspects possible disc herniation or inflammatory arthropathy. Orders CRP, ESR, HLA-B27. Prescribes cyclobenzaprine (muscle relaxant). Continues opioid at current dose. Will consider MRI if labs abnormal."},

    {"claim_id": "CLM-90006", "service_date": "2024-12-05", "provider": "LabCorp",
     "provider_type": "Lab", "place_of_service": "Lab",
     "primary_dx": "M54.5", "dx_description": "Low back pain",
     "procedure": "86140", "proc_description": "C-reactive protein (CRP)",
     "allowed": 32.00, "paid": 25.60, "network_status": "In-Network",
     "notes": "CRP result: 14.8 mg/L (ELEVATED — normal < 3.0)"},

    {"claim_id": "CLM-90007", "service_date": "2024-12-05", "provider": "LabCorp",
     "provider_type": "Lab", "place_of_service": "Lab",
     "primary_dx": "M54.5", "dx_description": "Low back pain",
     "procedure": "85652", "proc_description": "Erythrocyte sed rate (ESR)",
     "allowed": 28.00, "paid": 22.40, "network_status": "In-Network",
     "notes": "ESR result: 38 mm/hr (ELEVATED — normal < 20 for males)"},

    {"claim_id": "CLM-90008", "service_date": "2024-12-05", "provider": "LabCorp",
     "provider_type": "Lab", "place_of_service": "Lab",
     "primary_dx": "M54.5", "dx_description": "Low back pain",
     "procedure": "86235", "proc_description": "HLA-B27 antigen",
     "allowed": 65.00, "paid": 52.00, "network_status": "In-Network",
     "notes": "HLA-B27 result: POSITIVE (associated with ankylosing spondylitis)"},

    {"claim_id": "CLM-90009", "service_date": "2024-12-18", "provider": "Urgent Care - CareNow",
     "provider_type": "Urgent Care", "place_of_service": "Urgent Care",
     "primary_dx": "M54.5", "dx_description": "Low back pain",
     "procedure": "99203", "proc_description": "Urgent care visit",
     "allowed": 210.00, "paid": 168.00, "network_status": "In-Network",
     "notes": "Pain flare-up over weekend. PCP unavailable. Prescribed tramadol 50mg. Patient reports hydrocodone not sufficient."},

    {"claim_id": "CLM-90010", "service_date": "2025-01-02", "provider": "Dr. Anita Patel (Pain Management)",
     "provider_type": "Specialist - Pain Management", "place_of_service": "Office",
     "primary_dx": "M54.5", "dx_description": "Low back pain",
     "procedure": "99204", "proc_description": "New patient office visit",
     "allowed": 380.00, "paid": 304.00, "network_status": "Out-of-Network",
     "notes": "New patient pain mgmt eval. Started gabapentin 300mg TID. Increased opioid to hydrocodone 10/325. Recommends MRI. Notes: patient reports worsening stiffness, worse in morning, improves with activity — classic inflammatory pattern."},

    {"claim_id": "CLM-90011", "service_date": "2025-01-08", "provider": "Dr. Lisa Huang (Rheumatology)",
     "provider_type": "Specialist - Rheumatology", "place_of_service": "Office",
     "primary_dx": "M45.0", "dx_description": "Ankylosing spondylitis, lumbar region",
     "procedure": "99205", "proc_description": "New patient, high complexity",
     "allowed": 420.00, "paid": 294.00, "network_status": "Out-of-Network",
     "notes": "New rheumatology consult. Suspects ankylosing spondylitis given HLA-B27+, elevated CRP/ESR, clinical presentation. Orders repeat CRP, ANA, RF, and lumbar MRI with contrast. Will likely need biologic therapy (adalimumab/secukinumab) — est. $60-80K/year."},

    {"claim_id": "CLM-90012", "service_date": "2025-01-10", "provider": "Quest Diagnostics",
     "provider_type": "Lab", "place_of_service": "Lab",
     "primary_dx": "M45.0", "dx_description": "Ankylosing spondylitis",
     "procedure": "86140", "proc_description": "C-reactive protein (CRP)",
     "allowed": 32.00, "paid": 25.60, "network_status": "In-Network",
     "notes": "CRP result: 22.4 mg/L (ELEVATED, up from 14.8 on 12/05) — worsening inflammation"},

    {"claim_id": "CLM-90013", "service_date": "2025-01-10", "provider": "Quest Diagnostics",
     "provider_type": "Lab", "place_of_service": "Lab",
     "primary_dx": "M45.0", "dx_description": "Ankylosing spondylitis",
     "procedure": "86039", "proc_description": "Antinuclear antibody (ANA)",
     "allowed": 48.00, "paid": 38.40, "network_status": "In-Network",
     "notes": "ANA result: Negative (helps rule out lupus/RA)"},
])

# ── PHARMACY ────────────────────────────────────────────────────────
PHARMACY_M1042 = pd.DataFrame([
    {"rx_id": "RX-50001", "fill_date": "2024-10-08", "drug_name": "Atorvastatin 20mg",
     "drug_class": "Statin", "prescriber": "Dr. Sarah Chen",
     "pharmacy": "CVS #4421 - Dallas", "days_supply": 90, "quantity": 90,
     "paid": 12.00, "is_opioid": False, "mme_per_day": 0,
     "notes": "Maintenance statin. Ongoing."},

    {"rx_id": "RX-50002", "fill_date": "2024-10-08", "drug_name": "Lisinopril 10mg",
     "drug_class": "ACE Inhibitor", "prescriber": "Dr. Sarah Chen",
     "pharmacy": "CVS #4421 - Dallas", "days_supply": 90, "quantity": 90,
     "paid": 8.00, "is_opioid": False, "mme_per_day": 0,
     "notes": "Maintenance BP med. Ongoing."},

    {"rx_id": "RX-50003", "fill_date": "2024-11-12", "drug_name": "Hydrocodone/APAP 5/325mg",
     "drug_class": "Opioid Analgesic", "prescriber": "ER Physician - Baylor",
     "pharmacy": "CVS #4421 - Dallas", "days_supply": 7, "quantity": 21,
     "paid": 15.00, "is_opioid": True, "mme_per_day": 22.5,
     "notes": "First opioid Rx. Post-ER for acute back pain. MME: 22.5/day"},

    {"rx_id": "RX-50004", "fill_date": "2024-11-12", "drug_name": "Ibuprofen 800mg",
     "drug_class": "NSAID", "prescriber": "ER Physician - Baylor",
     "pharmacy": "CVS #4421 - Dallas", "days_supply": 14, "quantity": 42,
     "paid": 6.00, "is_opioid": False, "mme_per_day": 0,
     "notes": "NSAID for inflammation."},

    {"rx_id": "RX-50005", "fill_date": "2024-11-22", "drug_name": "Hydrocodone/APAP 5/325mg",
     "drug_class": "Opioid Analgesic", "prescriber": "Dr. Sarah Chen",
     "pharmacy": "CVS #4421 - Dallas", "days_supply": 14, "quantity": 42,
     "paid": 18.00, "is_opioid": True, "mme_per_day": 22.5,
     "notes": "Refill from PCP. Same dose. MME: 22.5/day"},

    {"rx_id": "RX-50006", "fill_date": "2024-12-03", "drug_name": "Cyclobenzaprine 10mg",
     "drug_class": "Muscle Relaxant", "prescriber": "Dr. Mark Torres",
     "pharmacy": "CVS #4421 - Dallas", "days_supply": 30, "quantity": 60,
     "paid": 14.00, "is_opioid": False, "mme_per_day": 0,
     "notes": "Muscle relaxant added by orthopedist."},

    {"rx_id": "RX-50007", "fill_date": "2024-12-08", "drug_name": "Hydrocodone/APAP 5/325mg",
     "drug_class": "Opioid Analgesic", "prescriber": "Dr. Mark Torres",
     "pharmacy": "Walgreens #1187 - Plano",  # <-- DIFFERENT pharmacy
     "days_supply": 14, "quantity": 42,
     "paid": 18.00, "is_opioid": True, "mme_per_day": 22.5,
     "notes": "Second prescriber for opioids. Different pharmacy than usual. MME: 22.5/day"},

    {"rx_id": "RX-50008", "fill_date": "2024-12-18", "drug_name": "Tramadol 50mg",
     "drug_class": "Opioid Analgesic", "prescriber": "Urgent Care Physician",
     "pharmacy": "Walmart Pharmacy - Garland",  # <-- THIRD pharmacy
     "days_supply": 10, "quantity": 30,
     "paid": 11.00, "is_opioid": True, "mme_per_day": 15.0,
     "notes": "Third prescriber. Third pharmacy. Urgent care visit. MME: 15/day (concurrent with hydrocodone = 37.5 total MME)"},

    {"rx_id": "RX-50009", "fill_date": "2025-01-02", "drug_name": "Hydrocodone/APAP 10/325mg",
     "drug_class": "Opioid Analgesic", "prescriber": "Dr. Anita Patel",
     "pharmacy": "Walgreens #1187 - Plano",
     "days_supply": 30, "quantity": 90,
     "paid": 32.00, "is_opioid": True, "mme_per_day": 45.0,
     "notes": "DOSE DOUBLED. Now 10mg TID. Fourth prescriber total. MME jumped to 45/day."},

    {"rx_id": "RX-50010", "fill_date": "2025-01-02", "drug_name": "Gabapentin 300mg",
     "drug_class": "Anticonvulsant/Nerve Pain", "prescriber": "Dr. Anita Patel",
     "pharmacy": "Walgreens #1187 - Plano",
     "days_supply": 30, "quantity": 90,
     "paid": 18.00, "is_opioid": False, "mme_per_day": 0,
     "notes": "Gabapentin for neuropathic pain component."},

    {"rx_id": "RX-50011", "fill_date": "2025-01-08", "drug_name": "Atorvastatin 20mg",
     "drug_class": "Statin", "prescriber": "Dr. Sarah Chen",
     "pharmacy": "CVS #4421 - Dallas", "days_supply": 90, "quantity": 90,
     "paid": 12.00, "is_opioid": False, "mme_per_day": 0,
     "notes": "Maintenance refill."},

    {"rx_id": "RX-50012", "fill_date": "2025-01-08", "drug_name": "Lisinopril 10mg",
     "drug_class": "ACE Inhibitor", "prescriber": "Dr. Sarah Chen",
     "pharmacy": "CVS #4421 - Dallas", "days_supply": 90, "quantity": 90,
     "paid": 8.00, "is_opioid": False, "mme_per_day": 0,
     "notes": "Maintenance refill."},
])

# ── PRIOR AUTHORIZATIONS ───────────────────────────────────────────
PRIOR_AUTHS_M1042 = pd.DataFrame([
    {"pa_id": "PA-70001", "submit_date": "2025-01-03", "status": "Pending",
     "requesting_provider": "Dr. Anita Patel (Pain Management)",
     "service_requested": "MRI Lumbar Spine with and without Contrast",
     "cpt_code": "72148/72149", "dx_code": "M54.5",
     "dx_description": "Low back pain",
     "clinical_rationale": "47yo male with progressive low back pain x 2 months, "
                           "failed conservative therapy (NSAIDs, muscle relaxants, PT referral). "
                           "HLA-B27 positive, CRP 14.8, ESR 38. R/O disc herniation vs ankylosing spondylitis.",
     "estimated_cost": 3_200.00,
     "urgency": "Urgent",
     "notes": "If MRI confirms AS, expect prior auth for biologic therapy to follow."},

    {"pa_id": "PA-70002", "submit_date": "2025-01-09", "status": "Submitted - Under Review",
     "requesting_provider": "Dr. Lisa Huang (Rheumatology)",
     "service_requested": "MRI Lumbar Spine with Contrast + MRI Sacroiliac Joints",
     "cpt_code": "72149/72196", "dx_code": "M45.0",
     "dx_description": "Ankylosing spondylitis, lumbar region",
     "clinical_rationale": "Suspected ankylosing spondylitis. HLA-B27+, CRP now 22.4 (up from 14.8), "
                           "ESR 38. Classic inflammatory back pain pattern. Need sacroiliac joint imaging "
                           "to confirm diagnosis per modified NY criteria. If confirmed, will initiate "
                           "biologic therapy.",
     "estimated_cost": 5_800.00,
     "urgency": "Urgent",
     "notes": "IMPORTANT: If AS confirmed, biologic therapy (adalimumab/secukinumab) will be "
              "~$60,000-$80,000/year. This member will transition from low-cost to high-cost."},
])

# ── LAB RESULTS (timeline) ─────────────────────────────────────────
LABS_M1042 = pd.DataFrame([
    # Routine labs from Oct wellness visit — all normal
    {"lab_date": "2024-10-08", "test_name": "Total Cholesterol", "result": 198, "unit": "mg/dL",
     "reference_range": "<200", "flag": "Normal", "ordering_provider": "Dr. Sarah Chen"},
    {"lab_date": "2024-10-08", "test_name": "LDL Cholesterol", "result": 118, "unit": "mg/dL",
     "reference_range": "<130", "flag": "Normal", "ordering_provider": "Dr. Sarah Chen"},
    {"lab_date": "2024-10-08", "test_name": "HDL Cholesterol", "result": 52, "unit": "mg/dL",
     "reference_range": ">40", "flag": "Normal", "ordering_provider": "Dr. Sarah Chen"},
    {"lab_date": "2024-10-08", "test_name": "Glucose (fasting)", "result": 94, "unit": "mg/dL",
     "reference_range": "70-100", "flag": "Normal", "ordering_provider": "Dr. Sarah Chen"},
    {"lab_date": "2024-10-08", "test_name": "CBC - WBC", "result": 7.2, "unit": "K/uL",
     "reference_range": "4.5-11.0", "flag": "Normal", "ordering_provider": "Dr. Sarah Chen"},
    {"lab_date": "2024-10-08", "test_name": "Creatinine", "result": 1.0, "unit": "mg/dL",
     "reference_range": "0.7-1.3", "flag": "Normal", "ordering_provider": "Dr. Sarah Chen"},

    # Dec labs ordered by orthopedist — RED FLAGS
    {"lab_date": "2024-12-05", "test_name": "C-Reactive Protein (CRP)", "result": 14.8, "unit": "mg/L",
     "reference_range": "<3.0", "flag": "HIGH", "ordering_provider": "Dr. Mark Torres"},
    {"lab_date": "2024-12-05", "test_name": "ESR (Sed Rate)", "result": 38, "unit": "mm/hr",
     "reference_range": "<20", "flag": "HIGH", "ordering_provider": "Dr. Mark Torres"},
    {"lab_date": "2024-12-05", "test_name": "HLA-B27", "result": "Positive", "unit": "",
     "reference_range": "Negative", "flag": "ABNORMAL", "ordering_provider": "Dr. Mark Torres"},

    # Jan labs ordered by rheumatologist — WORSENING
    {"lab_date": "2025-01-10", "test_name": "C-Reactive Protein (CRP)", "result": 22.4, "unit": "mg/L",
     "reference_range": "<3.0", "flag": "HIGH", "ordering_provider": "Dr. Lisa Huang"},
    {"lab_date": "2025-01-10", "test_name": "ANA (Antinuclear Antibody)", "result": "Negative", "unit": "",
     "reference_range": "Negative", "flag": "Normal", "ordering_provider": "Dr. Lisa Huang"},
    {"lab_date": "2025-01-10", "test_name": "RF (Rheumatoid Factor)", "result": 8, "unit": "IU/mL",
     "reference_range": "<14", "flag": "Normal", "ordering_provider": "Dr. Lisa Huang"},
])


def get_claims(member_id: str) -> pd.DataFrame:
    if member_id == "M1042":
        return CLAIMS_M1042
    return pd.DataFrame()


def get_pharmacy(member_id: str) -> pd.DataFrame:
    if member_id == "M1042":
        return PHARMACY_M1042
    return pd.DataFrame()


def get_prior_auths(member_id: str) -> pd.DataFrame:
    if member_id == "M1042":
        return PRIOR_AUTHS_M1042
    return pd.DataFrame()


def get_labs(member_id: str) -> pd.DataFrame:
    if member_id == "M1042":
        return LABS_M1042
    return pd.DataFrame()


def get_opioid_timeline(member_id: str) -> pd.DataFrame:
    """Extract opioid-specific Rx fills with MME tracking."""
    rx = get_pharmacy(member_id)
    if rx.empty:
        return pd.DataFrame()
    opioids = rx[rx["is_opioid"]].copy()
    opioids = opioids.sort_values("fill_date")
    return opioids[["fill_date", "drug_name", "prescriber", "pharmacy",
                     "days_supply", "mme_per_day", "notes"]]


def get_abnormal_labs(member_id: str) -> pd.DataFrame:
    """Return only abnormal lab results."""
    labs = get_labs(member_id)
    if labs.empty:
        return pd.DataFrame()
    return labs[labs["flag"] != "Normal"]
