# VinCense PPG → Cuffless BP project

## Fixed rules (never break)
- Target = RMS cuff BP (column BPCalibrationRef, "SBP/DBP"). Device BP (BPSystolic/BPDiastolic) is NEVER a target or feature.
- Never show a cuffless BP number unless it comes from the validated saved model. No fallbacks, no placeholders, no estimates.
- ±10 mmHg is a development target, not a clinical claim. Label all BP output "Research only – not for clinical use".
- Do not use the separate ~1,000-sample online dataset. Do not pad/resample it to 7,500.
- Do not drop readings for having high BP. Clean only for data integrity and signal quality.
- Keep the source Excel untouched; all cleaning happens in code/outputs.

## Data facts (checked 29 Sep 2026, sheet "VinCense Readings")
- 224 readings, 7,500 samples @ 500 Hz, split over 5 "PPGValue cycle N" columns (contiguous, comma-separated).
- A person = PatientId + PatientName (trimmed, lowercase). PatientId alone is NOT unique: id "test" holds 5 people,
  ids 9/15/24/28/29 have 2 names each. This gives 39 people, matching the README count.
- 188 of 224 readings share their RMS value with another reading of the same person → copy-forward references.
  Treat each (person, RMS value) group as ONE cuff measurement for training and evaluation.
- 2 exact duplicate recordings (identical PPG). Keep the first only.
- Column "bp difference " has a trailing space; strip all column names.
- Timestamps: most are Excel date-times in India time, 8 are ISO strings ending in "Z" (UTC). Convert those to IST.
- The raw IR pulse is inverted (falls on systole). Flip it before extracting morphology.

## Working style (save tokens)
- Never print raw PPG arrays, whole dataframes or whole files. Print shapes, heads (≤10 rows) and summary stats only.
- Put analysis in scripts under `bp_model/` and run them; don't paste long code into chat.
- Commit to git at the end of each phase with a clear message. Stop and summarise in ≤10 lines after each phase.
