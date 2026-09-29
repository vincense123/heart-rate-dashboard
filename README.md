# VinCense PPG Waveform Analyzer

Research/engineering tool for inspecting VinCense raw PPG (7,500 samples at 500 Hz) and pairing
PPG features with RMS cuff BP. Not a medical device. No cuffless BP number is produced: no validated
model exists yet.

## Web dashboard (`index.html`)
Open `index.html` in a browser (or host it as a static page). Load a VinCense `.xlsx`; the file is read
locally and never uploaded. Source lives in `web/`; rebuild with `python3 web/build.py`.

## Python / Streamlit version (`streamlit_app/`)
`pip install -r streamlit_app/requirements.txt && streamlit run streamlit_app/app.py`

## Tests
`python -m pytest tests -q` (synthetic signals with known heart rates; set `PPG_SAMPLE_XLSX` to also
compare against the device heart rate in a real workbook).
`legacy_simulated_index.html` is the earlier simulated-data page, kept for reference.

## Note (Phase 0)
The Streamlit code (`streamlit_app/`) now flips the raw IR so the pulse points up, rejects beats below 50% of
the median beat prominence, scores noise before the band-pass filter, and measures rise/fall from pulse feet.
The browser dashboard (`index.html`, `web/ppg_core.js`) has NOT been updated to match yet, so its numbers differ
from Streamlit until it is ported. Project rules and data facts are in `CLAUDE.md`.
