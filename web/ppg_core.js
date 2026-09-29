/*
 * VinCense PPG analysis core (browser + Node).
 * Port of streamlit_app/ppg_analysis.py and ppg_features.py.
 * Research/engineering use only.
 */
(function (root) {
  "use strict";

  const SAMPLE_RATE = 500;
  const LOW_CUTOFF = 0.5;
  const HIGH_CUTOFF = 8.0;
  const FILTER_ORDER = 4;
  const MIN_HEART_RATE = 40;
  const MAX_HEART_RATE = 180;
  const MIN_PEAK_DISTANCE = Math.floor((SAMPLE_RATE * 60) / MAX_HEART_RATE); // 166

  // 4th-order Butterworth band-pass 0.5-8 Hz at 500 Hz, as second-order
  // sections (scipy.signal.butter(..., output="sos")) and the matching
  // steady-state values (scipy.signal.sosfilt_zi).
  const SOS = [
    [4.372688797820071e-06, 8.745377595640141e-06, 4.372688797820071e-06, 1.0, -1.843281077034628, 0.850729784704957],
    [1.0, 2.0, 1.0, 1.0, -1.9250646864028191, 0.9345460910125841],
    [1.0, -2.0, 1.0, 1.0, -1.987361748985374, 0.987410987372232],
    [1.0, -2.0, 1.0, 1.0, -1.9956047086893063, 0.995645292613521],
  ];
  const ZI = [
    [0.0023437870142540692, -0.001993276709832369],
    [0.9882897466820506, -0.9234486233180359],
    [-0.9906379063833222, 0.9906379063833446],
    [0.0, 0.0],
  ];

  // ---------------------------------------------------------------- helpers

  function sortedCopy(a) {
    const c = Float64Array.from(a);
    c.sort();
    return c;
  }

  function median(a) {
    if (!a.length) return NaN;
    const s = sortedCopy(a);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function mean(a) {
    if (!a.length) return NaN;
    let t = 0;
    for (let i = 0; i < a.length; i++) t += a[i];
    return t / a.length;
  }

  function std(a, ddof) {
    ddof = ddof || 0;
    if (a.length - ddof <= 0) return NaN;
    const m = mean(a);
    let t = 0;
    for (let i = 0; i < a.length; i++) t += (a[i] - m) * (a[i] - m);
    return Math.sqrt(t / (a.length - ddof));
  }

  // numpy.percentile, linear interpolation
  function percentile(a, q) {
    const s = sortedCopy(a);
    const idx = ((s.length - 1) * q) / 100;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  }

  function finite(a) {
    const out = [];
    for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) out.push(a[i]);
    return out;
  }

  // ----------------------------------------------------------------- filter

  function sosfilt(x, scale) {
    const y = Float64Array.from(x);
    for (let s = 0; s < SOS.length; s++) {
      const b0 = SOS[s][0], b1 = SOS[s][1], b2 = SOS[s][2];
      const a1 = SOS[s][4], a2 = SOS[s][5];
      let z0 = ZI[s][0] * scale, z1 = ZI[s][1] * scale;
      for (let n = 0; n < y.length; n++) {
        const xn = y[n];
        const yn = b0 * xn + z0;
        z0 = b1 * xn - a1 * yn + z1;
        z1 = b2 * xn - a2 * yn;
        y[n] = yn;
      }
    }
    return y;
  }

  // scipy.signal.sosfiltfilt (odd padding, steady-state initial conditions)
  function bandpassFilter(signal) {
    const x = Float64Array.from(signal);
    const n = x.length;
    if (n < 20) return x;
    const padlen = 3 * (2 * SOS.length + 1);
    if (n <= padlen) return x;

    const ext = new Float64Array(n + 2 * padlen);
    for (let i = 0; i < padlen; i++) ext[i] = 2 * x[0] - x[padlen - i];
    ext.set(x, padlen);
    for (let i = 0; i < padlen; i++) ext[padlen + n + i] = 2 * x[n - 1] - x[n - 2 - i];

    let y = sosfilt(ext, ext[0]);
    y.reverse();
    y = sosfilt(y, y[0]);
    y.reverse();
    return y.slice(padlen, padlen + n);
  }

  // centred rolling median, min_periods = 1 (pandas), window up to 201
  function removeDC(signal) {
    const x = Float64Array.from(signal);
    const n = x.length;
    const window = Math.min(201, n % 2 ? n : n - 1);
    const baseline = new Float64Array(n);

    if (window >= 3) {
      // Sliding sorted window: O(n * window) memmoves instead of n sorts.
      const half = (window - 1) >> 1;
      const buf = new Float64Array(window);
      let size = 0;
      const insert = (v) => {
        let lo = 0, hi = size;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (buf[mid] < v) lo = mid + 1; else hi = mid; }
        buf.copyWithin(lo + 1, lo, size);
        buf[lo] = v;
        size++;
      };
      const remove = (v) => {
        let lo = 0, hi = size;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (buf[mid] < v) lo = mid + 1; else hi = mid; }
        buf.copyWithin(lo, lo + 1, size);
        size--;
      };
      for (let k = 0; k <= Math.min(half, n - 1); k++) insert(x[k]);
      for (let i = 0; i < n; i++) {
        baseline[i] = size % 2 ? buf[size >> 1] : (buf[(size >> 1) - 1] + buf[size >> 1]) / 2;
        if (i - half >= 0) remove(x[i - half]);
        if (i + half + 1 < n) insert(x[i + half + 1]);
      }
    } else {
      baseline.fill(median(x));
    }

    const dc = new Float64Array(n);
    for (let i = 0; i < n; i++) dc[i] = x[i] - baseline[i];
    return { dcRemoved: dc, baseline };
  }

  function normalizeSignal(signal) {
    const n = signal.length;
    const med = median(signal);
    const dev = new Float64Array(n);
    for (let i = 0; i < n; i++) dev[i] = Math.abs(signal[i] - med);
    const mad = median(dev);
    const out = new Float64Array(n);
    if (mad < 1e-9) {
      const s = std(signal, 0);
      if (s < 1e-9) return out;
      for (let i = 0; i < n; i++) out[i] = (signal[i] - med) / s;
      return out;
    }
    for (let i = 0; i < n; i++) out[i] = (signal[i] - med) / (1.4826 * mad);
    return out;
  }

  // ------------------------------------------------------------- find_peaks

  function localMaxima(x) {
    const peaks = [];
    let i = 1;
    const imax = x.length - 1;
    while (i < imax) {
      if (x[i - 1] < x[i]) {
        let ahead = i + 1;
        while (ahead < imax && x[ahead] === x[i]) ahead++;
        if (x[ahead] < x[i]) {
          peaks.push((i + ahead - 1) >> 1);
          i = ahead;
        } else i++;
      } else i++;
    }
    return peaks;
  }

  function prominenceOf(x, peak) {
    let leftMin = x[peak];
    for (let i = peak; i >= 0 && x[i] <= x[peak]; i--) if (x[i] < leftMin) leftMin = x[i];
    let rightMin = x[peak];
    for (let i = peak; i < x.length && x[i] <= x[peak]; i++) if (x[i] < rightMin) rightMin = x[i];
    return x[peak] - Math.max(leftMin, rightMin);
  }

  // scipy.signal.find_peaks with distance and prominence (same order:
  // distance first, then prominence)
  function findPeaks(x, opts) {
    opts = opts || {};
    let peaks = localMaxima(x);

    if (opts.distance != null && peaks.length > 1) {
      const dist = Math.ceil(opts.distance);
      const keep = new Uint8Array(peaks.length).fill(1);
      const order = peaks.map((_, k) => k).sort((a, b) => x[peaks[a]] - x[peaks[b]]);
      for (let o = order.length - 1; o >= 0; o--) {
        const j = order[o];
        if (!keep[j]) continue;
        let k = j - 1;
        while (k >= 0 && peaks[j] - peaks[k] < dist) { keep[k] = 0; k--; }
        k = j + 1;
        while (k < peaks.length && peaks[k] - peaks[j] < dist) { keep[k] = 0; k++; }
      }
      peaks = peaks.filter((_, k) => keep[k]);
    }

    let prominences = peaks.map((p) => prominenceOf(x, p));
    if (opts.prominence != null) {
      const sel = prominences.map((v) => v >= opts.prominence);
      peaks = peaks.filter((_, k) => sel[k]);
      prominences = prominences.filter((_, k) => sel[k]);
    }
    return { peaks, prominences };
  }

  // -------------------------------------------------------- pulse detection

  function estimatePulsePeriod(signal, fs) {
    fs = fs || SAMPLE_RATE;
    let x = Float64Array.from(signal);
    const edge = Math.floor(fs * 0.5);
    if (x.length > 4 * edge) x = x.slice(edge, x.length - edge);

    const low = percentile(x, 2), high = percentile(x, 98);
    for (let i = 0; i < x.length; i++) x[i] = Math.min(Math.max(x[i], low), high);
    const m = mean(x);
    for (let i = 0; i < x.length; i++) x[i] -= m;
    const n = x.length;

    const lagMin = Math.floor((fs * 60) / (MAX_HEART_RATE + 20));
    const lagMax = Math.min(Math.floor((fs * 60) / (MIN_HEART_RATE - 5)), n >> 1);
    if (n < 4 * lagMin || lagMax <= lagMin + 2) return null;

    let acf0 = 0;
    for (let i = 0; i < n; i++) acf0 += x[i] * x[i];
    if (acf0 <= 0) return null;

    const win = new Float64Array(lagMax - lagMin + 1);
    for (let lag = lagMin; lag <= lagMax; lag++) {
      let t = 0;
      for (let i = 0; i + lag < n; i++) t += x[i] * x[i + lag];
      win[lag - lagMin] = (t / acf0) * (n / (n - lag));
    }

    const cand = localMaxima(win);
    if (!cand.length) return null;
    let strongest = -Infinity;
    for (const c of cand) if (win[c] > strongest) strongest = win[c];
    if (strongest < 0.25) return null;

    let chosen = cand[0];
    for (const c of cand) {
      if (win[c] >= 0.85 * strongest) { chosen = c; break; }
    }
    return (chosen + lagMin) / fs;
  }

  function detectPulses(signal, fs) {
    fs = fs || SAMPLE_RATE;
    const normalized = normalizeSignal(signal);
    if (normalized.length < 100) return [];

    const amplitude = percentile(normalized, 90) - percentile(normalized, 10);
    if (amplitude < 0.05) return [];
    const prominence = Math.max(0.15, amplitude * 0.10);

    let res = findPeaks(normalized, { distance: MIN_PEAK_DISTANCE, prominence });
    let peaks = res.peaks;

    // Second pass: keep one peak per real pulse period and drop the small
    // secondary (dicrotic / diastolic) bumps that inflate the rate.
    const period = estimatePulsePeriod(signal, fs);
    if (period !== null && peaks.length >= 3) {
      const distance = Math.max(Math.floor(0.2 * fs), Math.floor(0.6 * period * fs));
      res = findPeaks(normalized, { distance, prominence });
      peaks = res.peaks;
      if (peaks.length >= 3) {
        const typical = median(res.prominences);
        peaks = peaks.filter((_, k) => res.prominences[k] >= 0.4 * typical);
      }
    }

    const edge = Math.floor(fs * 0.5);
    return peaks.filter((p) => p >= edge && p < signal.length - edge);
  }

  // ---------------------------------------------------------------- quality

  function calculateQuality(signal, peaks, fs) {
    fs = fs || SAMPLE_RATE;
    const result = {
      score: 0, classification: "Poor", peak_count: peaks.length,
      interval_score: 0, amplitude_score: 0, noise_score: 0, coverage_score: 0,
      heart_rate: null, median_interval: null, interval_cv: null,
    };
    if (signal.length < 100) return result;

    const duration = signal.length / fs;
    const expectedMin = (duration * MIN_HEART_RATE) / 60;
    const expectedMax = (duration * MAX_HEART_RATE) / 60;

    let coverage = 0;
    if (peaks.length >= 3) {
      coverage = Math.min(100, (peaks.length / Math.max(expectedMin, 1)) * 100);
      if (peaks.length > expectedMax * 1.25) coverage *= 0.5;
    }

    let intervalScore = 0;
    if (peaks.length >= 3) {
      const iv = [];
      for (let i = 1; i < peaks.length; i++) iv.push((peaks[i] - peaks[i - 1]) / fs);
      const medIv = median(iv);
      const cv = medIv > 0 ? std(iv, 0) / medIv : 1;
      const hr = medIv > 0 ? 60 / medIv : null;
      result.median_interval = medIv;
      result.interval_cv = cv;
      result.heart_rate = hr;

      intervalScore = cv <= 0.03 ? 100 : cv <= 0.05 ? 90 : cv <= 0.08 ? 75 : cv <= 0.12 ? 55 : cv <= 0.18 ? 30 : 10;
      if (hr !== null && !(hr >= MIN_HEART_RATE && hr <= MAX_HEART_RATE)) intervalScore *= 0.4;
    }

    let amplitudeScore = 0;
    if (peaks.length >= 3) {
      const amps = [];
      for (let i = 0; i < peaks.length - 1; i++) {
        const seg = signal.subarray ? signal.subarray(peaks[i], peaks[i + 1]) : signal.slice(peaks[i], peaks[i + 1]);
        if (seg.length > 5) {
          let mx = -Infinity, mn = Infinity;
          for (let k = 0; k < seg.length; k++) { if (seg[k] > mx) mx = seg[k]; if (seg[k] < mn) mn = seg[k]; }
          amps.push(mx - mn);
        }
      }
      if (amps.length >= 2) {
        const medAmp = median(amps);
        const cv = medAmp > 0 ? std(amps, 0) / medAmp : 1;
        amplitudeScore = cv <= 0.10 ? 100 : cv <= 0.20 ? 85 : cv <= 0.30 ? 70 : cv <= 0.45 ? 50 : cv <= 0.60 ? 30 : 10;
      }
    }

    let noiseScore = 0;
    if (signal.length >= 11) {
      const n = signal.length;
      const residual = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const lo = Math.max(0, i - 5), hi = Math.min(n - 1, i + 5);
        let t = 0;
        for (let k = lo; k <= hi; k++) t += signal[k];
        residual[i] = signal[i] - t / (hi - lo + 1);
      }
      const sStd = std(signal, 0);
      const ratio = sStd > 1e-9 ? std(residual, 0) / sStd : 1;
      noiseScore = ratio <= 0.03 ? 100 : ratio <= 0.05 ? 90 : ratio <= 0.08 ? 75 : ratio <= 0.12 ? 55 : ratio <= 0.20 ? 30 : 10;
    }

    const score = Math.min(100, Math.max(0,
      intervalScore * 0.35 + amplitudeScore * 0.25 + noiseScore * 0.25 + coverage * 0.15));
    result.score = score;
    result.classification = score >= 80 ? "Good" : score >= 60 ? "Acceptable" : "Poor";
    result.interval_score = intervalScore;
    result.amplitude_score = amplitudeScore;
    result.noise_score = noiseScore;
    result.coverage_score = Math.min(coverage, 100);
    return result;
  }

  // ----------------------------------------------------------- reading input

  const CYCLE_COLUMNS = [
    "PPGValue cycle 1 (0-1500)",
    "PPGValue cycle 2 (1501-3000)",
    "PPGValue cycle 3 (3001-4500)",
    "PPGValue cycle 4 (4501-6000)",
    "PPGValue cycle 5 (6001-7500)",
  ];

  function extractPPG(reading) {
    if (reading && reading._ppg) return Float64Array.from(reading._ppg);
    const out = [];
    for (const col of CYCLE_COLUMNS) {
      const cell = reading[col];
      if (cell === undefined || cell === null || cell === "") continue;
      const parts = String(cell).split(",");
      for (const p of parts) {
        const v = parseFloat(p.trim());
        if (Number.isFinite(v) && p.trim() !== "") out.push(v);
      }
    }
    return Float64Array.from(out.slice(0, 7500));
  }

  function analyzeSamples(raw) {
    if (!raw.length) return null;
    const { dcRemoved, baseline } = removeDC(raw);
    const filtered = bandpassFilter(dcRemoved);
    const peaks = detectPulses(filtered);
    const quality = calculateQuality(filtered, peaks);
    return { raw, baseline, dcRemoved, filtered, peaks, quality };
  }

  function analyzeReading(reading) {
    return analyzeSamples(extractPPG(reading));
  }

  // ---------------------------------------------------------------- features

  function cv(a) {
    const x = finite(a);
    if (x.length < 2 || Math.abs(mean(x)) < 1e-12) return NaN;
    return (std(x, 1) / Math.abs(mean(x))) * 100;
  }
  const smean = (a) => { const x = finite(a); return x.length ? mean(x) : NaN; };
  const sstd = (a) => { const x = finite(a); return x.length > 1 ? std(x, 1) : NaN; };

  function timingFeatures(peaks, fs) {
    fs = fs || SAMPLE_RATE;
    const r = {
      "Detected Pulses": peaks.length, "Mean IBI (s)": NaN, "IBI SD (s)": NaN, "IBI CV (%)": NaN,
      "Mean HR (BPM)": NaN, "Median HR (BPM)": NaN, "HR SD (BPM)": NaN, "HR CV (%)": NaN,
    };
    if (peaks.length < 2) return r;
    const ibi = [];
    for (let i = 1; i < peaks.length; i++) {
      const d = (peaks[i] - peaks[i - 1]) / fs;
      if (d > 0) ibi.push(d);
    }
    if (!ibi.length) return r;
    const hr = ibi.map((d) => 60 / d).filter((h) => h >= 30 && h <= 220);
    r["Mean IBI (s)"] = smean(ibi);
    r["IBI SD (s)"] = sstd(ibi);
    r["IBI CV (%)"] = cv(ibi);
    r["Mean HR (BPM)"] = smean(hr);
    r["Median HR (BPM)"] = hr.length ? median(hr) : NaN;
    r["HR SD (BPM)"] = sstd(hr);
    r["HR CV (%)"] = cv(hr);
    return r;
  }

  function pulseFeatures(filtered, peaksIn, fs) {
    fs = fs || SAMPLE_RATE;
    const sig = filtered;
    const peaks = Array.from(new Set(peaksIn.filter((p) => p >= 0 && p < sig.length))).sort((a, b) => a - b);
    const rows = [];
    for (let i = 0; i < peaks.length; i++) {
      const peak = peaks[i];
      const start = i === 0 ? 0 : Math.floor((peaks[i - 1] + peak) / 2);
      const end = i === peaks.length - 1 ? sig.length - 1 : Math.floor((peak + peaks[i + 1]) / 2);
      if (end <= start) continue;

      let baseline = Infinity;
      for (let k = start; k <= end; k++) if (sig[k] < baseline) baseline = sig[k];
      const amplitude = sig[peak] - baseline;
      const rise = Math.max(1, peak - start) / fs;
      const fall = Math.max(1, end - peak) / fs;
      const width = (end - start) / fs;

      let area = 0;
      let prev = Math.max(sig[start] - baseline, 0);
      for (let k = start + 1; k <= end; k++) {
        const cur = Math.max(sig[k] - baseline, 0);
        area += ((prev + cur) / 2) / fs;
        prev = cur;
      }
      rows.push({
        "Pulse": i + 1, "Start Sample": start, "Peak Sample": peak, "End Sample": end,
        "Amplitude": amplitude, "Pulse Width (s)": width, "Rise Time (s)": rise, "Fall Time (s)": fall,
        "Rise Slope": amplitude / rise, "Fall Slope": amplitude / fall, "Pulse Area": area,
      });
    }
    return rows;
  }

  function summarizePulses(rows) {
    const col = (k) => rows.map((r) => r[k]);
    if (!rows.length) {
      return {
        "Pulses Analysed": 0, "Mean Amplitude": NaN, "Amplitude SD": NaN, "Amplitude CV (%)": NaN,
        "Mean Pulse Width (s)": NaN, "Pulse Width SD (s)": NaN, "Mean Rise Time (s)": NaN,
        "Mean Fall Time (s)": NaN, "Mean Rise Slope": NaN, "Mean Fall Slope": NaN,
        "Mean Pulse Area": NaN, "Pulse Area CV (%)": NaN,
      };
    }
    return {
      "Pulses Analysed": rows.length,
      "Mean Amplitude": smean(col("Amplitude")), "Amplitude SD": sstd(col("Amplitude")),
      "Amplitude CV (%)": cv(col("Amplitude")),
      "Mean Pulse Width (s)": smean(col("Pulse Width (s)")), "Pulse Width SD (s)": sstd(col("Pulse Width (s)")),
      "Mean Rise Time (s)": smean(col("Rise Time (s)")), "Mean Fall Time (s)": smean(col("Fall Time (s)")),
      "Mean Rise Slope": smean(col("Rise Slope")), "Mean Fall Slope": smean(col("Fall Slope")),
      "Mean Pulse Area": smean(col("Pulse Area")), "Pulse Area CV (%)": cv(col("Pulse Area")),
    };
  }

  function waveformFeatures(filtered, fs) {
    fs = fs || SAMPLE_RATE;
    const x = finite(filtered);
    if (!x.length) return {};
    let mn = Infinity, mx = -Infinity, sq = 0;
    for (const v of x) { if (v < mn) mn = v; if (v > mx) mx = v; sq += v * v; }
    return {
      "Samples": x.length, "Duration (s)": x.length / fs, "Mean": mean(x), "Std Dev": std(x, 0),
      "RMS": Math.sqrt(sq / x.length), "Peak-to-Peak": mx - mn, "Minimum": mn, "Maximum": mx,
    };
  }

  function extractAllFeatures(filtered, peaks, fs) {
    const pulses = pulseFeatures(filtered, peaks, fs);
    const summary = Object.assign({}, waveformFeatures(filtered, fs), timingFeatures(peaks, fs), summarizePulses(pulses));
    return { summary, pulses };
  }

  // -------------------------------------------------------------- BP dataset

  const FEATURE_KEYS = [
    "Detected Pulses", "Mean IBI (s)", "IBI SD (s)", "IBI CV (%)", "Mean HR (BPM)", "Median HR (BPM)",
    "HR SD (BPM)", "HR CV (%)", "Mean Amplitude", "Amplitude SD", "Amplitude CV (%)",
    "Mean Pulse Width (s)", "Pulse Width SD (s)", "Mean Rise Time (s)", "Mean Fall Time (s)",
    "Mean Rise Slope", "Mean Fall Slope", "Mean Pulse Area", "Pulse Area CV (%)", "Samples",
    "Duration (s)", "Mean", "Std Dev", "RMS", "Peak-to-Peak", "Minimum", "Maximum",
  ];

  function parseBP(value) {
    if (value === undefined || value === null) return [NaN, NaN];
    const m = /(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)/.exec(String(value));
    return m ? [parseFloat(m[1]), parseFloat(m[2])] : [NaN, NaN];
  }

  function pearson(xs, ys) {
    const n = xs.length;
    if (n < 3) return NaN;
    const mx = mean(xs), my = mean(ys);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) {
      sxy += (xs[i] - mx) * (ys[i] - my);
      sxx += (xs[i] - mx) ** 2;
      syy += (ys[i] - my) ** 2;
    }
    if (sxx < 1e-24 || syy < 1e-24) return NaN;
    return sxy / Math.sqrt(sxx * syy);
  }

  function correlationTable(rows, featureNames) {
    const out = [];
    for (const f of featureNames) {
      for (const target of ["RMS SBP", "RMS DBP"]) {
        const xs = [], ys = [];
        for (const r of rows) {
          if (Number.isFinite(r[f]) && Number.isFinite(r[target])) { xs.push(r[f]); ys.push(r[target]); }
        }
        const rr = pearson(xs, ys);
        out.push({ Feature: f, Target: target, N: xs.length, "Pearson r": rr, "Absolute |r|": Math.abs(rr) });
      }
    }
    return out.sort((a, b) =>
      a.Target === b.Target
        ? (Number.isNaN(b["Absolute |r|"]) ? -1 : 0) - (Number.isNaN(a["Absolute |r|"]) ? -1 : 0) || b["Absolute |r|"] - a["Absolute |r|"]
        : a.Target < b.Target ? -1 : 1);
  }

  // -------------------------------------------------------------- synthetic

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeSynthetic(hr, kind, opts) {
    opts = opts || {};
    const noise = opts.noise != null ? opts.noise : 0.02;
    const wander = opts.wander != null ? opts.wander : 0.15;
    const hrv = opts.hrv != null ? opts.hrv : 0.02;
    const dc = opts.dc != null ? opts.dc : 300000;
    const scale = opts.scale != null ? opts.scale : 2000;
    const rand = mulberry32((opts.seed || 0) + 1);
    const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
    const n = 7500, seconds = n / SAMPLE_RATE;
    const sig = new Float64Array(n);

    const g = (t, c, w) => Math.exp(-0.5 * ((t - c) / w) ** 2);
    const template = {
      notch: (t) => g(t, 0.18, 0.075) + 0.45 * g(t, 0.47, 0.10),
      strong_notch: (t) => g(t, 0.16, 0.07) + 0.70 * g(t, 0.46, 0.09),
      sawtooth: (t) => {
        const rise = Math.pow(Math.min(Math.max(t / 0.62, 0), 1), 1.3);
        const fall = Math.min(Math.max((t - 0.62) / 0.14, 0), 1);
        return rise * (1 - fall) + 0.10 * g(t, 0.86, 0.05);
      },
    }[kind || "notch"];

    const period = 60 / hr;
    const beats = [];
    let tb = rand() * period;
    while (tb < seconds + period) { beats.push(tb); tb += period * (1 + hrv * gauss()); }

    for (let i = 0; i < beats.length; i++) {
      const len = i + 1 < beats.length ? beats[i + 1] - beats[i] : period;
      const m = Math.max(8, Math.round(len * SAMPLE_RATE));
      const first = Math.round(beats[i] * SAMPLE_RATE);
      const amp = 1 + 0.05 * gauss();
      for (let k = 0; k < m; k++) {
        const idx = first + k;
        if (idx >= 0 && idx < n) sig[idx] += amp * template(k / m);
      }
    }

    const p1 = rand() * 6.28, p2 = rand() * 6.28;
    const samples = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / SAMPLE_RATE;
      const v = sig[i] + wander * Math.sin(2 * Math.PI * 0.25 * t + p1) +
        0.5 * wander * Math.sin(2 * Math.PI * 0.07 * t + p2) + noise * gauss();
      samples[i] = Math.round(dc + scale * v);
    }
    const inside = beats.filter((b) => b >= 0 && b < seconds);
    let trueHr = hr;
    if (inside.length > 2) trueHr = 60 / ((inside[inside.length - 1] - inside[0]) / (inside.length - 1));
    return { samples, trueHr };
  }

  const api = {
    SAMPLE_RATE, LOW_CUTOFF, HIGH_CUTOFF, FILTER_ORDER, MIN_HEART_RATE, MAX_HEART_RATE, MIN_PEAK_DISTANCE,
    CYCLE_COLUMNS, FEATURE_KEYS,
    bandpassFilter, removeDC, normalizeSignal, findPeaks, estimatePulsePeriod, detectPulses,
    calculateQuality, extractPPG, analyzeSamples, analyzeReading, extractAllFeatures,
    timingFeatures, pulseFeatures, summarizePulses, waveformFeatures,
    parseBP, pearson, correlationTable, makeSynthetic, median, mean, std,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PPG = api;
})(typeof window !== "undefined" ? window : globalThis);
