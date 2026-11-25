# Laptop as a Bat — 48-hour implementation plan

Build the judged demo around two continuous-wave (CW) tones. Use fast Doppler sidebands for gestures, slow carrier phase and amplitude for breathing, and left/right echo strength for a signed-speed heatmap. Render all three from the same microphone samples into one frame for the preview and virtual webcam. Reserve frequency-modulated continuous wave (FMCW) ranging for the final stretch. The two performance measures on the challenge card are gesture accuracy and breathing-rate error.

## 1. Host and repository audit

Observed on 2026-11-25, before implementation, on an ASUS TUF Dash F15 FX516PM with an Intel Core i5-11300H.

| Item | Diagnostic and result | Implementation consequence |
| --- | --- | --- |
| OS and interpreter | Windows branding API returned **Windows 11 Pro**. The OS registry reported **26H2, build 26300.9974**. Active interpreter was 64-bit CPython **3.12.10** at `D:\PythonLibs\env\Scripts\python.exe`. | Use a project-local Python 3.12 x64 environment and the Windows WASAPI host API. |
| Python packages | Interpreter package lookup found NumPy 1.26.2, SciPy 1.15.3 and opencv-python 4.13.0.92. All three imported successfully. OpenCV reported the WIN32UI GUI backend. `sounddevice`, `pyaudio` and `pyvirtualcam` were absent. | Install NumPy, SciPy, opencv-python and sounddevice together in the project environment during implementation. Add pyvirtualcam for webcam output. Record the resolved, import-tested versions. |
| Physical audio route | `pnputil /enum-devices /connected /class AudioEndpoint` and the active WASAPI endpoint enumeration both found **Speakers (Realtek(R) Audio)** and **Microphone Array (Realtek(R) Audio)**. These were the console, multimedia and communications defaults. | Select these endpoints by name and host API, resolve their indices at startup, and show the selected route. |
| Other active audio routes | **AI Noise-cancelling Input** and **AI Noise-cancelling Output** (ASUS Utility) also advertised 48 kHz stereo. | Exclude these processed routes from automatic selection. During setup, disable voice enhancement, echo cancellation, AGC, spatial audio and mono downmix on the selected Realtek path where available. |
| Verified audio format | Native WASAPI `GetMixFormat` reported 48,000 Hz, 2 channels, 32 bits on all four endpoints. `IsFormatSupported` returned `S_OK` for 48,000 Hz, 2-channel float32 in shared mode on each. Both Realtek audio clients also returned `S_OK` from `Initialize` with that format while held open together. | **48 kHz stereo capture and render are available at the API.** The clients stayed stopped. Sustained duplex operation, independent microphone signals and the acoustic response at the carriers require the bench tests below. [WASAPI mix formats](https://learn.microsoft.com/windows/win32/api/audioclient/nf-audioclient-iaudioclient-getmixformat) |
| Virtual cameras | The x64 DirectShow video-input enumerator returned `S_FALSE` (empty). The capture category had no entries in either registry view, the OBS virtual-camera CLSID was absent in both, and installed-program checks found no OBS Studio or Unity Capture entry. | No compatible webcam backend was detected. Start in OpenCV preview on this host until a driver is available. Verify export with a second application. |
| Git and supplied material | Project root contained tracked `README.md` and untracked `task/task.png` (challenge card). There was no application code. Branch `master` tracked `origin/master`. Local and queried remote heads were `62ffb5e`. Fetch and push URL was `https://github.com/mihu19/Laptop-as-a-Bat.git`. | Phase 1 adds only this plan. Use the challenge card's two judging measures in the acceptance run. |

## 2. Acquisition, calibration and output limits

| Parameter | Initial decision |
| --- | --- |
| Sample format and routing | One full-duplex float32 `sounddevice.Stream`, 48,000 frames/s, 2 outputs, 2 inputs if usable. Each capture channel has its own analysis path. Accept one physical mic for carrier-based sensing. |
| Transmit signal | Left speaker 18,500 Hz, right speaker 20,500 Hz. Generate each sinusoid from a continuous sample counter. Each carrier lands on an analysis bin and has room for ±400 Hz of Doppler below the 24 kHz Nyquist frequency. |
| Buffering | Request 480 frames per callback (10 ms). If the driver incurs dropouts with that block size, use host-selected blocks and assemble 480-frame hops from a preallocated 2 s ring. Carry ADC/DAC timestamps, sample indices and gap flags. |
| Gesture and display spectrum | Trailing 3,840 real samples, periodic Hann window, `n_fft = 3840`, hop 480, 3,360 samples of overlap (87.5%). Window length 80 ms, update every 10 ms, bin spacing **12.5 Hz**. Carrier bin indices are 1480 and 1640. |
| Work outside the callback | The callback copies input into the ring and fills the entire output buffer from the capped oscillator or zeros. A DSP worker owns overlapping windows, filters and event state. Publish the newest completed snapshot to the UI. Keep allocation, file I/O, FFTs, locks that can block, and rendering off the callback. [sounddevice streams](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html) |

**Output controls.** Start muted. Space arms or mutes, Esc mutes and exits, and window close mutes. Bind transmission to the selected built-in speakers. On a route change, device loss, expired control heartbeat (0.5 s), or stream fault, clear the transmit flag and require manual re-arm. Apply a 20 ms raised-cosine ramp for normal starts and stops. Faults and emergency mute zero the next available output block.

Start each channel at **−48 dBFS peak** (0.004 full scale). Apply a final per-channel peak ceiling of **−30 dBFS** (0.0316) to every output mode, including calibration and chirps. Keep system volume low and fixed during the demo. The audio path enforces a 120 s continuous timeout and at most 120 s of emission in any rolling 180 s, across mutes, re-arms and mode changes. The meter displays post-limiter peak dBFS, on-time, remaining budget, and muted/armed state. This is a **digital output and time meter**. Acoustic SPL depends on the transducers, system gain and listener position. For a venue exposure assessment, measure at listener positions with calibrated equipment covering the emitted bands. Some people can hear the lower carrier; stop on discomfort. [Health Canada guidance on airborne ultrasound measurement](https://www.canada.ca/en/health-canada/services/environmental-workplace-health/reports-publications/radiation/guidelines-safe-use-ultrasound-part-industrial-commercial-applications-safety-code-24.html)

**First hardware session.** Run these steps with the laptop lid and gain fixed. Save the carrier pair, channel map, calibration values and quality flags against that device configuration.

1. Enumerate devices with `python -m sounddevice`. Use `check_input_settings` and `check_output_settings`, then open the pair and run 10 s of muted duplex. Check elapsed samples, timestamps, peak input and underflow/overflow counts. If stereo input fails, try 1 in / 2 out. [sounddevice hardware checks](https://python-sounddevice.readthedocs.io/en/latest/api/checking-hardware.html)
2. With nobody moving in front of the laptop, take 3 s of muted ambient spectrum. After manual arm, play one carrier on one speaker at a time, then both for 5 s. Measure carrier-to-local-floor ratio per mic, clipping and spurious sidebands. Seek ≥15 dB carrier SNR on at least one mic per carrier, and ≥6 dB sideband rise during a repeatable moving-hand pass. A strong direct speaker-to-mic path alone cannot establish echo sensitivity. If a carrier fails, compare limited probes on the 12.5 Hz grid between 18 and 21 kHz, ≥1 kHz apart. Select the pair with the best weaker motion response within the cap.
3. With one carrier at a time, test left-only and right-only speaker routing near each side. Move a reflector in marked left, center and right zones at a fixed distance. Record normalized moving-echo energies and compare the two mic waveforms for identical samples, persistent near-unity correlation and useful differential response. Accept lateral sensing only when repeated left and right trials have a repeatable, correctly ordered contrast. Try speaker-only contrast if the array is duplicated or beamformed. If only one path survives, show signed radial motion, breathing when qualified, and a visible lateral-unavailable state.
4. During the quiet, armed interval estimate a median power spectrum and median absolute deviation (MAD) per mic, carrier and offset bin, plus complex direct-path means for breathing. Recalibrate after a route, gain, lid or position change. Adapt the motion floor slowly (30 s time constant), exclusively in quiet frames, so a gesture cannot train away its own signal.

## 3. Gesture detection

**Features.** For each mic and carrier, subtract its quiet power floor and clamp residual power at zero. Exclude offsets within ±25 Hz of the fitted carrier center to reject the direct tone. Analyze offsets from 37.5 to 400 Hz on each side. A bin is active above both its quiet median + 6 MAD and 6 dB over its median. Sum active power separately above (`E+`) and below (`E−`) the carrier and measure the outermost offset supported by two adjacent active bins. Normalize each path by its response on the center calibration passes before combining mics. Positive shifts correspond to approach and negative shifts to retreat. This use of sideband broadening follows the laptop Doppler method demonstrated in SoundWave. [SoundWave paper](https://ubicomplab.cs.washington.edu/pdfs/soundwave.pdf)

Open an event after 3 consecutive active hops and close it after 8 quiet hops. Compute signed energy dominance, duration, bandwidth on each side, lateral position over time (section 5) and the order of left/right energy peaks. The initial thresholds below are tuning values. Directions are from the seated user's viewpoint.

| Event | Decision rule |
| --- | --- |
| Push | Hand moves toward screen. For 80–600 ms of motion, pooled positive energy is at least 2× negative energy, with positive broadening on both usable carriers and no qualified lateral crossing. |
| Pull | Hand moves away from screen. Apply the reciprocal rule with negative energy and negative broadening. |
| Swipe Left | Within 120–800 ms, lateral score moves from ≥+0.25 to ≤−0.25. Right energy peaks before left by ≥30 ms. Require a calibrated lateral cue and accept mixed Doppler signs during the crossing. |
| Swipe Right | Lateral score crosses the same thresholds from left to right and the left energy peak leads by ≥30 ms. |
| Double Tap | Two short in-air jab-and-retract pulses near center. Each pulse lasts 80–250 ms, has approach then retreat broadening, and is followed by an energy trough. Require 100–400 ms between the first pulse ending and the next starting. Use a two-pulse state machine across event boundaries. |

Hold a tap-like candidate until its pairing window expires, give a completed tap pair priority, then a qualified lateral crossing, then radial dominance. Emit one timestamped event and impose a 300 ms refractory interval after classification. Discard conflicting, weak or overlong events as unknown. Measure latency from gesture completion, including the wait for ambiguous single pulses. Slow chest motion stays on a separate signal path.

## 4. Breathing-rate estimator

Use one seated subject facing the open laptop. Start bench trials at 0.4–0.8 m with a still torso apart from natural breathing. The gesture FFT's 12.5 Hz spacing is too coarse for the breathing cycle. Extract a continuous complex baseband signal directly from the capture samples.

1. For each transmit frequency and mic, mix the input with the complex conjugate of the transmit oscillator. Low-pass and anti-alias in stages (48 kHz → 1 kHz → 20 Hz), retaining the complex signal through 3 Hz. Carry oscillator and filter state across blocks. Estimate transmit/receive clock offset from the strong static carrier during calibration, derotate each path, and track residual drift below 0.03 Hz. Check the direct path for phase stability during operation.
2. Subtract the calibrated static complex return. For each usable path form an unwrapped residual phase signal and a separate amplitude envelope. Near-colocated geometry gives a phase magnitude change of approximately `4π Δz / λ` for a small radial displacement. Multipath and phase nulls can invalidate this relationship. Reject phase samples near the residual noise floor; the amplitude branch can recover a periodic signal on a stronger path.
3. Detrend each candidate and apply a stateful Butterworth bandpass (second-order prototype, SOS implementation) at **0.1–0.6 Hz** on the 20 Hz stream, equivalent to **6–36 breaths/min**. Use a trailing **60 s** window, recomputed each second. Display acquisition progress immediately. Allow a provisional value after ≥30 clean seconds and ≥3 observed cycles. Show a full-confidence value only after the 60 s window and quality checks.
4. Score normalized autocorrelation for periods 1.667–10 s (about 33–200 samples), interpolate around each candidate maximum, and compute `BPM_acf = 60 / period_seconds`. Independently detect waveform peaks with adaptive prominence and at least 1.65 s separation. Use `BPM_peaks = 60 / median(interpeak_seconds)`. Compare full- and half-period candidates to catch amplitude doubling. Require ACF ≥0.5, ≥3 peak intervals and agreement within 2 BPM. Favor the highest-quality phase path; use amplitude only if its fundamental is corroborated by another usable path or the signed phase waveform. Report the median of agreeing estimates and smooth the last 3 accepted updates.
5. Mark motion, clipping, phase slips and stream gaps on the same sample timeline as gestures. Immediately replace the BPM number with **motion**, **weak signal** or **acquiring** as appropriate. Restart the clean window after gross motion or a discontinuity, reset affected filter state, and initially discard 10 s after a reset for filter settling. Verify that settling interval with a step-input replay. A mute clears the displayed estimate. Retain a confidence/coverage counter so withheld readings are visible in validation.

## 5. Two-dimensional Bat Cam

**CW map for the demo.** Horizontal position is a calibrated relative left/right score. Vertical position is signed radial speed. Use 64 horizontal columns and 81 vertical bins spanning −4 to +4 m/s with zero at mid-height and approach at the top. Map each detection's frequency offset using `v ≈ 343 Δf / (2 fc)` m/s. This is a monostatic-equivalent speed; a separated speaker and mic measure the projection along a bistatic path. Label the axes **Left ↔ Right (relative)** and **Away ↔ Toward (approx. m/s)**. CW supplies motion evidence without an absolute distance measurement.

For each velocity bin combine floor-subtracted sideband powers from the corresponding offsets at both carriers. Compute speaker contrast `(P_right − P_left) / (P_right + P_left + ε)` using center-normalized powers. When the mic pair passes calibration, compute the analogous mic contrast at each carrier before pooling. Fit piecewise-linear mappings from the marked left/center/right medians to −1/0/+1. Weight the usable cues by inverse repeat-trial variance, and clamp the combined score to [−1,1]. Use the energy-weighted score across active speeds for swipe detection. Absolute carrier phases wrap over centimeter-scale paths, so the baseline design uses calibrated amplitude for lateral position.

Deposit each speed bin's energy as a horizontal Gaussian lobe, wider when lateral cues disagree, scaled by sideband SNR. Decay old intensity with a 0.35 s time constant, and leave the display dark below the motion floor. If lateral calibration fails, use a full-width row labeled **lateral unavailable**. A single lobe represents dominant motion in a speed bin; two hands moving at the same speed can merge.

**Optional FMCW distance mode.** Only attempt this after the CW acceptance run. Replace CW temporarily with a single-speaker 18–21 kHz linear up-chirp, bandwidth 3 kHz, duration 40 ms (1,920 samples), slope 75,000 Hz/s. Use the same limiter, and acquire a direct-path timing reference on each sweep. Align to that reference, discard the sweep-wrap transient, dechirp complex receive samples against the reference, Hann-window and transform the valid interval. For a stationary reflector the beat offset gives excess path length `343 × Δf_beat / slope`. Add the known direct-path length before dividing by two to display an equivalent range. Calibrate delay bias on a board at a measured distance and cross-check at 0.3, 0.6 and 0.9 m. Ideal full-band range resolution is **5.7 cm** (`343 / (2 × 3000)`); windowing, lost overlap and multipath broaden it. Use a stationary or slowly moving target because Doppler biases single-slope range. Show range only after the board check, with mic-based X only if independently qualified. Returning to CW restarts calibration and breathing acquisition. [Acoustic FMCW on commodity hardware](https://modernmobile.cs.washington.edu/docs/apneaapp.pdf)

## 6. Dashboard and webcam

- Compose one contiguous 1280 × 720, three-channel `uint8` BGR frame on a **30 FPS** monotonic clock. Put the heatmap on the left. Stack the two carrier-centered Doppler spectrogram strips (last 4 s, signed Hz) above the 60 s breathing waveform, BPM and confidence on the right. The footer shows the latest gesture with timestamp, output/time meter, mute control, stream health and camera status. All panels read the same timestamped DSP snapshot.
- Attempt `pyvirtualcam.Camera` with the dimensions above, `fps=30`, `fmt=PixelFormat.BGR` and the Windows `obs` backend. Try `unitycapture` if registered. Send the composed frame unchanged. A separate sender with a single latest-frame slot uses `sleep_until_next_frame()`; slow consumers drop old video frames. Keep OpenCV event handling and mute keys on the main thread. [pyvirtualcam backends](https://github.com/letmaik/pyvirtualcam) · [BGR sender example](https://github.com/letmaik/pyvirtualcam/blob/main/examples/webcam_filter.py)
- If import, open or send fails, close the sender, report the reason and continue the same dashboard in the paced `cv2.imshow` preview. Provide `--preview-only` for a deterministic demo. On this host, the webcam milestone includes installing/registering OBS Studio's bundled Windows Virtual Camera and leaving OBS's own producer stopped. Start LaptopBat, select **OBS Virtual Camera** in a second app, and verify changing timestamps, colors and frame cadence there. Python owns the single OBS camera while transmitting.

## 7. Planned repository layout

Paths below describe the implementation phase. The supplied challenge image is a local input.

~~~text
LaptopBat/
├── PLAN.md
├── README.md                   setup, controls, demo and results
├── task/task.png               supplied challenge card
├── pyproject.toml              Python 3.12 dependencies and test configuration
├── laptopbat/
│   ├── __init__.py
│   ├── __main__.py             python -m laptopbat, options and lifecycle
│   ├── audio.py                WASAPI or WAV replay input, callback, ring and timestamps
│   ├── transmit.py             oscillators, ramps, limiter, timers and mute
│   ├── calibration.py          channel qualification, quiet floors and lateral fit
│   ├── dsp.py                  STFT, complex demodulation and timestamped snapshots
│   ├── gestures.py             features, state machine and debouncing
│   ├── breathing.py            filtering, periodicity, confidence and BPM
│   ├── radar.py                map signed velocity and lateral evidence to heatmap
│   ├── dashboard.py            OpenCV composition, input and preview pacing
│   ├── webcam.py               virtual device selection, frame sender and fallback
│   └── fmcw.py                 optional chirp ranging after core validation
└── tests/
    ├── signals.py              deterministic stereo echoes, noise, drift and dropouts
    ├── test_transmit.py        peak, timeout, budget, mute and device fault cases
    ├── test_sensing.py         gesture, breathing and spatial replay cases
    └── test_output.py          BGR dimensions, timing and missing/busy camera cases
~~~

## 8. Build order and acceptance gates

Targets below are **demo acceptance criteria**. Record measured results and coverage alongside them. Use separate tuning and scored sessions, with ground truth recorded before reviewing predictions.

**Planned Windows entry points.** Once the modules and project metadata exist, create the environment with `python -m venv .venv` and install with `.\.venv\Scripts\python -m pip install -e ".[camera,test]"`. From that environment, `python -m laptopbat` starts muted with camera auto-detection, `python -m laptopbat --preview-only` selects the local window, and `python -m laptopbat --replay trial.wav --preview-only` feeds a 48 kHz capture through the same DSP with playback disabled. Require carrier and channel metadata next to the WAV when capture settings differ from the defaults. Run `python -m pytest` for the replay and fault tests.

| Hours | Build and test sequence | Exit evidence |
| --- | --- | --- |
| 0–4 | Create the project environment. Implement device selection, muted duplex, central output guard and status counters. Start webcam driver setup early. Exercise arming, mute, timeout, re-arm and device changes with a fake clock, then with silent output. | Zero unexpected nonzero samples in mute/fault tests. Peak and rolling on-time stay within limits. Realtek duplex runs 10 s without discontinuities. |
| 4–10 | Run the staged physical calibration. Implement carrier synthesis, FFT and direct-path tracking. Inject positive/negative frequency shifts and move a hand past the left, center and right marks. | Record the chosen carriers and measured SNR, clipping, xrun counts, channel duplication and left/right ordering. A failed gate selects an explicitly labeled reduced mode. |
| 10–18 | Implement gesture segmentation and arbitration. Replay each label, reversed directions, isolated taps, quiet scenes and ambiguous motions. Tune on a pilot session. | One event per intended gesture and no double counting of tap pairs. Freeze thresholds before scoring. |
| 18–28 | Implement breathing IQ and amplitude paths, confidence and motion rejection. Generate known micro-motion at 6, 12, 18, 24 and 36 BPM with leakage, harmonic content, drift and signal loss. Test live against independently timestamped observed breaths or a reference belt. | Clean synthetic windows within 1 BPM at each rate. Motion and lost signal suppress stale values. Log paired reference and estimate over identical clean windows. |
| 28–36 | Integrate radar and dashboard. Test a missing package, missing driver, busy producer and sender failure. Open the webcam from a second application when the backend is available. | Both swipes move in the correct on-screen direction. Push and pull light opposite speed bands. Preview survives each camera failure. External receiver shows a changing 720p dashboard. |
| 36–44 | Score two operators in a fresh session, at least 5 randomized repetitions per gesture per operator (≥50 events). Match predictions one-to-one within 0.5 s after completion, allowing overlap with the annotated event. Include 5 accumulated minutes of armed idle time in budgeted sessions. Collect ≥6 separate 60 s breathing windows across two subjects and several comfortable rates. Timestamp/count actual cycles as reference, even when pacing is used. | Target macro recall ≥85%, ≤1 idle false trigger/min and median post-completion gesture latency ≤300 ms. Publish confusion matrix including misses/unknowns and extra triggers. Target live breathing mean absolute error ≤2 BPM, reporting per-window error, worst error and valid-output coverage ≥80% of eligible post-warm-up seconds. Include motion-recovery time separately. |
| 44–48 | Rehearse a 10 min mixed preview/webcam run including muted intervals and enforced cool-downs. Check clip/overrun counters, actual sender FPS and receiver continuity. Spend spare time on FMCW only after core gates pass. | Aim for 29–31 average FPS, zero audio overruns/underruns, and successful mute and preview fallback. Record any missed gate in the demo notes. |

For each scored breathing window, derive reference BPM from independent breath timestamps as `60 × (count − 1) / (last_time − first_time)` with time in seconds. Compute the mean of the absolute per-window BPM errors. Count a missing estimate against coverage, and record how many of the six windows produced a result.

**Demo sequence.** Open the receiver or preview, show the muted state, arm and perform quiet calibration. Demonstrate the five gestures and the labeled velocity map. Let the subject settle, show breathing acquisition and compare a 60 s estimate with the reference. Mute before the session limit. If venue acoustics defeat a live channel, show its quality state and use a clearly labeled, transmit-muted replay for that portion.
