# Laptop-as-a-Bat

Bat Cam uses a Windows laptop's speakers and microphone array as a near-field sonar. The 1280 × 720 preview and OBS Virtual Camera share one dashboard. The default view shows a live echo profile, range history and an estimated top-down footprint when several sound paths agree.

## Run

In PowerShell from the project root, use 64-bit Python 3.12. Skip the first two commands if the project environment already exists.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
~~~

Keep the area in front of the laptop clear at launch. After three seconds Bat Cam starts the audible 6–13.5 kHz probe and measures the empty room. When the dashboard asks you to move an object into range, place it about 30–75 cm away near laptop height. The probe uses the existing −48 dBFS default peak gain per channel and stops at its configured output limits. Use the following command for a silent preview.

~~~powershell
.\.venv\Scripts\python.exe main.py --simulate --no-virtual-cam
~~~

The simulator uses generated microphone samples and an acoustic room model. It takes no input from the microphone and leaves the physical speakers silent. Enter `--muted` to open the live microphone and preview with the speakers muted.

### Spatial echo imaging

~~~powershell
.\.venv\Scripts\python.exe main.py --spatial
~~~

`--spatial` opens the same viewer with manual arming. Clear the area, press **M**, and wait for room calibration. Recalibrate after moving the laptop or changing the lid angle. Add `--simulate` to try the dashboard with moving reflectors and no speaker output.

Each speaker uses different subcarriers so the app can recover four speaker-to-microphone paths from two capture channels. It subtracts the empty-room profile and plots locations where at least two calibrated paths agree near laptop height. The footprint shows reflecting patches. Its width and shape depend on echo strength, room acoustics and the estimated laptop geometry. The echo profile and range history remain available even when the footprint is blank, including on mono input.

### Learn left, center and right

~~~powershell
.\.venv\Scripts\python.exe main.py --learn-zones
~~~

Clear the area and press **M** to start the audible spatial probe. After the room reference, the dashboard guides you through 12 short trials. Press **C** on each prompt. You have two seconds to step clear for an **Empty** trial or move into the indicated zone. Face the laptop from about 35–75 cm away. Gently shift in place until the next prompt appears. Keep the laptop, lid angle and volume fixed during all trials.

The model learns from room-subtracted echo windows, including weak reflections below the spatial projection threshold. The last four trials are held out before training. A profile is saved only when at least 75% of those checks are correct, each person zone scores at least 60%, and clear-area checks score at least 80%. Bat Cam saves model parameters locally and mutes when calibration ends. Run `--spatial` with the same audio route, gain and room setup to use the profile. Zone labels are estimates of coarse position. This is a check in one room and session. [EcholocationDNN research code](https://github.com/OptMLGroup/EcholocationDNN)

### OFDM motion tracking

~~~powershell
.\.venv\Scripts\python.exe main.py --ofdm
~~~

Clear the area in front of the laptop and press **M** to start the probe. Stand clear until the empty-room measurement finishes, then move toward or away from the laptop in the 25–150 cm range. The measured echoes enter the trail at the right edge and move vertically as the distance changes. The dashboard reports a tracked range after several echo readings agree. Press **M** to mute, then press it again with the area clear to recalibrate. Select **OBS Virtual Camera** in the receiving app.

The OFDM tracker adapts the 6–10 kHz sounder and channel estimation from [Laptop Acoustic Radar](https://github.com/enricguaschmesia-lab/laptop-acoustic-radar/tree/b00695c5472ce705532aaa1cb02bf3a87d2cceba) and applies Bat Cam's existing transmission caps. The probe is audible. Add `--simulate` for a silent demonstration. Use `--chirp` for the gesture and breathing readouts.

### Windows audio and OBS

On the audited ASUS laptop, Bat Cam automatically selects the Realtek microphone array and speakers. Use `--list-devices` to identify WASAPI IDs on other laptops, then pass `--input-device ID --output-device ID` at launch. Windows needs microphone access for desktop apps and a two-channel 48 kHz output format. An input endpoint can fall back to mono. Bat Cam requests WASAPI RAW. If the device applies OEM audio processing, disable noise cancellation, beamforming and spatial effects on the selected route. [Windows audio processing modes](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/audio-signal-processing-modes)

Install the OBS Studio Virtual Camera component. Leave OBS's own virtual-camera producer stopped and select **OBS Virtual Camera** in the receiving app. If the driver is missing or busy, Bat Cam continues in the OpenCV preview. [OBS Studio](https://obsproject.com/download)

## Controls

| Key | Action |
| --- | --- |
| **M** or **Space** | Mute or re-arm transmission within the exposure budget. |
| **R** | Reset the demo timer and resume when exposure budget permits. |
| **Esc** or **Q** | Mute and close the app. |

Chirp mode needs no calibration keys. Gesture detection uses tracks from range-gated echoes. Push approaches the laptop, Pull moves away, and Swipe Left/Right changes relative lateral position.

## What the camera measures

- Each speaker takes a turn transmitting a 512-sample, Tukey-windowed chirp over 12.0–16.5 kHz. Its 13.0 or 15.5 kHz pilot fits in the slot after the chirp. Audio runs at 48 kHz. The model updates from 25 acoustic packets per second, independently of the webcam's 30 FPS frame clock.
- Stereo matched filtering separates the two chirps. Direct-path estimates and rolling band gains normalize the paths automatically. The motion detector adapts its room model at alpha 0.02, so a stationary return can fade from its motion estimate.
- The processor admits taps estimated at 0.15–0.95 m to image, gesture and breathing processing. The delay samples are spaced 3.57 mm apart. With 4.5 kHz of bandwidth, ideal reflector separation is about 3.8 cm. Device geometry, noise and multipath can shift estimated echo ranges. Use `--direct-path-m` to supply a measured speaker-to-mic distance. [Pulse compression and range resolution](https://www.ll.mit.edu/sites/default/files/outreach/doc/2018-07/lecture%205.pdf)
- The echo trail shows measured strength by distance over time. The profile shows the currently accepted range bins. The strongest echo in the chirp profile can come from a stationary object in the room. The plots clear when the probe is muted or the timing reference drops.
- Breathing uses a stable high-SNR return in the 0.35–0.85 m zone. It combines complex phase and amplitude with a 0.12–0.55 Hz bandpass and autocorrelation. The estimator primarily uses a 10-second window and retains 20 seconds for slower cycles. The dashboard displays BPM when the estimate qualifies.

## Acoustic model

~~~powershell
.\.venv\Scripts\python.exe scripts/train_acoustic_ml.py
~~~

The first launch also trains automatically if the generated model is absent. The trainer fits a small MLP and a Ridge position model with scikit-learn, then saves the local `models/acoustic_sonar_ml.joblib` artifact. The training set contains simulated range profiles, multipath, signed Doppler estimates, stereo asymmetry and noise. Validation uses independent simulator seeds. The three-second real microphone sample supplies the runtime noise baseline without assigning human labels. Signed Doppler features are estimates from motion of gated range taps.

In live chirp mode, startup and per-second terminal messages show the opened input channels, peak microphone level, gated echo SNR when available, and inference time.

## Limits and diagnostics

Peak amplitude is capped at approximately −30 dBFS per channel. The default rolling limit is 120 seconds of emission in 180 seconds. The separate demo timer defaults to 300 seconds. A reset preserves the rolling limit. `--auto-cooldown` waits at least two seconds and until enough budget is available to resume. These values describe digital output. Sound-pressure level depends on the laptop, system volume and listener position. [Airborne ultrasound measurement guidance](https://www.canada.ca/en/health-canada/services/environmental-workplace-health/reports-publications/radiation/guidelines-safe-use-ultrasound-part-industrial-commercial-applications-safety-code-24.html)

| Option | Purpose |
| --- | --- |
| `--no-virtual-cam` | Run with the OpenCV preview only. |
| `--spatial` | Open the stereo echo dashboard with manual arming. |
| `--chirp` | Open the original gesture and breathing dashboard. |
| `--learn-zones` | Record short live trials and validate a local position profile. |
| `--ofdm` | Use the OFDM tracker on the same dashboard. |
| `--muted` | Open the live app without transmitting until you press M. |
| `--headless --duration S` | Run a timed smoke test without a preview. Physical output is muted by default. |
| `--headless --transmit --duration S` | Run a bounded acoustic test with real output. |
| `--max-volume GAIN` | Set linear peak amplitude within the existing cap. |
| `--max-seconds S` | Set the demo timer. |
| `--doppler-only` | Start the legacy two-tone mode for comparison. CW does not estimate range. |
| `--replay FILE.wav` | Replay a 48 kHz mono/stereo capture without transmission. Use `--nearfield-replay` for Phase 4 chirp captures or `--ranging-replay` for Phase 3 multitone captures. |

## Verification

Run the live tuner from the project root.

~~~powershell
.\.venv\Scripts\python.exe scripts/tune_and_verify.py
~~~

The tuner transmits in six-second bursts and scores the pure sonar frames. It measures in-gate return SNR against the matched-filter residual floor, the largest connected foreground region, consecutive-frame IoU, and left/right echo balance. Missing echoes count against the SNR score and blank frame pairs score zero IoU. A qualifying burst triggers a 20-second breathing check. Bat Cam loads the local `sonar_config.json` automatically when the tuner saves a profile that passes every check.

Each burst saves a CIR plot, radar, final and peak-coverage webcam frames, and JSON under `sonar_debug/`. The tuner uses the existing callback limits and caps test gain at 0.010. The range gate applies to accepted distance estimates. Unresolved physical echoes can be assigned an incorrect distance. Connected-component scores measure acoustic blobs and cannot verify facial anatomy.

[Live tuning results](TUNING.md) record the measured hardware outcomes and the reason no profile was promoted.

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py --simulate --headless --duration 13 --no-virtual-cam
~~~

To test with a separate DirectShow receiver, set `LAPTOPBAT_OBS_INDEX` to the camera index for OBS Virtual Camera. On the audited machine it is 0.

~~~powershell
$env:LAPTOPBAT_OBS_INDEX = '0'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_webcam_windows.py -v
Remove-Item Env:\LAPTOPBAT_OBS_INDEX
~~~

All 34 tests passed, including a separate DirectShow receiver that received changing 720p frames from OBS. The synthetic microphone tests check stationary in-gate reflectors, rejection of out-of-gate reflectors, clutter decay, four gesture directions, and breathing lock.

A live desktop test auto-armed and delivered 301 frames to OBS at approximately 30 FPS with zero reported audio faults or dropped buffers. The first probe timing reference took 0.12 seconds in that run and 0.08 seconds in a follow-up run. Range lock remained intermittent and the app withheld BPM.

The architecture plan for the original version is in [PLAN.md](PLAN.md). The near-field implementation is in `nearfield.py`, with duplex I/O in `audio_engine.py` and OpenCV rendering in `visualizer_cam.py`.
