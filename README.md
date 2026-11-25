# Laptop-as-a-Bat

Bat Cam uses a Windows laptop's speakers and microphone array as a near-field sonar. It auto-arms at launch, estimates echo range and relative left/right position, and sends the shaded view to an OpenCV preview and OBS Virtual Camera at a target of 1280 × 720 and 30 FPS.

## Run

In PowerShell from the project root, use 64-bit Python 3.12. Skip the first two commands if the project environment already exists.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
~~~

The live app starts transmitting as soon as the audio stream and preview are ready. It uses the existing −48 dBFS default peak gain per channel and shuts down at its configured output limits. The range view adapts while you remain in front of the laptop. Use the following command for a silent preview.

~~~powershell
.\.venv\Scripts\python.exe main.py --simulate --no-virtual-cam
~~~

The simulator uses generated microphone samples and an acoustic room model. It takes no input from the microphone and leaves the physical speakers silent. Enter `--muted` to open the live microphone and preview with the speakers muted.

### Windows audio and OBS

On the audited ASUS laptop, Bat Cam automatically selects the Realtek microphone array and speakers. Use `--list-devices` to identify WASAPI IDs on other laptops, then pass `--input-device ID --output-device ID` at launch. Windows needs microphone access for desktop apps and a two-channel 48 kHz output format. An input endpoint can fall back to mono. Bat Cam requests WASAPI RAW. If the device applies OEM audio processing, disable noise cancellation, beamforming and spatial effects on the selected route. [Windows audio processing modes](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/audio-signal-processing-modes)

Install the OBS Studio Virtual Camera component. Leave OBS's own virtual-camera producer stopped and select **OBS Virtual Camera** in the receiving app. If the driver is missing or busy, Bat Cam continues in the OpenCV preview. [OBS Studio](https://obsproject.com/download)

## Controls

| Key | Action |
| --- | --- |
| **V** | Cycle the near-field view with HUD, gated-return debug view and pure sonar view. |
| **H** | Hide or restore the HUD and tracking overlays. |
| **M** or **Space** | Mute or re-arm transmission within the exposure budget. |
| **R** | Reset the demo timer and resume when exposure budget permits. |
| **Esc** or **Q** | Mute and close the app. |

The camera mode needs no calibration keys. Gesture detection uses tracks from range-gated echoes. Push approaches the laptop, Pull moves away, and Swipe Left/Right changes relative lateral position.

## What the camera measures

- Each speaker takes a turn transmitting a 512-sample, Tukey-windowed chirp over 17.5–21.5 kHz. Its 18.5 or 20.5 kHz pilot fits in the slot after the chirp. Audio runs at 48 kHz. A range update is computed up to 25 times a second, independently of the webcam's 30 FPS frame clock.
- Stereo matched filtering separates the two chirps. Direct-path estimates and rolling band gains normalize the paths automatically. The room model updates at alpha 0.02. A stationary return can fade as the model learns it. Nearby movement and chest micro-motion refresh the image.
- The processor admits taps estimated at 0.15–0.95 m to image, gesture and breathing processing. The delay samples are spaced 3.57 mm apart. With 4 kHz of bandwidth, ideal reflector separation is about 4.3 cm, and the effective bandwidth on laptop transducers can be narrower. Device geometry, noise and multipath affect where an echo is estimated, especially close to a gate edge. The default reference path is 0.15 m. Use `--direct-path-m` to supply a measured speaker-to-mic distance. [Pulse compression and range resolution](https://www.ll.mit.edu/sites/default/files/outreach/doc/2018-07/lecture%205.pdf)
- Nearer echoes render brighter in a perspective range-versus-lateral mesh. Contours and reticles come from the supported echoes. The horizontal laptop array provides two useful spatial axes. Measuring face geometry, elevation, yaw or distinguishing shoulders from the torso would require additional spatial measurements.
- Breathing uses a stable high-SNR return in the 0.25–0.85 m zone. It combines complex phase and amplitude with a 0.12–0.55 Hz bandpass, spectral peak and autocorrelation. It updates twice a second after at least 7 seconds of signal and about two consistent breathing cycles. Slow breathing requires a longer window. BPM is withheld when confidence is insufficient.

## Limits and diagnostics

Peak amplitude is capped at approximately −30 dBFS per channel. The default rolling limit is 120 seconds of emission in 180 seconds. The separate demo timer defaults to 300 seconds. A reset preserves the rolling limit. `--auto-cooldown` waits at least two seconds and until enough budget is available to resume. These values describe digital output. Sound-pressure level depends on the laptop, system volume and listener position. [Airborne ultrasound measurement guidance](https://www.canada.ca/en/health-canada/services/environmental-workplace-health/reports-publications/radiation/guidelines-safe-use-ultrasound-part-industrial-commercial-applications-safety-code-24.html)

| Option | Purpose |
| --- | --- |
| `--no-virtual-cam` | Run with the OpenCV preview only. |
| `--view pure` or `--view debug` | Set the initial view mode. |
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
