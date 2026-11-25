# Laptop-as-a-Bat

Bat Cam turns a Windows laptop's speakers and microphone array into an acoustic camera. The default view shows a full-frame thermal-colored image of echoes by approximate range and relative left/right position. The same 1280 × 720 frames go to the preview and OBS Virtual Camera at a target of 30 FPS. A debug view shows the existing Doppler radar, spectra and gesture log.

## Run it

From the project root in PowerShell, use 64-bit Python 3.12. The project environment can be used directly without activating it.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --simulate --no-virtual-cam
~~~

The synthetic demo uses no microphone or speakers. It produces delayed microphone samples for a stationary room, a breathing torso and a moving foreground echo. The range view appears after about 5 s and a 15 BPM estimate normally appears within 20 s. Close the window or press Esc to quit.

### Set up Windows audio and the webcam

1. In **Settings → System → Sound**, select **Microphone Array (Realtek(R) Audio)** for input and **Speakers (Realtek(R) Audio)** for output. Open each device's properties. Select a two-channel 48,000 Hz format where available, set **Audio enhancements** to **Off**, and disable spatial sound and mono audio. Older drivers expose enhancement and format controls under **More sound settings → Recording/Playback → Properties → Enhancements/Advanced**.
2. Disable echo cancellation, noise suppression, automatic gain control, beamforming and AI noise cancellation in the manufacturer's audio utility where offered. Enable microphone access for desktop apps in **Privacy & security → Microphone**. Use the laptop's built-in speakers at low, fixed system volume. Keep the lid angle and microphone gain fixed for a session.
3. The engine requests **WASAPI RAW** on both stream endpoints with format auto-conversion disabled. This uses PortAudio's native `eStreamOptionRaw` through the pinned sounddevice binding. Windows drivers advertise RAW capability separately. On the audited ASUS laptop both Realtek endpoints advertised support. For other drivers the console reports the request, and the device settings above matter. [Windows signal processing modes](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/audio-signal-processing-modes) · [Windows RAW capability property](https://learn.microsoft.com/en-us/windows/win32/properties/props-system-devices-audiodevice-rawprocessingsupported)
4. For webcam output, install OBS Studio with its Windows Virtual Camera component. Leave OBS's **Start Virtual Camera** producer stopped. Start Bat Cam, then select **OBS Virtual Camera** in the receiving app. A registered Unity Capture device is a secondary backend. A missing or busy driver leaves the OpenCV preview running. [OBS Studio](https://obsproject.com/download) · [pyvirtualcam Windows backends](https://github.com/letmaik/pyvirtualcam)

~~~powershell
.\.venv\Scripts\python.exe main.py --list-devices
.\.venv\Scripts\python.exe main.py
~~~

On the audited laptop the Realtek WASAPI input was ID 15 and output was ID 12. IDs can change when devices reconnect. Automatic selection uses the WASAPI host and Realtek names. This command explicitly selects the audited pair.

~~~powershell
.\.venv\Scripts\python.exe main.py --input-device 15 --output-device 12 --left-freq 18500 --right-freq 20500 --max-volume 0.004 --calibration-duration 3
~~~

### Live session

1. Leave the scene clear for **3 s of muted ambient calibration**. Focus Bat Cam and press **M** or **Space** to arm. Keep the scene clear for another **5 seconds** so the range processor can learn the room background and the Doppler processor can measure direct-carrier gain. Step into view after calibration.
2. Press **C** when range has locked. Wave on the left for the first half of the three-second guide, then wave on the right at the same depth. Distinguishable returns calibrate the lateral cue, enable the tracking box, and save offsets to `calibration.json` for that audio route and carrier pair. Weak or overlapping samples prompt a retry.
3. Sit facing the laptop about 0.3–1.0 m away and keep your hands still. The camera estimator tracks complex phase and amplitude within the strongest stable torso range cell. The first estimate needs at least 10 s and about two complete breathing cycles, so a slow rate can take longer. Confidence and the waveform update once a second. Muting or losing the tracked echo clears the estimate.
4. Press **V** to cycle Pure Bat Cam, Bat Cam + HUD and Debug Spectrogram. **H** toggles the overlays. **R** resets the demo timer and attempts to resume within the available output budget. **M** mutes immediately or arms again after ambient calibration. **Esc**, **Q** or window close exits.

The range view places nearer returns toward the bottom. It uses 18.0–21.5 kHz multitone probes on disjoint speaker bins and correlates their reflections against the transmitted waveform. Direct-path timing cancels device latency and normalizes gain asymmetry. The vertical axis measures approximate range relative to the direct speaker-to-microphone path. The horizontal axis estimates relative lateral position using amplitude and a bounded stereo phase cue. This array measures range and lateral position. Elevation and optical body outlines require a different sensor geometry. Acoustic measurements update at up to 25 Hz independently of the 30 FPS output.

The original CW debug view still shows signed radial speed. **Push** moves toward the screen and **Pull** moves away. Press **1**, **2**, then **3** while moving in the left, center and right zones for the original Doppler swipe calibration. Use `--doppler-only` to give the original two tones the full configured output on hardware with weak range returns.

### Output limits and CLI

Live transmission starts muted at approximately −48 dBFS peak per channel. The hard per-channel ceiling is −30 dBFS (0.03162 linear). A 20 ms onset ramp, callback-enforced timeout and rolling time budget apply to all live emission. The default is at most 120 s on in any 180 s. Manual toggles preserve the budget. A stream fault or missed control heartbeat mutes the next available block. The meter shows digital gain and transmit time. Acoustic exposure depends on the speakers, Windows volume and listener position. Stop on discomfort. [Airborne ultrasound measurement guidance](https://www.canada.ca/en/health-canada/services/environmental-workplace-health/reports-publications/radiation/guidelines-safe-use-ultrasound-part-industrial-commercial-applications-safety-code-24.html)

The separate demo timer defaults to 300 seconds from arming. If the rolling exposure budget runs out first, the app mutes at that point. `--auto-cooldown` waits at least two seconds and resumes when enough rolling budget is available for another acquisition. A full 120-second burst can require much more than two seconds of silence. Ordinary mutes preserve the learned room background and restart breathing acquisition.

| Option | Use |
| --- | --- |
| `--input-device ID --output-device ID` | Select a WASAPI mic and stereo speakers. Stereo capture falls back to one channel if necessary. |
| `--left-freq HZ --right-freq HZ` | Set carriers between 18 and 22 kHz, with the right at least 1 kHz above the left. Defaults are 18.5 and 20.5 kHz. |
| `--max-volume GAIN` | Set the per-channel linear peak amplitude, up to the hard ceiling. |
| `--session-seconds S --duty-window S` | Shorten the permitted on-time or lengthen the rolling window. Defaults are 120 and 180 s. |
| `--max-seconds S --auto-cooldown` | Set the separate demo timer and optionally resume after budget allows. |
| `--view hud`, `--view pure`, `--view debug` | Select the initial video mode. |
| `--calibration-file PATH` | Select the route-specific lateral calibration file. |
| `--direct-path-m M` | Supply an approximate speaker-to-mic distance to shift the range axis. |
| `--doppler-only` | Use the original carrier power and start in debug view. |
| `--calibration-duration S` | Set the initial muted interval, from 0.1 to 30 s. The separate quiet armed interval lasts 5 s. |
| `--exclusive` | Request exclusive WASAPI with the same RAW flag. Other audio apps may need to release the device. |
| `--no-virtual-cam` or `--preview-only` | Run the local window without opening a virtual webcam. |
| `--headless --duration S` | Render for a bounded interval without a window. Live input stays muted. Exit prints frame counts, stream closure, drops and camera status as JSON. |
| `--simulate` or `--replay CAPTURE.wav` | Use generated echoes or a mono/stereo 48 kHz WAV. Both modes disable physical transmission. For replay, match the carrier and ambient-duration flags to the recording, which must begin with muted ambient followed by 5 s of quiet carriers. Add `--ranging-replay` for recordings containing the Phase 3 probe. |

## Reproduce verification

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py --headless --no-virtual-cam --duration 10
.\.venv\Scripts\python.exe main.py --simulate --headless --duration 24
~~~

The first command tests synthetic microphone PCM including unequal path gains, additive noise, clock drift, randomized gesture order, phase and amplitude breathing modulation, and missing or interrupted signals. It also exercises the exposure limiter, capture queue, WAV teardown, frame format and camera failure paths. The second uses the real duplex stream and checks muted shutdown. The third sends a generated scene through the normal webcam pipeline.

The real receiver test is opt-in. Set `LAPTOPBAT_OBS_INDEX` to the **DirectShow camera** index of OBS Virtual Camera (0 on the audited host). Audio device IDs are separate. Close other webcam producers first. The test starts Bat Cam in a separate process and checks changing echo pixels in the received 720p image.

~~~powershell
$env:LAPTOPBAT_OBS_INDEX = '0'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_webcam_windows.py -v
Remove-Item Env:\LAPTOPBAT_OBS_INDEX
~~~

### Phase 3 verification on the audited laptop

| Check | Observed result |
| --- | --- |
| Runtime | Windows 11 Pro, Python 3.12.10 x64. NumPy 2.5.5, SciPy 1.18.1, sounddevice 0.5.5, opencv-python 4.14.0.94, pyvirtualcam 0.15.0 and OBS Studio 33.0.3. |
| Synthetic gestures | 25/25 randomized trials classified correctly across the five labels, with exactly one event per intended gesture. |
| Synthetic range and breathing | Stationary echoes were located at their injected ranges and the learned wall return was suppressed. A 24 s run produced 720 frames at 29.98 FPS, detected all five gestures and estimated 15.01 BPM from a 15 BPM torso echo. |
| OBS Virtual Camera | A separate DirectShow process received 45 frames at 1280 × 720 and observed 31 changes in the sonar image interior. |
| Physical I/O | The Realtek pair opened at 48 kHz stereo and closed cleanly in the muted smoke test. An eight-second check of the first Phase 3 probe produced intermittent range lock with zero stream faults and zero dropped blocks. |

The accuracy figures use generated echoes. Stable live imaging, guided calibration and BPM lock on this laptop still need a person and a known target at the device.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Cannot open audio | Use the WASAPI entries from `--list-devices`, enable desktop microphone access and check both devices' 48 kHz format. Switch from exclusive to the default shared mode if another app owns the device. |
| No range reference or empty frame | Check that output is armed and use `V` to inspect the Doppler SNR. Select the built-in Realtek route, disable OEM processing and keep the scene clear during startup. `--doppler-only` restores the original two-tone mode. |
| Left/right unavailable or reversed | Press `C` and repeat the guided wave from the seated viewpoint. Keep depth similar on both sides. Weak paths or duplicated mic channels can leave range as the usable axis. |
| Waiting for torso echo | Sit quietly in the torso range zone after the empty-room calibration. Shift slightly to avoid an echo null, then hold still through several breath cycles. |
| Camera missing, frozen or busy | Keep OBS's own virtual-camera producer stopped and select OBS Virtual Camera as the input in the receiving app. The dashboard's top-right label and exit summary show the backend status. Use `--no-virtual-cam` to isolate audio/DSP. |
| Output timed out | Wait for rolling on-time to become available, then press R. If a stream fault stopped audio, check the route and restart Bat Cam. |

The original DSP and device choices are documented in [PLAN.md](PLAN.md). Implementation lives in `audio_engine.py` (duplex capture and limits), `dsp_processor.py` (Doppler sensing), `sonar_imaging.py` (ranging, lateral calibration and torso tracking), `visualizer_cam.py` (frames and camera output), and `main.py` (CLI, controls and replay).
