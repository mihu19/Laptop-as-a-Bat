# Live sonar tuning results

The Realtek endpoints ran at 48 kHz. Seven bursts emitted for a combined 41.98 seconds and sent 1,262 frames to OBS Virtual Camera. PortAudio reported zero stream faults and the capture queue dropped zero blocks. The emitted signal stayed below the existing peak cap.

| Trial | Settings changed | Peak in-gate SNR | Result |
| --- | --- | ---: | --- |
| Standard burst 1 | 17.5–21.5 kHz, gain 0.00398, MTI alpha 0.020 | 19.6 dB | Timing reference held for 2.4% of scored acoustic samples. |
| Standard burst 2 | Gain 0.00617, alpha 0.0134, noise multiplier 3.7 | 14.0 dB | Timing reference held for 3.9% of samples. |
| Standard burst 3 | Gain 0.00956, alpha 0.0090, noise multiplier 3.4 | 16.3 dB | Timing reference held for 68.9% of samples. Final frame showed separate echo patches. |
| Wide burst 1 | 17–22 kHz and gain 0.00956 | 13.3 dB | Median foreground coverage and IoU were zero. |
| Wide burst 2 | Gain 0.010, alpha 0.0134, noise multiplier 3.7 | 15.0 dB | Timing reference held for 99.3% of samples, but a usable in-gate return appeared in only 6.0%. The peak-coverage frame showed a broad range band. |
| Stability trial | Wide chirp, gain 0.010, persistence 0.8, Gaussian smoothing 0.8 | 14.1 dB | Median foreground coverage and IoU remained zero. |
| Final verification with localization check | Repeated stability settings | 20.6 dB | Detectable returns appeared in 1.3% of scored acoustic samples. Peak foreground coverage reached 5.1%, with median coverage and IoU at zero, stereo imbalance of 37.9%, and no BPM reading. |

All trials had zero energy in the post-gate output outside the represented range. None met the median SNR, connected foreground coverage, IoU, stereo balance or breathing checks. Peak SNR describes a momentary return. The wide-band result illustrates why a stable transmit timing reference alone is insufficient for imaging. No settings qualified for promotion.

## Low-band portrait trials

Two six-second Realtek captures used a 14.5–17.5 kHz sweep and 14.8/16.8 kHz positioning pilots. At gain 0.00398 the tuner found no usable in-gate echoes. At gain 0.010 the probe timing reference held throughout, but a usable return appeared in 23.6% of scored samples and came from only one speaker path. Peak SNR was 17.9 dB. Median coverage and IoU were zero, BPM was unavailable, and neither trial qualified for promotion. OBS received 361 frames across the two trials with zero reported audio faults or dropped blocks.

## Phase 7 acoustic model and display

The final MLP achieved 98.33% accuracy on 720 simulated validation windows. The Ridge model's mean absolute errors on the same simulated subjects were 0.0345 on the normalized lateral axis and 1.39 cm in depth. Controlled microphone PCM replay exercised all four learned gesture classes. These figures characterize the simulator and replay fixture.

For the final six-second Realtek verification, the tuner used 12.0–16.5 kHz chirps, 13.0/15.5 kHz pilots, and a capped gain of 0.010 after a three-second muted ambient measurement. OBS received 180 frames. The renderer produced 29.83 FPS, peak model update time was 0.84 ms, and there were no reported audio faults or dropped blocks. The labeled ambient display covered 45.7% of the frame and had mean grayscale brightness of 71.1 on the 0–255 scale.

The live burst produced only intermittent returns. Peak in-gate SNR was 16.0 dB, gated detections occurred in 1.25% of scored samples, and no stable subject or BPM qualified. Median acoustic foreground coverage was zero. The visible ambient scan passed the display target, while the independent acoustic checks rejected the trial. No live tuning profile qualified for promotion.

## Spatial echo projection

The 6–13.5 kHz stereo multicarrier probe acquired timing on the Realtek device in 0.15 seconds at the default transmit gain and completed room calibration. A guided hand-wave run emitted for 21.17 seconds with zero reported audio faults or dropped blocks. Two calibrated paths picked up a new echo in 5 of 219 snapshots during the left-hand segment and 3 of 216 snapshots during the right-hand segment. The rest had evidence on at most one path. No snapshot met the three-path requirement for the front projection. The range history remains available for those returns.

Simulated microphone samples confirmed that the projection appears when delays agree across multiple paths, moves as the reflector positions change, and clears after loss of timing. These tests exercise the signal processing and dashboard. Positions on the physical laptop remain unverified.

All 40 automated tests passed. A separate DirectShow receiver saw 42 changing frames out of 45 at 1280 × 720. A muted five-second run through the normal `main.py` entry point also opened both Realtek channels, sent 150 frames to OBS, and closed the stream cleanly.
