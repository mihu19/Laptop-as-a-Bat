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
