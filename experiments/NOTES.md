# SR-CNN Single-Variable Auto Iter (target F1=0.6)
## Label rule
- pod: cartservice
- metric: cpu_usage
- normal: value < 0.5 * max(value)
- anomaly: value >= 0.5 * max(value)

## iter 03 - sr_iter03_podcartservice_mcpu_v3
- F1=0.6923, P=0.6000, R=0.8182
- amp=9, sw=27, diff=0, smooth=0, preproc=zscore, q=0.97

