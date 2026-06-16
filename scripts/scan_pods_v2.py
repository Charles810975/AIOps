import pandas as pd
import numpy as np

fp = r'd:\刘从睿\软件测试与维护\Final\data-collection-strong\combined_cartservice-cpu-extreme-corrected.csv'
df = pd.read_csv(fp)
df['ts'] = pd.to_datetime(df['timestamp'], unit='s')

print('=== Per pod: max util% and 50%-of-max threshold ===')
print(f'{"pod":<40} {"max%":>8} {"50%thresh%":>12} {"<50%max":>10} {">50%max":>10} {"max%":>8}')

candidates = []
for pod in sorted(df['pod'].unique()):
    sub = df[(df['pod']==pod) & (df['metric']=='cpu_usage')].sort_values('ts').reset_index(drop=True)
    if len(sub) == 0:
        continue
    # 相对该 pod max 的 50% 阈值
    mx = sub['value'].max()
    thr = mx * 0.5
    n_low = (sub['value'] < thr).sum()
    n_high = (sub['value'] >= thr).sum()
    util_max_pct = mx / df[(df['pod']==pod)&(df['metric']=='pod_cpu_limit')]['value'].iloc[0] * 100
    print(f'{pod:<40} {mx:>8.6f} {thr:>12.6f} {n_low:>10} {n_high:>10} {util_max_pct:>8.2f}')
    if n_low >= 50 and n_high >= 5:
        candidates.append((pod, mx, thr, n_low, n_high, util_max_pct))

print()
print(f'candidates (>=50 low + >=5 high): {len(candidates)}')
for c in candidates:
    print(f'  {c[0]}: max={c[1]:.6f} thr={c[2]:.6f} low={c[3]} high={c[4]} max%={c[5]:.2f}')
