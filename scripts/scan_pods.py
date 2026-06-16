import pandas as pd
import numpy as np

fp = r'd:\刘从睿\软件测试与维护\Final\data-collection-strong\combined_cartservice-cpu-extreme-corrected.csv'
df = pd.read_csv(fp)
df['ts'] = pd.to_datetime(df['timestamp'], unit='s')

# 拿每个 pod 的 CPU limit
limits = {}
for pod in df['pod'].unique():
    sub = df[(df['pod']==pod) & (df['metric']=='pod_cpu_limit')]
    if len(sub) > 0:
        limits[pod] = sub['value'].iloc[0]

print('Per pod CPU stats:')
print(f'{"pod":<40} {"limit":>8} {"mean%":>8} {"max%":>8} {"<15%":>8} {">20%":>8} {">50%":>8} {">80%":>8}')
for pod in sorted(df['pod'].unique()):
    sub = df[(df['pod']==pod) & (df['metric']=='cpu_usage')]
    if len(sub) == 0:
        continue
    limit = limits.get(pod, 1.0)
    util = sub['value'] / limit * 100
    mean_pct = util.mean()
    max_pct = util.max()
    n15 = (util < 15).sum()
    n20 = (util > 20).sum()
    n50 = (util > 50).sum()
    n80 = (util > 80).sum()
    print(f'{pod:<40} {limit:>8.2f} {mean_pct:>8.2f} {max_pct:>8.2f} {n15:>8} {n20:>8} {n50:>8} {n80:>8}')

# 找最有可能"有 <50% 且 >20% 段"的 pod
print()
print('=== Look for pods with both <50% normal and >20% anomaly segments ===')
for pod in sorted(df['pod'].unique()):
    sub = df[(df['pod']==pod) & (df['metric']=='cpu_usage')].sort_values('ts').reset_index(drop=True)
    if len(sub) == 0:
        continue
    limit = limits.get(pod, 1.0)
    sub['util_pct'] = sub['value'] / limit * 100
    has_low = (sub['util_pct'] < 50).sum()
    has_high = (sub['util_pct'] > 20).sum()
    if has_low > 50 and has_high > 5:
        print(f'  {pod}: <50%={has_low}, >20%={has_high}, max util={sub["util_pct"].max():.1f}%')
