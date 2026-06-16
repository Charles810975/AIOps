import pandas as pd
import numpy as np

fp = r'd:\刘从睿\软件测试与维护\Final\data-collection-strong\combined_cartservice-cpu-extreme-corrected.csv'
df = pd.read_csv(fp)
df['ts'] = pd.to_datetime(df['timestamp'], unit='s')

cart_cpu = df[(df['pod'].str.contains('cartservice')) & (df['metric']=='cpu_usage')].copy()
cart_cpu = cart_cpu.sort_values('ts').reset_index(drop=True)
print(f'cartservice cpu_usage total points: {len(cart_cpu)}')
print(f'time range: {cart_cpu["ts"].min()} ~ {cart_cpu["ts"].max()}')
print(f'step: {(cart_cpu["ts"].diff().dt.total_seconds()).median():.1f}s')
print()

# 看 value 分布 + 正常/异常分段
cart_cpu['value_pct'] = cart_cpu['value'].rank(pct=True) * 100
print('value percentiles:')
for p in [1, 25, 50, 75, 90, 95, 99, 99.5, 100]:
    print(f'  p{p}: {cart_cpu["value"].quantile(p/100):.6f}')
print()

# 找正常段（value < 15%分位 或 value < 某个绝对阈值）和异常段
# 注意：value 是 CPU usage rate，0.01 = 1% 单核
# 实际 cartservice 的 cpu_usage 在 normal 时 0.0004，anomaly 时 0.0037
# 0.0037 已经是 0.37% 看着不大，但实际是 0.3 核的 cpu_limit 的相对值

# 算前后段
n = len(cart_cpu)
mid = n // 2
first_half = cart_cpu.iloc[:mid]
second_half = cart_cpu.iloc[mid:]

print(f'first half: {len(first_half)} points, mean={first_half["value"].mean():.6f}, max={first_half["value"].max():.6f}')
print(f'  label counts: {first_half["label"].value_counts().to_dict()}')
print(f'second half: {len(second_half)} points, mean={second_half["value"].mean():.6f}, max={second_half["value"].max():.6f}')
print(f'  label counts: {second_half["label"].value_counts().to_dict()}')
print()

# 看看 cpu limit
limits = df[(df['pod'].str.contains('cartservice')) & (df['metric']=='pod_cpu_limit')]
if len(limits) > 0:
    print(f'cartservice cpu_limit: {limits["value"].iloc[0]} cores (1.0 = 1 full core)')
    # 实际负载率
    print(f'normal 相对 cpu_limit: {first_half["value"].mean() / limits["value"].iloc[0] * 100:.2f}%')
    print(f'anomaly 相对 cpu_limit: {second_half["value"].mean() / limits["value"].iloc[0] * 100:.2f}%')
print()

# 找正常段（< 15% of limit）和异常段（> 20% of limit）
limit_val = limits['value'].iloc[0] if len(limits) > 0 else 1.0
cart_cpu['util_pct'] = cart_cpu['value'] / limit_val * 100

print('utilization pct:')
print(f'  mean: {cart_cpu["util_pct"].mean():.2f}%')
print(f'  max: {cart_cpu["util_pct"].max():.2f}%')
print(f'  points < 15% util: {(cart_cpu["util_pct"] < 15).sum()}')
print(f'  points > 20% util: {(cart_cpu["util_pct"] > 20).sum()}')
print(f'  points > 50% util: {(cart_cpu["util_pct"] > 50).sum()}')
print(f'  points > 80% util: {(cart_cpu["util_pct"] > 80).sum()}')
print()

# 找连续正常段
cart_cpu['is_normal_util'] = cart_cpu['util_pct'] < 15
cart_cpu['is_anom_util'] = cart_cpu['util_pct'] > 20

# 滑动窗口找最长连续段
def longest_run(mask):
    if not mask.any():
        return 0
    groups = (mask != mask.shift()).cumsum()
    return mask.groupby(groups).sum().max()

print(f'longest <15% run: {longest_run(cart_cpu["is_normal_util"])} points')
print(f'longest >20% run: {longest_run(cart_cpu["is_anom_util"])} points')
print(f'longest >50% run: {longest_run(cart_cpu["util_pct"] > 50)} points')
print(f'longest >80% run: {longest_run(cart_cpu["util_pct"] > 80)} points')

# 看第二段最后一段
print('\nlast 30 points util% and label:')
for i, r in cart_cpu.tail(30).iterrows():
    print(f'  {r["ts"]} util={r["util_pct"]:.1f}% label={r["label"]}')
