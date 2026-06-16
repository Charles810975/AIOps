import pandas as pd
import numpy as np

fp = r'd:\刘从睿\软件测试与维护\Final\data-collection-strong\combined_cartservice-cpu-extreme-corrected.csv'
df = pd.read_csv(fp)
df['ts'] = pd.to_datetime(df['timestamp'], unit='s')
cart_cpu = df[(df['pod'].str.contains('cartservice')) & (df['metric']=='cpu_usage')].sort_values('ts').reset_index(drop=True)

print(f'=== 1. cartservice cpu_usage 数据 ===')
print(f'共 {len(cart_cpu)} 点')
print(f'time range: {cart_cpu["ts"].min()} ~ {cart_cpu["ts"].max()}')

# 1) 步长连续性
diffs = cart_cpu['ts'].diff().dt.total_seconds().dropna()
print()
print('=== 2. 采集步长 ===')
print(f'  mean={diffs.mean():.2f}s, median={diffs.median():.2f}s, min={diffs.min():.1f}s, max={diffs.max():.1f}s')
print(f'  步长=5s: {(diffs==5).sum()}/{len(diffs)} = {(diffs==5).mean()*100:.1f}%')
print(f'  步长>10s (>=2步缺失): {(diffs>10).sum()}')
print(f'  步长>30s: {(diffs>30).sum()}')

# 找缺失段
big_gap = cart_cpu[cart_cpu['ts'].diff().dt.total_seconds() > 30]
if len(big_gap) > 0:
    print('  跳变 > 30s:')
    print(big_gap[['ts','value','label']].to_string())
else:
    print('  没有任何 > 30s 的跳变 (步长完全均匀)')

# 2) 是不是只有 1 段连续
print()
print('=== 3. 分钟分桶 ===')
cart_cpu['minute'] = cart_cpu['ts'].dt.floor('min')
print(f'  共 {cart_cpu["minute"].nunique()} 个不重复分钟')
sizes = cart_cpu.groupby('minute').size()
print(f'  每分钟点数: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}')

# 3) label 时序分布
print()
print('=== 4. label 时序分布 ===')
print('前 5 行的 label 和 value:')
print(cart_cpu[['ts','value','label']].head().to_string(index=False))
print('最后 5 行的 label 和 value:')
print(cart_cpu[['ts','value','label']].tail().to_string(index=False))
print()
# 找 normal 和 anomaly 的分界点
switch = cart_cpu[cart_cpu['label'].shift() != cart_cpu['label']]
print('normal/anomaly 切换点:')
print(switch[['ts','value','label']].to_string(index=False))

# 4) 与窗口元数据对比
import json
print()
print('=== 5. window_info.json ===')
with open(r'd:\刘从睿\软件测试与维护\Final\data-collection-strong\window_info.json') as f:
    info = json.load(f)
print(json.dumps(info, indent=2))
