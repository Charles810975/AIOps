import pandas as pd
import json
fp = r'd:\刘从睿\软件测试与维护\Final\data-collection-strong\combined_cartservice-cpu-extreme-corrected.csv'
df = pd.read_csv(fp)
print('rows:', len(df))
print('columns:', list(df.columns))
print('pods:', df['pod'].unique()[:5])
print('metrics:', df['metric'].unique()[:10])
print('label counts:')
print(df['label'].value_counts())
print('first 3 rows:')
print(df.head(3))
print('last 3 rows:')
print(df.tail(3))
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
print('time range:', df['timestamp'].min(), '~', df['timestamp'].max())
dur_h = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
print(f'duration hours: {dur_h:.2f}')
print()
print('per label time range:')
for lbl in df['label'].unique():
    sub = df[df['label']==lbl]
    print(f'  {lbl}: {len(sub)} rows, time {sub["timestamp"].min()} ~ {sub["timestamp"].max()}')
print()
# 看 cartservice 的 cpu_usage 序列
cart = df[(df['pod'].str.contains('cartservice')) & (df['metric']=='cpu_usage')].copy()
print('cartservice cpu_usage rows:', len(cart))
if len(cart) > 0:
    cart = cart.sort_values('timestamp')
    print('first 3 cart cpu_usage:')
    print(cart.head(3))
    print('last 3 cart cpu_usage:')
    print(cart.tail(3))
    print('value stats:')
    print(cart['value'].describe())
    print('value stats by label:')
    for lbl in cart['label'].unique():
        s = cart[cart['label']==lbl]['value']
        print(f'  {lbl}: mean={s.mean():.6f}, std={s.std():.6f}, max={s.max():.6f}')
