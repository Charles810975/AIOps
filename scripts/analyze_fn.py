import pandas as pd
import numpy as np

# 读 iter2 best
fp = r'd:\刘从睿\软件测试与维护\Final\experiments\sr_iter02_podcartservice_mcpu_v2\sr_cnn_results.csv'
df = pd.read_csv(fp)
print('total rows:', len(df))
print('label_rule counts:', df['label_rule'].value_counts().to_dict())
print('pred counts:', df['pred'].value_counts().to_dict())

# anomaly 段：value >= 0.006553
anom = df[df['label_rule'] == 'anomaly'].copy()
print(f'\nanomaly 段 ({len(anom)} rows):')
print(anom[['timestamp', 'value', 'sr_score', 'pred']].to_string())

# 漏检：anomaly 但 pred=0
fn = anom[anom['pred'] == 0]
print(f'\nFN ({len(fn)} rows, 漏检):')
print(fn[['timestamp', 'value', 'sr_score']].to_string())
print(f'\nFN value range: {fn["value"].min():.6f} ~ {fn["value"].max():.6f}')
print(f'TP value range: {anom[anom["pred"]==1]["value"].min():.6f} ~ {anom[anom["pred"]==1]["value"].max():.6f}')
print(f'\nthreshold in best: {df["pred"].max() if "pred" in df.columns else "?"}')

# FP：normal 但 pred=1
fp_df = df[(df['label_rule'] == 'normal') & (df['pred'] == 1)]
print(f'\nFP ({len(fp_df)} rows, 误报):')
print(fp_df[['timestamp', 'value', 'sr_score']].to_string())
print(f'FP value range: {fp_df["value"].min():.6f} ~ {fp_df["value"].max():.6f}')
