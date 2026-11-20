# 导读 遍历各应用与设备的放置画像，统计每个 local_bsz 的记录数，并把样本数相同的配置分组。
# 导读 来源 traces/check_sizes.py。以下带「导读」的行是阅读注释。
import pandas as pd
import os

num_rows_to_configs = {}
models = os.listdir('./')
models.remove('philly.csv')
models.remove('check_sizes.py')
for model in models:
    for device in ['rtx', 'dgx', 'aws']:
        df = pd.read_csv('%s/placements-%s.csv'%(model, device))
        bszs = set(df[['local_bsz']].to_numpy()[:, 0])
        for bsz in bszs:
            if df[df['local_bsz']==bsz].shape[0] in num_rows_to_configs:
                num_rows_to_configs[df[df['local_bsz']==bsz].shape[0]].append((model, device, bsz))
            else:
                num_rows_to_configs[df[df['local_bsz']==bsz].shape[0]] = [(model, device, bsz)]
