#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
home = str(Path.home())
pth_base =  home+"/.kaggle/competitions/ai-hack-2018-minsk-stc/"
train_video_emb_fc_pth = pth_base + "train_video_emb_fc.csv"

data_train_video_emb_fc = pd.read_csv(train_video_emb_fc_pth)

print(data_train_video_emb_fc.columns)