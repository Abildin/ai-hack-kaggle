#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Get home path 
home = str(Path.home())

# Define base location of datasets
pth_base =  home+"/.kaggle/competitions/ai-hack-2018-minsk-stc/"

# Dataset paths
train_video_emb_fc_pth = pth_base + "train_video_emb_fc.csv"
train_target_pth = pth_base + "train_target.csv"

# Load data
train_in = pd.read_csv(train_video_emb_fc_pth)
train_out = pd.read_csv(train_target_pth)

# Choose all points for learning (ommitting 'ID' column)
cols = [str(x) for x in range(1024)]
X = train_in[cols]
print(X)
