#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Get home path 
home = str(Path.home())

# Define base location of datasets
pth_base =  home+"/.kaggle/competitions/ai-hack-2018-minsk-stc/"

# Dataset paths
t_video_emb_fc_pth   = pth_base + "train_video_emb_fc.csv"
t_target_pth         = pth_base + "train_target.csv"
v_video_emb_fc_pth   = pth_base + "val_video_emb_fc.csv"
v_target_pth         = pth_base + "val_target.csv"
# Load data
train_in = pd.read_csv(t_video_emb_fc_pth)
train_out = pd.read_csv(t_target_pth)
val_in = pd.read_csv(v_video_emb_fc_pth)
val_out = pd.read_csv(v_target_pth)

# Choose all points for learning (ommitting 'ID' column)
cols = [str(x) for x in range(1024)]
t_X = train_in[cols]
v_X = val_in[cols]
#print(X)

# Choose output values
t_y = train_out['Label']
v_y = val_out['Label']
#print(y)

# Load model
forest = RandomForestClassifier()
forest.fit(t_X,t_y)
out = forest.predict(v_X)

correct = [index for index, value in enumerate(out) if v_y[index] == value]
wrong = [index for index, value in enumerate(out) if v_y[index] != value]

print(len(correct), len(wrong))
