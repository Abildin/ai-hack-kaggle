#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# Get home path 
home = str(Path.home())

# Define base location of datasets
pth_base =  home+"/.kaggle/competitions/ai-hack-2018-minsk-stc/"

# Dataset paths
t_video_emb_fc_pth   = pth_base + "train_" + "video_emb_fc.csv"
t_video_emb_pool_pth = pth_base + "train_" + "video_emb_pool.csv"
t_audio_feat_1_pth   = pth_base + "train_" + "audio_feat_1.csv"
t_audio_feat_2_pth   = pth_base + "train_" + "audio_feat_2.csv"
t_target_pth         = pth_base + "train_" + "target.csv"

v_video_emb_fc_pth   = pth_base + "val_" + "video_emb_fc.csv"
v_video_emb_pool_pth = pth_base + "val_" + "video_emb_pool.csv"
v_audio_feat_1_pth   = pth_base + "val_" + "audio_feat_1.csv"
v_audio_feat_2_pth   = pth_base + "val_" + "audio_feat_2.csv"
v_target_pth         = pth_base + "val_" + "target.csv"

tst_video_emb_fc_pth   = pth_base + "test_" + "video_emb_fc.csv"
tst_video_emb_pool_pth = pth_base + "test_" + "video_emb_pool.csv"
tst_audio_feat_1_pth   = pth_base + "test_" + "audio_feat_1.csv"
tst_audio_feat_2_pth   = pth_base + "test_" + "audio_feat_2.csv"

# Load data
train_in_video_fc   = pd.read_csv(t_video_emb_fc_pth)
train_in_video_pool = pd.read_csv(t_video_emb_pool_pth)
train_in_audio_1    = pd.read_csv(t_audio_feat_1_pth)
train_in_audio_2    = pd.read_csv(t_audio_feat_2_pth)
train_sets          = [train_in_audio_1, train_in_audio_2, train_in_video_fc, train_in_video_pool]
train_in_all        = pd.concat(train_sets, axis=1)
train_out           = pd.read_csv(t_target_pth)

val_in_video_fc     = pd.read_csv(v_video_emb_fc_pth)
val_in_video_pool   = pd.read_csv(v_video_emb_pool_pth)
val_in_audio_1      = pd.read_csv(v_audio_feat_1_pth)
val_in_audio_2      = pd.read_csv(v_audio_feat_2_pth)
val_sets            = [val_in_audio_1, val_in_audio_2, val_in_video_fc, val_in_video_pool]
val_in_all          = pd.concat(val_sets, axis=1)
val_out             = pd.read_csv(v_target_pth)

#print(train_in_video_fc.describe())
#print(train_in_video_pool.describe())
#print(train_in_audio_1.describe())
#print(train_in_audio_2.describe())
#print(train_in_all.describe())


# Choose all points for learning (ommitting 'ID' column)
cols = [str(x) for x in range(9216)]
t_X = train_in_all[cols]
v_X = val_in_all[cols]
#print(X)

# Choose output values
t_y = train_out['Label']
v_y = val_out['Label']
#print(y)

# Load model
classifier = RandomForestClassifier()
classifier.fit(t_X,t_y)
out = classifier.predict(v_X)

MAE = mean_absolute_error(v_y,out)
print(MAE)
correct = [index for index, value in enumerate(out) if v_y[index] == value]
wrong = [index for index, value in enumerate(out) if v_y[index] != value]

print(len(correct), len(wrong))

submission = pd.DataFrame({
    'ID':    val_out.ID,
    'Label': out
})
submission.to_csv("submission.csv",index=False)