#!/usr/bin/env python3
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier

# Get home path
home = os.environ.get('HOME', '/root')

# Define base location of datasets
pth_base = os.path.join(home, ".kaggle/competitions/ai-hack-2018-minsk-stc")

# Dataset paths
t_video_emb_fc_pth = os.path.join(pth_base, "train_video_emb_fc.csv")
t_video_emb_pool_pth = os.path.join(pth_base, "train_video_emb_pool.csv")
t_audio_feat_1_pth = os.path.join(pth_base, "train_audio_feat_1.csv")
t_audio_feat_2_pth = os.path.join(pth_base, "train_audio_feat_2.csv")
t_target_pth = os.path.join(pth_base, "train_target.csv")

v_video_emb_fc_pth = os.path.join(pth_base, "val_video_emb_fc.csv")
v_video_emb_pool_pth = os.path.join(pth_base, "val_video_emb_pool.csv")
v_audio_feat_1_pth = os.path.join(pth_base, "val_audio_feat_1.csv")
v_audio_feat_2_pth = os.path.join(pth_base, "val_audio_feat_2.csv")
v_target_pth = os.path.join(pth_base, "val_target.csv")

tst_video_emb_fc_pth = os.path.join(pth_base, "test_video_emb_fc.csv")
tst_video_emb_pool_pth = os.path.join(pth_base, "test_video_emb_pool.csv")
tst_audio_feat_1_pth = os.path.join(pth_base, "test_audio_feat_1.csv")
tst_audio_feat_2_pth = os.path.join(pth_base, "test_audio_feat_2.csv")

# Load data
train_in_video_fc = pd.read_csv(t_video_emb_fc_pth)
train_in_video_pool = pd.read_csv(t_video_emb_pool_pth)
train_in_audio_1 = pd.read_csv(t_audio_feat_1_pth)
train_in_audio_2 = pd.read_csv(t_audio_feat_2_pth)
train_sets = [train_in_audio_1, train_in_audio_2,
              train_in_video_fc, train_in_video_pool]
train_in_all = pd.concat(train_sets, axis=1)
train_out = pd.read_csv(t_target_pth)

val_in_video_fc = pd.read_csv(v_video_emb_fc_pth)
val_in_video_pool = pd.read_csv(v_video_emb_pool_pth)
val_in_audio_1 = pd.read_csv(v_audio_feat_1_pth)
val_in_audio_2 = pd.read_csv(v_audio_feat_2_pth)
val_sets = [val_in_audio_1, val_in_audio_2, val_in_video_fc, val_in_video_pool]
val_in_all = pd.concat(val_sets, axis=1)
val_out = pd.read_csv(v_target_pth)

tst_in_video_fc = pd.read_csv(tst_video_emb_fc_pth)
tst_in_video_pool = pd.read_csv(tst_video_emb_pool_pth)
tst_in_audio_1 = pd.read_csv(tst_audio_feat_1_pth)
tst_in_audio_2 = pd.read_csv(tst_audio_feat_2_pth)
tst_sets = [tst_in_audio_1, tst_in_audio_2, tst_in_video_fc, tst_in_video_pool]
tst_in_all = pd.concat(tst_sets, axis=1)

# print(train_in_video_fc.describe())
# print(train_in_video_pool.describe())
# print(train_in_audio_1.describe())
# print(train_in_audio_2.describe())
# print(train_in_all.describe())


# Choose all points for learning (ommitting 'ID' column)
cols = [str(x) for x in range(9216)]
t_X = train_in_all[cols]
v_X = val_in_all[cols]
tst_X = tst_in_all[cols]
# print(X)

# Rename columns
t_X.columns = [ str(x) for x in range(14091) ]
v_X.columns = [ str(x) for x in range(14091) ]
tst_X.columns = [ str(x) for x in range(14091) ]

# Choose output values
t_y = train_out['Label']
v_y = val_out['Label']
# print(y)

# Load model
classifier = XGBClassifier()
classifier.fit(t_X, t_y)
v_out = classifier.predict(v_X)

correct = [index for index, value in enumerate(v_out) if v_y[index] == value]
wrong = [index for index, value in enumerate(v_out) if v_y[index] != value]

print(len(correct), len(wrong))

tst_out = classifier.predict(tst_X)

submission = pd.DataFrame({
    'ID': val_out.ID,
    'Label': tst_out
})
submission.to_csv("submission.csv", index=False)


correct = 0
best = 0
for i in range(100,1000):
    classifier = XGBClassifier(n_estimators=i)
    classifier.fit(t_X, t_y)
    v_out = classifier.predict(v_X)
    c = [index for index, value in enumerate(v_out) if v_y[index] == value]
    if c > correct:
        correct = c
        best = i
        print(best,correct)

print(best,correct)
