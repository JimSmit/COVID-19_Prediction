
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import random
from functions import *
from classes import *

pred_window = 64
gap = 20
int_neg = 4
int_pos = 4
feature_window = 3
val_share = 0.4
test_share = 0.2
label_type = 'mortality'
model = 'RF'

print('RESULTS FOR MODEL:',model)
print('PRED W:',pred_window,'Hours')
print('FEATURE W:',feature_window, 'samples')

#%%
parchure = Parchure(inputs=inputs,encoders=encoders)
parchure.Prepare(random.randint(0, 10),val_share=val_share) # Train / val  / test split and normalization
parchure.Build_feature_vectors(pred_window,gap,int_neg,int_pos,feature_window) 
parchure.Balance(undersampling=True)
parchure.Train(model=model,balance=True)
auc,tn, fp, fn, tp = parchure.Evaluate()
