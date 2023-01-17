import sys

import librosa
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


afpid_dataset_dir = "../acoustic_footstep/AFPID_FE1"

rk_train_file = 'AFPID_FE1_train.csv'
rk_test_file = 'AFPID_FE1_test.csv'

# load AFPID_RK data file
rk_train_dataframe = pd.read_csv(os.path.join(afpid_dataset_dir, rk_train_file))
rk_test_dataframe = pd.read_csv(os.path.join(afpid_dataset_dir, rk_test_file))

# exclude the first column
rk_train_dataframe = rk_train_dataframe.iloc[:, 1:]
rk_test_dataframe = rk_test_dataframe.iloc[:, 1:]

# concat train and test
rk_total = pd.concat([rk_train_dataframe, rk_test_dataframe], axis=0)
rk_label = rk_total['person_label']

# split with sklearn function
rd_train_x, rd_test_x, rd_train_y, rd_test_y = train_test_split(rk_total, rk_label, test_size=1/3, random_state=2)
# print(pd.value_counts(rd_train_y))
# print(pd.value_counts(rd_test_y))

# and the final labels respec
rd_train_x['person_label'] = rd_train_y
rd_test_x['person_label'] = rd_test_y

# save AFPID_RD data file
rd_train_x.to_csv(os.path.join(afpid_dataset_dir, 'AFPID_FE1_train_rd.csv'))
rd_test_x.to_csv(os.path.join(afpid_dataset_dir, 'AFPID_FE1_test_rd.csv'))
