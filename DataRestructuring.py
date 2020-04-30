"""
<Assignment Name>
<DEPT CrsNum-SecNum>: <Course Name>
Mon Apr 27 16:16:01 2020
Joshua Ortiz
"""
from time import time
import pandas as pd
import numpy as np
from numpy.random import randint
from numpy.random import rand
from sklearn.linear_model import Lasso
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

prg_t0 = time()
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

def randRg (low,high):
	return (high - low) * rand() + low

#%% Import and format data
FS = pd.read_csv("FunctionStructureData.csv").set_index("ProductName")
MV = pd.read_csv("MarketValueData.csv").set_index("ProductName")

# Shortening col names for easier viewing
newCols = ['Sz_Dim_Els', 'Sz_Dim_Rel', 'Sz_Con_DOF',
      'Sz_Con', 'ItrCn_Pth_Sum',
       'ItrCn_Pth_Max', 'ItrCn_Pth_Mn',
       'ItrCn_Pth_Dns', 'ItrCn_Flw_Sum',
       'ItrCn_Flw_Max', 'ItrCn_Flw_Mn',
       'ItrCn_Flw_Dns', 'Ctr_Btw_Sum',
       'Ctr_Btw_Max', 'Ctr_Btw_Mn',
       'Ctr_Btw_Dns',
       'Ctr_Clst_Sum',
       'Ctr_Clst_Max',
       'Ctr_Clst_Mn',
       'Ctr_Clst_Dns',
       'Dcp_AmS', 'Dcp_CrIn_Sum',
       'Dcp_CrIn_Max', 'Dcp_CrIn_Mn',
       'Dcp_CrIn_Dns', 'Dcp_CrO_Sum',
       'Dcp_CrO_Max', 'Dcp_CrO_Mn',
       'Dcp_CrO_Dns']

FS.columns = newCols

Data = pd.merge(FS, MV.iloc[:,-1], left_index=True, right_index=True)

#%% Add random small perturbations as new data points

discreteCols = []
continuousCols = []

for col in Data.columns:
	discSum = sum(np.mod(Data[col],1)**2)
	discMean = np.mean(np.mod(Data[col],1)**2)

	if discSum==0 and discMean==0:
		discreteCols.append(col)
	else:
		continuousCols.append(col)

Data_pert = Data.copy()

for i in range(10):
	addData = Data.copy()

	for col in discreteCols:
		addData['Sz_Con'].apply(lambda x: x + randint(-1,2))

	for col in continuousCols:
		addData['Sz_Con'].apply(lambda x: x + randint(-1,2)*rand())

	Data_pert = Data_pert.append(addData, ignore_index=True)
