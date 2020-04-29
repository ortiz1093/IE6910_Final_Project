"""
<Assignment Name>
<DEPT CrsNum-SecNum>: <Course Name>
Mon Apr 27 16:16:01 2020
Joshua Ortiz
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


#%% Function definitions

# Use Lasso to prune unnecessary variables
def lassoPrune(X,y,alpha):
	lasso = Lasso(alpha=alpha,normalize=True,random_state=42,positive=True)
	lasso.fit(X,y)

	delIdx = np.argwhere(lasso.coef_==0)
	delCols = X.columns[delIdx].flatten()
	X = X.drop(columns = delCols)

	Data = pd.merge(X, y, left_index=True, right_index=True)
	Data.rename({'Mean':'MarketValue'},axis=1,inplace=True)

	return Data


# Create neural network
def createNN(Inputs, Outputs, Layers, Nodes, Activations):
	seed = 42
	np.random.seed(seed)
	tf.random.set_seed(seed)

	tf.keras.backend.clear_session()

	model = tf.keras.models.Sequential([
	    tf.keras.layers.Flatten(input_shape=(Inputs,)),
	    ])

	for i in range(Layers-1):

		if isinstance(Activations,str)==1:
			activation = Activations
		else:
			activation = Activations[i]

		if isinstance(Nodes,int)==1:
			nodes = Nodes
		else:
			nodes = Nodes[i]

		model.add(tf.keras.layers.Dense(nodes,activation=activation))

	model.add(tf.keras.layers.Dense(1, activation=activation))

	np.random.seed(seed)
	tf.random.set_seed(seed)
	model.compile(optimizer='adam',
              loss='hinge')

	return model

# Split data and standardize train/test data separately
def splitScaleData(Data,test_pct):
	split_idx = int(np.floor((1-test_pct) * len(Data)))
	X_train,X_test,y_train,y_test = Data.iloc[:split_idx,0:-1], \
						  Data.iloc[split_idx:,0:-1], \
						  Data.iloc[:split_idx,-1], \
						  Data.iloc[split_idx:,-1]

	y_train = y_train.to_numpy().reshape(-1,1)
	y_test = y_test.to_numpy().reshape(-1,1)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)
	y_train = scaler.fit_transform(y_train)
	y_test = scaler.fit_transform(y_test)

	return X_train,X_test,y_train,y_test,scaler

# Tabulate performance information
def performanceTable(y_test,y_pred,model,scaler):
	y_test = scaler.inverse_transform(y_test)
	y_pred = scaler.inverse_transform(y_pred)
	y_diff = 100*(y_pred-y_test)/y_test
	Performance = pd.DataFrame(
				{'Actual':y_test.flatten(),'Predicted':y_pred.flatten(), \
			       'Pct Diff':y_diff.flatten()},
			       index=np.arange(len(y_test))
				 )

	return Performance
#%% Import and format data
FS = pd.read_csv("FunctionStructureData.csv").set_index("ProductName")
AM = pd.read_csv("AssemblyModelData.csv").set_index("ProductName")
MV = pd.read_csv("MarketValueData.csv").set_index("ProductName")
AT = pd.read_csv("AssemblyTimeData.csv").set_index("ProductName")

FS_AM_Diffs = pd.DataFrame(columns=FS.columns)

for product in FS.index:
	FS_AM_Diffs.loc[product] = 100 * (FS.loc[product] - AM.loc[product]) \
						/ FS.loc[product]

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
AM.columns = newCols


#%% Perform variable selection and combine variables and targets
X = FS
y = MV.iloc[:,-1]

Data = lassoPrune(X,y,4)

numVars = Data.shape[1]-1
numSamp = Data.shape[0]


#%% Split data and scale

test_pct = 0.25

X_train,X_test,y_train,y_test,scaler = splitScaleData(Data,test_pct)

#%% Train and Evaluate model

print("\nTraining:")
model = createNN(numVars,numSamp,20,100,'elu')
model.fit(X_train,y_train,epochs=20,batch_size=2**2)

print("\nEvaluation:")
y_pred = model.predict(X_test)
model.evaluate(X_test,y_test)

Performance = performanceTable(y_test,y_pred,model,scaler)