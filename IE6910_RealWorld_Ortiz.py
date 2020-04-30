"""
<Assignment Name>
<DEPT CrsNum-SecNum>: <Course Name>
Thu Apr 30 02:12:18 2020
Joshua Ortiz
"""

from time import time
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy.random import rand

prg_t0 = time()
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

#%% Function definitions

# Use Lasso to prune unnecessary variables
def lassoPrune(X,y,alpha):
	lasso = Lasso(alpha=alpha,normalize=False,random_state=42)
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
	model.compile(optimizer='nadam',
		        loss='mape',
			  metrics=['mape'])

	return model

# Function to scale arrays to between 1 and 1
def arrayScaler(arr):
	arrMin = np.min(arr)
	arrMax = np.max(arr)
	interval = arrMax - arrMin

	return [(val-arrMin)/(interval) for val in arr]

# Run entire optimization routine to choose best model for data
def OptimizeNN(Data):

	cols = Data.columns
	scaler = MinMaxScaler()
	Data = pd.DataFrame(scaler.fit_transform(Data))
	Data.columns = cols

	#################### Lasso Variable Selection ##########################
	test_pct = 0.2
	X,y = Data.iloc[:,:-1],Data.iloc[:,-1]
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_pct)

	alphas = np.arange(0.01,0.15,step=0.01)
	scores = []

	for alpha in alphas:
		lasso = Lasso(alpha=alpha,normalize=True,random_state=42)
		lasso.fit(X_train,y_train)

		scores.append(lasso.score(X_test,y_test))

	best_idx = np.argmax(scores)
	best_alpha = alphas[best_idx]

	fig,ax = plt.subplots()
	ax.semilogx(alphas,scores,basex=2)
	plt.show()

	X = FS
	y = MV.iloc[:,-1]

	Data = lassoPrune(X,y,best_alpha)

	#################### Architecture Selection ##########################
	X,y = Data.iloc[:,:-1],Data.iloc[:,-1]

	numVars = X.shape[1]
	numSamp = y.shape[0]

	num_layers = np.append(np.random.randint(2,50,49),20)
	num_nodes = np.append(np.random.randint(10,100,49),100)

	arch_perf = []

	for i in range(len(num_layers)):

		model = createNN(numVars,numSamp,int(num_layers[i]),
				     int(num_nodes[i]),'linear')
		model.fit(X,y,epochs=20,batch_size=2**2,verbose=0)

		arch_perf.append(model.history.history['mape'][-1])

	best_arch_idx = np.argmin(arch_perf)
	best_arch = (num_layers[best_arch_idx],num_nodes[best_arch_idx])


	# Plot architecture performance
	arch_sz = [i for i in arch_perf]
	#arch_sz = [(i**3)*3000 for i in arch_perf]
	fig,ax = plt.subplots()
	plt.xticks(np.arange(2, max(num_layers)+1, 2.0))
	color = [1/i for i in arch_perf]
	ax.scatter(num_layers,num_nodes,s=arch_sz,c=color,cmap='viridis')
	ax.set_title("Model Architecutres \n\n Annotation: Accuracy, (Layers x Nodes)")
	ax.set_xlabel("Number of Layers")
	ax.set_ylabel("Number of Nodes per Layer")


	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()

	for i,perf in enumerate(arch_perf):
	#	offset = perf*0.55
		offset = perf*0.001
		ptX = num_layers[i]+offset
		ptY = num_nodes[i]+offset
		txt = str(np.round(perf,3)) + ", (" + str(num_layers[i]) + " x " \
			+ str(num_nodes[i]) + ")"
		ax.annotate(txt,(ptX,ptY),size='medium')

	#################### Activation Selection ##########################

	# Specifiy certain activation functions to test
	activations = ['relu','elu','tanh','sigmoid', 'linear', 'exponential']

	# Initialize array for tracking the performance of each activation
	act_perf = []

	# Everything the same as for architecture except looping over activations

	for activation in activations:
		model = createNN(numVars,numSamp,int(best_arch[0]),
				     int(best_arch[1]),activation)
		model.fit(X,y,epochs=20,batch_size=2**2,verbose=0)

		act_perf.append(model.history.history['mape'][-1])

	best_activation = np.nanargmin(act_perf)
	best_act = activations[best_activation]

	fig,ax = plt.subplots()
	ax.bar(activations,act_perf)
	ax.set_ylabel('Accuracy Score',size='x-large')

	ax.set_title('Accuracy by Activation Function',size='x-large')
	ax.set_xticks(np.arange(len(activations)))
	ax.set_xticklabels(activations, rotation = 45,size='x-large')
	ax.set_yticks(np.arange(0, 1.1, 0.1))

	for index, value in enumerate(act_perf):
		if not np.isnan(value):
			plt.text(index-0.2, value+0.02, str(value),size='large',
			         weight='bold')

	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()

	#################### Batch Size Selection ##########################
	# Test batch sizes in powers of 2, (e.g. 2, 4, 8, 16, etc.)
	batch_sizes = 2**np.array(range(2,9))

	# Pretty much same process again
	batch_perf = []
	batch_times = []

	for batch_size in batch_sizes:
		model = createNN(numVars,numSamp,int(best_arch[0]),
				     int(best_arch[1]),best_act)
		m0 = time()
		model.fit(X,y,epochs=20,batch_size=batch_size,verbose=0)
		m_f = time() - m0

		batch_perf.append(model.history.history['mape'][-1])
		batch_times.append(m_f)

	scaled_batch_perf = arrayScaler(batch_perf)
	scaled_batch_times = arrayScaler(batch_times)
	batch_diff = []
	for i in range(len(batch_sizes)):
		batch_diff.append(scaled_batch_perf[i] - scaled_batch_times[i])
	best_batch = np.argmax(batch_diff)
	batch_sz = batch_sizes[best_batch]


	# Plotting batch size sensitivity
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.semilogx(batch_sizes, batch_perf, basex=2)
	ax1.set_title('Batch Size Performance',size='x-large')
	ax1.set_xlabel('Batch Size',size='x-large')
	ax1.set_xticks(batch_sizes)
	ax1.set_ylabel('Accuracy Score', color=color,size='x-large')
	ax1.plot(batch_sizes, batch_perf, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('Training Time', color=color,size='x-large')
	ax2.plot(batch_sizes, batch_times, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # prevents right y-label from being clipped

	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()

	#################### Epoch Quantity Selection ##########################

	# Same process again
	epochs = 2**np.array(range(2,9))
	epoch_perf = []
	epoch_times = []

	for epoch in epochs:
		model = createNN(numVars,numSamp,int(best_arch[0]),
				     int(best_arch[1]),best_act)
		m0 = time()
		model.fit(X,y,epochs=epoch,batch_size=batch_sz,verbose=0)
		m_f = time() - m0

		epoch_perf.append(model.history.history['mape'][-1])
		epoch_times.append(m_f)

	# This time, I look for where the performance begins to level off as the number
	# of epochs increases. Though, in hindsight this is not the correct method, it
	# worked in this case.
	epoch_perf_dydx = [(epoch_perf[i+1]-epoch_perf[i])/ \
				 (np.log2(epochs[i+1])-np.log2(epochs[i])) \
				 for i in range(len(epochs)-1)]
	best_epoch = np.argmax(np.abs(np.diff(epoch_perf_dydx))) + 1
	epoch_sz = epochs[best_epoch]

	# Epoch sensitivity plotting
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.semilogx(epochs, epoch_perf, basex=2)
	ax1.set_title('Epoch Performance',size='xx-large')
	ax1.set_xlabel('Epochs',size='x-large')
	ax1.set_xticks(batch_sizes)
	ax1.set_ylabel('Accuracy Score', color=color,size='x-large')
	ax1.plot(epochs, epoch_perf, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('Training Time', color=color,size='x-large')  # we already handled the x-label with ax1
	ax2.plot(epochs, epoch_times, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()

	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()

	#################### 5-Fold Validation ##########################

	model = createNN(numVars,numSamp,int(best_arch[0]),
				     int(best_arch[1]),best_act)

	MAPEs = []

	K = 5
	sz = int(np.floor(len(Data)/K))
	for fold in range(K):
		TestRows = Data.index[sz*fold+np.arange(sz)]
		print(TestRows)
		foldTrain = Data.drop(TestRows)
		foldTest = Data.loc[TestRows]
		print(foldTrain)
		print(foldTest)

		foldX_train,foldY_train = foldTrain.iloc[:,:-1],foldTrain.iloc[:,-1]
		foldX_test,foldY_test = foldTest.iloc[:,:-1],foldTest.iloc[:,-1]

		model.fit(foldX_train,foldY_train,epochs=epoch_sz,batch_size=batch_sz,
		          verbose=0)
		foldLoss,foldMape = model.evaluate(foldX_test,foldY_test,verbose=0)
		MAPEs.append(foldMape)

	MAPEscore = np.mean(np.array(MAPEs))
	print("Average mean absolute percent error from K-fold cross val: " \
		+ str(MAPEscore))

	SampleValData = Data.sample(5)
	valX,valY = SampleValData.iloc[:,:-1],SampleValData.iloc[:,-1]
	predY = model.predict(valX)


	y_diff = 100*(predY.flatten()-valY)/valY
#	Performance = pd.DataFrame(
#				{'Actual':valY.to_numpy(),'Predicted': predY.flatten(), \
#			       'Pct Diff':y_diff.to_numpy()},
#			       index=np.arange(len(valY))
#				 )

	prg_tf = time() - prg_t0
	print("Program ran in %0.3gs" % prg_tf)

	return valY.to_numpy(),predY.flatten(),y_diff.to_numpy()

#%% Import and format data
path = "D:/Users/Josh/Documents/School/Spring 2020/IE6910 Python Machine Learning/Project/4 Real World/"
FS = pd.read_csv(path + "FunctionStructureData.csv").set_index("ProductName")
MV = pd.read_csv(path + "MarketValueData.csv").set_index("ProductName")

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


#%% Check sensitivity and choose best Lasso alpha for variable pruning

Data = pd.merge(FS, MV.iloc[:,-1], left_index=True, right_index=True)


#%% No augmentation

valY,predY,y_diff = OptimizeNN(Data)
Performance = pd.DataFrame(
				{'Actual':valY,'Predicted': predY, \
			       'Pct Diff':y_diff},
			       index=np.arange(len(valY))
				 )


#%% Random perturbation oversampling

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

#valY,pertY,pertDiff = OptimizeNN(Data)
#Performance['Pert Pred'] = pertY
#Performance['Pert Diff'] = pertDiff

valY,predY,y_diff = OptimizeNN(Data)
Performance = pd.DataFrame(
				{'Actual':valY,'Predicted': predY, \
			       'Pct Diff':y_diff},
			       index=np.arange(len(valY))
				 )

#%% SMOTE Oversampling