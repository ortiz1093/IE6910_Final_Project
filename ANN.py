"""
Real World Data
IE 6910: Python Machine Learning Applications in IE
Fri Apr 24 15:13:29 2020
Joshua Ortiz
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

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
              loss='mse')

	return model

Data = pd.read_csv("ComplexityData_Value.csv").set_index("Column1")

for i in range(5):
	Data = Data.append(Data.sample(frac=1))
Data = Data.to_numpy()

test_pct = 0.1
split_idx = int(np.floor((1-test_pct) * len(Data)))
X_train,X_test,y_train,y_test = Data[:split_idx,0:30],Data[split_idx:,0:30], \
					  Data[:split_idx,-1],Data[split_idx:,-1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test = scaler.fit_transform(y_test.reshape(-1,1))

layers = np.random.randint(1,3)
nodes = np.random.randint(1,15)

print("\nTraining:")
model = createNN(Data.shape[1],Data.shape[0],20,100,'elu')
model.fit(X_train,y_train,epochs=20,batch_size=2**5)

print("\nEvaluation:")
y_pred = model.predict(X_test)
model.evaluate(X_test,y_test)

y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)
y_diff = 100*(y_pred-y_test)/y_test
Performance = pd.DataFrame(
			{'Actual':y_test.flatten(),'Predicted':y_pred.flatten(), \
		       'Pct Diff':y_diff.flatten()},
		       index=np.arange(len(y_test))
			 )