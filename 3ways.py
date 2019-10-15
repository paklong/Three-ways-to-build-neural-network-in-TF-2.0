import tensorflow as tf
import numpy as np

## Data
SIZE = 200000
EPOCHS = 10
x1 = np.random.randint(0, 100, SIZE)
x2 = np.random.randint(0, 100, SIZE)
x = np.dstack((x1, x2))[0]
y1 = 3*(x1**(1/2)) + 2*(x2**2)
y2 = x1 + x2 + 80
y = np.dstack((y1, y2))[0]

## tf.keras.Sequential
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Dense(64, input_shape=(2,) , activation='sigmoid'))
model1.add(tf.keras.layers.Dense(128, activation = 'relu'))
model1.add(tf.keras.layers.Dense(2))
model1.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MSE)
model1.fit(x=x, y=y, epochs=EPOCHS)

## tf.keras.Model
class Model2(tf.keras.Model):
	def __init__(self):
		super(Model2, self).__init__()
		self.hidden1 = tf.keras.layers.Dense(64, input_shape=(2,) ,activation='sigmoid')
		self.hidden2 = tf.keras.layers.Dense(128, activation='relu')
		self.out = tf.keras.layers.Dense(2)
		self.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MSE)

	def call(self, inputs):
		x = self.hidden1(inputs)
		x = self.hidden2(x)
		x = self.out(x)
		return x

model2 = Model2()
model2.fit(x=x, y=y, epochs=EPOCHS)

## Custome layers
# https://www.tensorflow.org/guide/keras/custom_layers_and_models


# Testing
x1 = np.array([100, 9, 62, 79, 94, 91, 71, 41])
x2 = np.array([65, 39, 40, 44, 77, 42, 36, 74])
x = np.dstack((x1, x2))[0]
y1 = 3*(x1**(1/2)) + 2*(x2**2)
y2 = x1 + x2 + 80
y = np.dstack((y1, y2))[0]
yM1 = model1.predict(x)
yM2 = model2.predict(x)

for i in range(5):
	print ("Input: ({}, {}), Answer: ({:.0f}, {:.0f}), \
Model1: ({:.0f}, {:.0f}), Model2: ({:.0f}, {:.0f})".format(x1[i], x2[i], y1[i], y2[i], yM1[i][0], yM1[i][1], yM2[i][0], yM2[i][1]))

'''
Input: (100, 65), Answer: (8480, 245), Model1: (8476, 243), Model2: (8468, 244)
Input: (9, 39), Answer: (3051, 128), Model1: (3045, 122), Model2: (3064, 134)
Input: (62, 40), Answer: (3224, 182), Model1: (3214, 182), Model2: (3212, 181)
Input: (79, 44), Answer: (3899, 203), Model1: (3891, 207), Model2: (3884, 204)
Input: (94, 77), Answer: (11887, 251), Model1: (11864, 254), Model2: (11864, 249)
'''



