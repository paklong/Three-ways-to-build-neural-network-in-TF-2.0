import tensorflow as tf
import numpy as np

## Data
SIZE = 200000
EPOCHS = 10
x1 = np.random.randint(0, 100, SIZE)
x2 = np.random.randint(0, 100, SIZE)
x = np.dstack((x1, x2))[0]
y = 3*(x1**(1/2)) + 2*(x2**2) #Sample non linear function that map x1, x2 to y


## tf.keras.Sequential
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Dense(32, input_shape=(2,) , activation='sigmoid'))
model1.add(tf.keras.layers.Dense(32, activation = 'relu'))
model1.add(tf.keras.layers.Dense(1))
model1.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MSE)
model1.fit(x=x, y=y, epochs=EPOCHS)


## tf.keras.Model
class Model2(tf.keras.Model):
	def __init__(self):
		super(Model2, self).__init__()
		self.hidden1 = tf.keras.layers.Dense(32, input_shape=(2,) ,activation='sigmoid')
		self.hidden2 = tf.keras.layers.Dense(32, activation='relu')
		self.out = tf.keras.layers.Dense(1)
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


## Testing
x1 = np.array([54, 1, 15, 97, 26])
x2 = np.array([97, 94, 1, 68, 41])
x = np.dstack((x1, x2))[0]
y = 3*(x1**(1/2)) + 2*(x2**2)
y1 = model1.predict(x)
y2 = model2.predict(x)

for i in range(5):
	print ("Input: ({}, {}), Answer: {:.1f}, Model1: {:.1f}, Model2: {:.1f}".format(x1[i], x2[i], y[i], y1[i][0], y2[i][0]))
	
"""
***performance may vary, increase sample or epochs can recude variance***
Input: (54, 97), Answer: 18840.0, Model1: 18834.4, Model2: 18844.1
Input: (1,  94), Answer: 17675.0, Model1: 17703.1, Model2: 17729.1
Input: (15,  1), Answer: 13.6,    Model1: 13.5,    Model2: 14.8
Input: (97, 68), Answer: 9277.5,  Model1: 9296.2,  Model2: 9276.6
Input: (26, 41), Answer: 3377.3,  Model1: 3384.9,  Model2: 3377.8
"""

