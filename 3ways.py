import tensorflow as tf
import numpy as np

## Data
SIZE = 100000
EPOCHS = 10
x1 = np.random.randint(0, 100, SIZE)
x2 = np.random.randint(0, 100, SIZE)
x = np.dstack((x1, x2))[0]
y = 3*(x1**(1/2)) + 2*(x2**2)


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


# Testing
x1 = np.array([54, 1, 15, 97, 26])
x2 = np.array([97, 94, 1, 68, 41])
x = np.dstack((x1, x2))[0]
y = 3*(x1**(1/2)) + 2*(x2**2)
y1 = model1.predict(x)
y2 = model2.predict(x)

for i in range(5):
	print ("Input: ({}, {}), Answer: {:.1f}, Model1: {:.1f}, Model2: {:.1f}".format(x1[i], x2[i], y[i], y1[i][0], y2[i][0]))

