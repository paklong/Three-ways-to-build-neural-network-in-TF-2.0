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

## tf.keras.Sequential + fit
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

## tf.keras.Sequential without fit but tf.GradientTape()
model3 = tf.keras.Sequential()
model3.add(tf.keras.layers.Dense(64, input_shape=(2,) , activation='sigmoid'))
model3.add(tf.keras.layers.Dense(128, activation = 'relu'))
model3.add(tf.keras.layers.Dense(2))
# model3.build()
opt =tf.keras.optimizers.Adam(0.001)
mse =tf.keras.losses.MSE
batch = 100
for i in range(10):
	for j in range(0, SIZE, batch):
		with tf.GradientTape() as tape:
			y_pre = model3(x[j:j+batch])
			loss = mse(y[j:j+batch], y_pre)
		grads = tape.gradient(loss, model3.trainable_variables)	
		processed_grads = [g for g in grads]
		grads_and_vars = zip(processed_grads, model3.trainable_variables)
		opt.apply_gradients(grads_and_vars)
	print ('Model3: ', i, ' : ', np.mean(loss.numpy()))



# Testing
x1 = np.array([100, 9, 62, 79, 94, 91, 71, 41])
x2 = np.array([65, 39, 40, 44, 77, 42, 36, 74])
x = np.dstack((x1, x2))[0]
y1 = 3*(x1**(1/2)) + 2*(x2**2)
y2 = x1 + x2 + 80
y = np.dstack((y1, y2))[0]
yM1 = model1.predict(x)
yM2 = model2.predict(x)
yM3 = model3.predict(x)

for i in range(5):
	print ('''Input: ({}, {}), Answer: ({:.0f}, {:.0f}), \
Model1: ({:.0f}, {:.0f}), \
Model2: ({:.0f}, {:.0f}), \
Model3: ({:.0f}, {:.0f})'''.format(x1[i], x2[i], y1[i], y2[i], yM1[i][0], yM1[i][1], yM2[i][0], yM2[i][1], yM3[i][0], yM3[i][1]))

'''
Input: (100, 65), Answer: (8480, 245), Model1: (8473, 243), Model2: (8491, 247), Model3: (8477, 236)
Input: (9, 39), Answer: (3051, 128), Model1: (3057, 126), Model2: (3037, 129), Model3: (3050, 127)
Input: (62, 40), Answer: (3224, 182), Model1: (3229, 182), Model2: (3210, 180), Model3: (3217, 186)
Input: (79, 44), Answer: (3899, 203), Model1: (3907, 208), Model2: (3888, 203), Model3: (3897, 207)
Input: (94, 77), Answer: (11887, 251), Model1: (11848, 253), Model2: (11893, 255), Model3: (11876, 242)
'''
