import tensorflow.contrib.keras as k
import numpy as np


model = k.models.Sequential()

model.add(k.layers.Dense(8, input_dim=3, activation='tanh'))

model.add(k.layers.Dense(1, input_dim=3, activation='linear'))

model.compile(loss='mean_squared_error', optimizer="RMSProp", metrics=['accuracy'])

model.fit([[1, 1, 1], [1, 0, 1], [1, 2, 3]], [2, 1, 3], epochs=1000, batch_size=1, verbose=2)

xTestData = np.array([[1, 2, 2], [2, 3, 3]], dtype=np.float32)

resultAry = model.predict(xTestData)
print(resultAry)

