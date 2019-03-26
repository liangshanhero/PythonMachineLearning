import tensorflow.contrib.keras as k
import numpy as np


model = k.models.Sequential()

model.add(k.layers.Dense(3, input_dim=3, activation='linear'))

model.add(k.layers.Dense(1, input_dim=3, activation='linear'))

model.compile(loss='mean_squared_error', optimizer="RMSProp", metrics=['accuracy'])

model.fit([[90, 80, 70], [98, 95, 87]], [85, 96], epochs=10000, batch_size=1, verbose=2)

xTestData = np.array([[80, 80, 80], [99, 98, 97]], dtype=np.float32)

resultAry = model.predict(xTestData)
print(resultAry)

