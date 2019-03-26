import numpy as np
import pandas as pd
import random


fileData = pd.read_csv('dataq.csv', dtype=np.float32, header=None, converters={(2): lambda s: random.random()})

wholeData = fileData.as_matrix()

print(wholeData)
