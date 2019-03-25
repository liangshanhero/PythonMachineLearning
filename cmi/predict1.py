import numpy as np
import sys

predictData = None

argt = sys.argv[1:]

for v in argt:
    if v.startswith("-predict="):
        tmpStr = v[len("-predict="):]
        print("tmpStr: %s" % tmpStr)
        predictData = np.fromstring(tmpStr, dtype=np.float32, sep=",")

print("predictData: %s" % predictData)

