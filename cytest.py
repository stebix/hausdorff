import dhd
import numpy as np

a = np.arange(0, 9, dtype=np.float64).reshape(3, 3)
b = np.arange(0, 9, dtype=np.float64).reshape(3, 3)


result = dhd.mysum(a, b)
print(result)