import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "iter1.txt"
data = pd.read_table(filename)
data_mean, data_max, data_min = np.mean(data), np.max(data), np.min(data)

print("data_mean, data_max, data_min:", data_mean, data_max, data_min)
print(len(data))
print(data_min)
idx_end = int(3e3)
idx_start = int(50e+3)

# print(data[0:idx_end])

plt.plot(data[:])

plt.xlabel("Iteration")
plt.ylabel("MSD")
plt.show()

