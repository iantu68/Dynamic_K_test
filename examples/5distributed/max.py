import numpy as np

acc = np.load('acc.npy')
max_value = np.max(acc)
max_index = np.argmax(acc)


print(f"The maximum value is {max_value} at index {max_index}")
