import numpy as np
import matplotlib.pyplot as plt

def threshold_sheduler(state, target):
    return (target-state) * 0.1 + state


num_samples = 60000
batchsize = 256
batches_per_epoch = num_samples // batchsize
num_epochs = 25

num_batches = num_epochs = 25*batches_per_epoch
init_state = -100
target = 0.9
state = np.empty(num_batches+1)
state[0] = init_state
for i in range(num_batches):
    state[i+1] = threshold_sheduler(state[i], target)

epochs = np.arange(num_batches+1).astype(np.float32)/batches_per_epoch


plt.figure()
plt.plot(np.arange(num_batches+1), state)
plt.show()
