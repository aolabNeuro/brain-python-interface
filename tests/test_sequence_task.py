from built_in_tasks.target_capture_task import ScreenTargetCapture
import matplotlib.pyplot as plt
import numpy as np

seq = ScreenTargetCapture.sequence_2D(nblocks=1, distance=5)
#seq = ScreenTargetCapture.centerout_2D_different_center(nblocks=100, distance=5, origin=(0,0,0))
#seq = list(seq)
for _ in range(8):
    idx, pos = next(seq)
    print(idx, idx.shape)
    print(pos, pos.shape)
    fig,ax=plt.subplots()
    ax.scatter(pos[0,0],pos[0,2], c='k', s=50, alpha = 0.5)
    ax.scatter(pos[1,0],pos[1,2], c='b', s=30, alpha = 0.5)
    ax.scatter(pos[2,0],pos[2,2], c='r', s=30, alpha = 0.5)
    ax.set(xlim=(-12,12),ylim=(-12,12))
    plt.show()

# number = np.array([1,2,3,4,5,6,7,8])
# itheta = 2*np.pi*number/8
# plt.plot(number, np.sin(itheta))
# plt.show()