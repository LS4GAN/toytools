import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/yhuang2/PROJs/LS4GAN/toytools/toytools/')
from transform import isMultiTrack, multitrack_preprocess

def get_data(fname):
    with np.load(fname) as f:
        data = f[f.files[0]]
    return data

if __name__ == "__main__":
    fname = "batch_00227_protodune-orig-5-1-U.npz"
    data = get_data(fname)
    print(isMultiTrack(data))

#     output = multitrack_preprocess(data)
#     fig, axes = plt.subplots(1, 2, figsize=(3 * 2, 3))
#     ax = axes[0]
#     ax.pcolormesh(data)
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     ax = axes[1]
#     ax.pcolormesh(output)
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     plt.savefig(f'{fname[:-4]}.png', dpi=200, bbox_inches='tight')
