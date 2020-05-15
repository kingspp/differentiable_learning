import glob
import matplotlib.pyplot as plt
import numpy as np


def visualize_input_saliency(data_dir, dataset_name, save_path=None):
    data_dict = {}
    for f in sorted(glob.glob(data_dir + '/*.npy')):
        d = np.load(f)
        ep = f.split('_')[-1].split('.')[0]
        ly = f.split('/')[-1].split('_')[0]
        if ly not in data_dict:
            data_dict[ly] = {}
            data_dict[ly][ep] = d
        else:
            data_dict[ly][ep] = d

    for k, v in data_dict.items():
        srt_keys = list(v.keys())
        srt_keys.sort(key=int)
        data_dict[k] = {l: data_dict[k][l] for l in srt_keys}

    rows = len(data_dict[list(data_dict.keys())[0]])
    cols = len(data_dict)
    fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=(2 * cols, 2 * rows))

    for c, (k, v) in enumerate(data_dict.items()):
        for r, (ep, d) in enumerate(v.items()):
            ax[r][c].imshow(d.reshape(28, 28), cmap='magma')
            ax[r][0].set_ylabel(f'epoch {ep}')
            ax[0][c].set_title(f'Layer {c}')

    plt.tight_layout()
    fig.suptitle(dataset_name, y=1.02)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
