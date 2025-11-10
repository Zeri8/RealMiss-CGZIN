import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def save_vis(output, img, gt, sid, path='results/vis', sl=64):
    os.makedirs(path, exist_ok=True)
    pred = output['seg'].argmax(1)[0].cpu().numpy()
    prob = output['mod_prob'][0].cpu().numpy()
    colors = np.array([[0,0,0], [255,0,0], [0,255,0], [0,0,255]]) / 255

    def colorize(s):
        return np.stack([colors[s[i]] for i in range(s.shape[0])]) if s.ndim == 3 else colors[s]

    img_sl = (img[0,:,:,sl].cpu().numpy() - img.min()) / (img.ptp() + 1e-8)

    fig, ax = plt.subplots(2, 4, figsize=(20,10))
    ax[0,0].imshow(img_sl, cmap='gray'); ax[0,0].set_title('Input')
    ax[0,1].imshow(colorize(gt[:,:,sl])); ax[0,1].set_title('GT')
    ax[0,2].imshow(colorize(pred[:,:,sl])); ax[0,2].set_title('Pred')
    ax[1,0].bar(['T1','T1ce','T2','FLAIR'], prob); ax[1,0].set_title('Contribution')
    ax[1,1].text(0.1, 0.5, output['suggestion'], fontsize=16, color='red' if '补扫' in output['suggestion'] else 'green')
    [a.axis('off') for a in ax.ravel()]
    plt.savefig(f"{path}/{sid}_s{sl}.png", dpi=300, bbox_inches='tight')
    plt.close()