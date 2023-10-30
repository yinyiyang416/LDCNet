import matplotlib.pyplot as plt
import torch
import numpy as np

#raed npy file for train,and plot them as picture
def plot_loss(save_path,npy_path):
    y_temp = np.load(npy_path,allow_pickle=True)
    y = list(y_temp)
    x = range(0,len(y))
    plt.plot(x, y, '.-')
    plt_title = 'pretrain'
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    npy_path = "PATH TO NPY"
    save_path = "PATH TO SAVE PICTURE"
    plot_loss(save_path,npy_path)

