import numpy as np
import matplotlib.pyplot as plt
import cv2


def segmentation(img0,thresh=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
    plt.subplot(121)
    img1 = img0>thresh
    plt.imshow(img1, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(fraction=0.035, pad=0.04,ticks=[0, 1])
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.title("Demo image")
    plt.subplot(122)
    plt.hist(img0.flatten()*(img1.flatten()), bins = np.arange(1,256,1),color='grey',alpha = 1)
    plt.hist(img0.flatten()*(1-img1.flatten()), bins = np.arange(1,256,1),color='grey', alpha = .2)
    plt.axvline(x=thresh+1,color = 'r', linestyle = '--', linewidth = 1)
    plt.xlabel("Intensity, $i$ [-]")
    plt.ylabel("Count [-]")
    plt.ylim(0,400)
    plt.rcParams.update({'font.size': 12})
    plt.show()

