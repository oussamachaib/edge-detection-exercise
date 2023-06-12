import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sci
from skimage.restoration import denoise_tv_bregman as edge_preserving_filter

def filter(img_raw):
    img = cv2.medianBlur(img_raw,13)
    img = edge_preserving_filter(img,weight = 1, max_num_iter=10)
    img = 255*(img/np.max(img))
    img = img.astype(np.uint8)
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 3))
    plt.subplot(121)
    plt.imshow(img_raw, cmap='RdBu_r', vmin=0, vmax=255)
    plt.title("Raw image")
    plt.colorbar(fraction=0.035, pad=0.04,ticks=[0, 255])
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    
    plt.subplot(122)
    plt.imshow(img, cmap='RdBu_r', vmin=0, vmax=255)
    plt.title("Filtered image")
    plt.colorbar(fraction=0.035, pad=0.04,ticks=[0, 255])
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.show()
    return img
    

def filt_grad(img, sigma = 2, flag = 0):
    # Input parameters
    width = int(np.ceil(4*sigma))

    # Creating Gaussian kernel
    H = np.zeros(width)
    for i in range(0,int(width)):
        x = i - ((width-1)/2)
        H[i] = np.exp(-(x**2)/(2*sigma**2))

    H = H*(1/(np.sqrt(2*np.pi)*sigma))
    H = H/np.sum(H) # ensuring the kernel sums up to zero

    # Creating 1D DoG kernel
    DoG = np.gradient(H)
    DoG[np.sign(DoG) == -1] = DoG[np.sign(DoG) == -1]/-np.sum(DoG[np.sign(DoG) == -1]) # ensuring the kernel sums up to zero
    DoG[np.sign(DoG) == +1] = DoG[np.sign(DoG) == +1]/np.sum(DoG[np.sign(DoG) == +1]) # ensuring the kernel sums up to zero

    # Computing gradient via convolution
    # Reshaping the 1D filter into a 2D matrix with a single column
    DoGx = np.reshape(DoG, (1, -1))
    DoGy = np.reshape(DoG, (-1, 1))
    # Performing 1D convolution along rows
    dx = sci.signal.convolve2d(img, DoGx, mode='same', boundary= "symm") # x component
    dy = -sci.signal.convolve2d(img, DoGy, mode='same', boundary= "symm") # y component
    D = np.sqrt(dx**2+dy**2)  # 2D gradient norm
    D = D/np.max(D) # Normalized 2D gradient
    
    # Angle with respect to the axis of the burner in degrees
    theta = np.arctan2(dy,dx)
    theta = np.rad2deg(theta)
    #theta[theta < 0] += 180   
    
    if flag:
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 3))
        # plt.subplot(133)
        # plt.plot(np.linspace(-len(DoG)/2,len(DoG)/2, num = len(DoG)),H, label = "Gaussian")
        # plt.plot(np.linspace(-len(DoG)/2,len(DoG)/2, num = len(DoG)),DoG, label = "DoG")
        # plt.xlabel("x [px]")
        # plt.ylabel("Magnitude [-]")
        # plt.xlim(-25,25)
        # plt.ylim(-.4,.4)
        # plt.grid()
        # plt.legend()
        # plt.title("1D kernels")

        plt.subplot(121)
        d = width
        cv2.rectangle(D, (20, 20), (20+d, 20+d), np.array([1,1,1])/2, 1)
        plt.imshow(D,cmap = "turbo",vmin = 0, vmax = 1)
        plt.xlabel("x [px]")
        plt.ylabel("y [px]")
        plt.colorbar(fraction=0.035, pad=0.04,ticks=[0, 1])
        plt.title("2D Gradient")
        
        plt.subplot(122)
        plt.imshow(theta,cmap = "turbo",vmin = -180, vmax = 180)
        plt.xlabel("x [px]")
        plt.ylabel("y [px]")
        plt.rcParams.update({'font.size': 12})
        plt.colorbar(fraction=0.035, pad=0.04,ticks=[-180, 180])
        plt.title("Gradient angle [deg]")
        plt.show()
    else:
        return D, theta


def nms(D, theta):
    # Defining max range
    y_max, x_max = D.shape
    
    # Defining final thin image
    T = np.ones(D.shape)
    T[0:,0] = 0
    T[0:,-1] = 0
    T[0,0:] = 0
    T[-1,0:] = 0
    
    mask = np.zeros((3,3))
    
    for j in range(1,y_max-1):
        for i in range(1,x_max-1):
            mask = D[j-1:j+2,i-1:i+2]
            if 0<=theta[j,i]<22.5 or -180<theta[j,i]<=-157.5:
                if mask[1,0]>D[j,i] or mask[1,2]>D[j,i]:
                    T[j,i] = 0
            elif 22.5<=theta[j,i]<67.5 or -157.5<theta[j,i]<=-112.5:
                if mask[0,2]>D[j,i] or mask[2,0]>D[j,i]:
                    T[j,i] = 0
            elif 67.5<=theta[j,i]<112.5 or -112.5<theta[j,i]<=-67.5:
                if mask[0,1]>D[j,i] or mask[2,1]>D[j,i]:
                    T[j,i] = 0
            elif 112.5<=theta[j,i]<157.5 or -67.5<theta[j,i]<=-22.5:
                if mask[0,0]>D[j,i] or mask[2,2]>D[j,i]:
                    T[j,i] = 0
            else:
                if mask[1,0]>D[j,i] or mask[1,2]>D[j,i]:
                    T[j,i] = 0                   
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 3))
    plt.subplot(121)
    plt.imshow(T,cmap = "gray",vmin = 0, vmax = 1)
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.title("Thin gradient map")
    
    plt.subplot(122)
    plt.imshow(D*T,cmap = "turbo",vmin = 0, vmax = 1)
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.colorbar(fraction=0.035, pad=0.04,ticks=[0, 1])
    plt.title("Thin gradient map colored by gradients")
    plt.show()

    return T


def hysteresis_thresholding(T, D, t_high = 0, t_low = 0):
    if t_low > t_high:
        t_low = t_high
        
    F = T*(D>=t_high)
    coords = np.where((T-F)>0)

    for k in range(0,int(np.sum(T-F))):
        j = coords[0][k]
        i = coords[1][k]
        nms2 = F[j-1:j+2,i-1:i+2]
        nms2[1,1] = 0
        if D[j,i]>= t_low and np.sum(nms2)>0:
            F[j,i] = 1
            
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 3))
    plt.subplot(122)
    plt.imshow(F,cmap = "gray",vmin = 0, vmax = 1)
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.title("Final flame edge")

    plt.subplot(121)
    plt.imshow(D*F,cmap = "turbo",vmin = 0, vmax = 1)
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.colorbar(fraction=0.035, pad=0.04,ticks=[0, 1])
    plt.title("Thin gradient map colored by gradients")
    plt.show()







