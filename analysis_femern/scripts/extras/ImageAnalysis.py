#!/usr/bin/env python
# Created by Jonathan Mikler on 06/November/22

# Library for class Aerial Imaging DTU
import numpy as np
import pandas as pd
import seaborn as sns
import cv2 as cv
import scipy.stats as sp

from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors

def PCA(img_:np.ndarray):
    """
    img_ is a (m,n,b)
    """
    covariance = np.cov(flatten_image(img_).transpose())
    eigen_vals, eigen_vecs = np.linalg.eig(covariance)

    # eigen values score
    score = (eigen_vals * 100)/np.sum(eigen_vals)

    # Loads
    loads = eigen_vecs * np.sqrt(eigen_vals)

    return covariance, eigen_vals, eigen_vecs, score, loads

def img_stats(img_:np.ndarray, name_:str='No name given', print_=True):
    """
    returns mean, max, min
    """
    
    _global_mean = img_.mean()
    _global_max = img_.max()
    _global_min = img_.min()
    _shape = img_.shape

    if print_:
        print(f"Image: {name_} shape:{_shape}")
        print(f"Overall \
            mean:{_global_mean} \
            max: {_global_max},\
            min: {_global_min}")

        if len(img_.shape) == 3:
            print("Band individual")
            for i in range(_shape[2]):
                _mean = img_[:,:,i].mean()
                _max = img_[:,:,i].max()
                _min = img_[:,:,i].min()
                print(f"Band {i+1} \
                mean:{np.round(_mean,4)} \
                max: {np.round(_max,4)},\
                min: {np.round(_min,4)}")

    return _global_mean, _global_max, _global_min

def normalize_img(image_:np.ndarray):
    norm_img = np.zeros(image_.shape)

    for i in range(image_.shape[2]):
        norm_img[:,:,i] = (image_[:,:,i] - image_[:,:,i].mean()) / image_[:,:,i].std() 
    return norm_img

def flatten_image(image_:np.ndarray):
    """Image needs to be a (m,n,u) u>0"""

    if len(image_.shape) == 3:
        channel_vectors = np.zeros((image_.shape[0] * image_.shape[1],image_.shape[2]))
        for i in range(image_.shape[2]):
            flat_band = image_[:,:,i].flatten()
            channel_vectors[:,i] = flat_band
    else:
        channel_vectors = np.zeros((image_.shape[0] * image_.shape[1]))
        channel_vectors = image_.flatten()

    return channel_vectors

def plot_bands(img_:np.ndarray, name_:str, plot_bands_together=False):
    band_n = img_.shape[2]
    print(f"plotting {band_n} bands")

    fig = plt.figure(constrained_layout=True, figsize=(14,6))
    # fig.suptitle(f"{name_} indicidual bands")
    ax_no = band_n if not plot_bands_together else band_n +1
    ax_array = fig.subplots(1, ax_no, squeeze=False)

    for i in range(band_n):
        pos = ax_array[0,i].imshow(img_[:,:,i],cmap='gray')
        ax_array[0,i].set_title(f"Band {i}")
        fig.colorbar(pos, ax=ax_array[0,i], shrink=0.25)
    
    if plot_bands_together:
        ax_array[0,ax_no-1].imshow(img_)
        ax_array[0,ax_no-1].set_title("All bands")

    plt.show()
    return fig

def scale_img(img_:np.ndarray):
    scaled_img = np.zeros(img_.shape)
    if len(img_.shape) == 3:
        for i in range(img_.shape[2]):
            scaled_img[:,:,i] = scale(img_[:,:,i])
    else:
        scaled_img[:,:] = scale(img_[:,:])
    return scaled_img

def scale(x):
    return((x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x)))

def get_rgb(img_:np.ndarray, rgb_idxs=(2,1,0)):
    r = img_[:,:,rgb_idxs[0]]
    g = img_[:,:,rgb_idxs[1]]
    b = img_[:,:,rgb_idxs[2]]
    return np.stack((r,g,b),axis=2)

def crop_image(img_:np.ndarray, w_name_:str):
    """
    Crops the image as selected
    returns cropped array and roi idxs
    """
    rgb_channels = get_rgb(img_)
    # Select ROI
    r = cv.selectROI(w_name_,rgb_channels)

    # Crop image
    imCrop = img_[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    cv.destroyAllWindows()
    return imCrop, r

def crop_with_indices(image_:np.ndarray, crop_idxs_:tuple):
    """
    with idxs (x,ox,y,oy)
    ox and oy are offsets in x and y
    return image cropped in indices (x:x+ox,y:y+oy)
    """
    assert(isinstance(image_, np.ndarray)), "Image must be a numpy array"
    return image_[int(crop_idxs_[1]):int(crop_idxs_[1]+crop_idxs_[3]), int(crop_idxs_[0]):int(crop_idxs_[0]+crop_idxs_[2])]

def estimate_category_liklehood(img_:np.ndarray, subset_:np.ndarray):
    """
    @param img_: matrix (m,n,b) with b>0. Is the entire image to be analyzed\n
    @param train_subset_: (a,b). Flattened subset of the image
    returns matrix with img_shape with the likleyhood of each pixel belonging to the class represented by train_subset
    """
    assert(img_.shape[2]==subset_.shape[1]),"input image and subset should have same size in dim 2. img:{} subset:{}".format(img_.shape[2],subset_.shape[1])
    assert(img_.shape[2])
    
    return sp.multivariate_normal.pdf(
        x = flatten_image(img_),
        cov = np.cov(subset_, rowvar=False),
        mean = subset_.mean(axis=0))

def get_clasified_roi(img_:np.ndarray, class_idx:int):
    """
    return array with flatten image bands and an additional vector with class_idx
    """
    train_x = flatten_image(img_)
    train_y = class_idx*np.ones((train_x.shape[0],1))
    return np.hstack((train_x, train_y))

# Plotting
def plot_image(img_:np.ndarray, cmap=None, norm=None, save_=False, filename_:str=""):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # TODO: add configuration for axes and title
    # plt.figure()
    # # plt.imshow(classified_yukon)
    # plt.imshow(classified_yukon,  cmap=cmap, norm=norm)
    # plt.title("Clasified")
    # plt.colorbar(ticks=range(n_categories))
    # plt.clim(-0.5, n_categories - 0.5)

    ax.imshow(img_, aspect='auto', cmap=cmap, norm=norm)
    if save_:
        assert(filename_!=""), "To save image, please provide a filename"
        fig.savefig(f'images/{filename_}.jpg')

def discrete_custom_cmap(colors_:list):
    cmap = plt_colors.ListedColormap(colors_)
    b = np.array(range(0,len(colors_)+1))-0.5
    boundaries = b.tolist()
    norm = plt_colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm

def plot_6(img_:np.ndarray, title_:str, header_name_:str, vmax_=None,vmin_=None, share_scale_:bool=False,cmap_='gray', save_fig_:bool=False,  fname_:str=""):
    fig = plt.figure(constrained_layout=True,figsize=(14,6))
    fig.suptitle(f"{title_}")
    n_bands = img_.shape[2]
    # fig.suptitle(f"{name_} indicidual bands")
    ax_array = fig.subplots(2, 3, squeeze=False)

    idx = 0
    for i in range(2):
        for j in range(3):
            if idx == n_bands:
                continue
            if share_scale_ and vmax_ != None and vmin_ != None:
                pos = ax_array[i,j].imshow(img_[:,:,idx],cmap=cmap_,vmax=vmax_,vmin=vmin_)
            else:
                pos = ax_array[i,j].imshow(img_[:,:,idx],cmap=cmap_)
            ax_array[i,j].set_title(f"{header_name_} {idx+1}")
            fig.colorbar(pos, ax=ax_array[i,j])
            idx += 1

        if idx == n_bands:
            continue
    
    plt.show()

    if save_fig_:
        fig.savefig(f"images/{fname_}.jpg")
    return fig

def plot_6bands_hist(img_:np.ndarray, header_name_:str, title_:str, save_fig_:bool=False, fname_:str="Image", color_:str='skyblue'):
    # bands_flat = flatten_image(img_)
    bands_flat = flatten_image(img_)
    n_bands = bands_flat.shape[1]

    fig = plt.figure(constrained_layout=True,figsize=(14,6))
    fig.suptitle(f"{title_}")
    ax_array = fig.subplots(2, 3, squeeze=False, sharey=True, sharex=True)

    idx = 0
    for i in range(2):
        for j in range(3):
            if idx == n_bands:
                continue
            ax_array[i,j].hist(bands_flat[:,idx], bins=50, color=color_)
            ax_array[i,j].set_title(f"{header_name_} {idx+1}")
            idx +=1
        
        if idx == n_bands:
            continue

    plt.show()
    if save_fig_:
        fig.savefig(f"images/{fname_}.jpg")

    return fig

def make_pairplot(flat_img_:np.ndarray, columns_:list, title_:str, save_img_:bool,fname_:str=''):
    b = sns.pairplot(pd.DataFrame(flat_img_, columns = columns_), diag_kind='kde',plot_kws={"s": 3})
    b.fig.suptitle(title_)

    if save_img_:
        b.fig.savefig(fname=f"images/{fname_}.jpg")
    return b

if __name__ == '__main__':
    pass