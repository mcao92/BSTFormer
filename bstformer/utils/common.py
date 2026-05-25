import torch
import scipy.io as scio
import numpy as np
import logging
import time
import os
import os.path as osp
import cv2
import math
import einops
import random

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compare_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))

def compare_psnr(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def random_inp_mask(frames=8,size_h=256,size_w=256,mask_path=None):
    mask_t = np.zeros(frames-1)
    mask_t = np.insert(mask_t,0,values=1)
    mask = einops.repeat(mask_t,"b->b h w",h=size_h,w=size_w)
    for i in range(size_h):
        for j in range(size_w):
            random.shuffle(mask[:,i,j])

    mask = mask.astype(np.float32)
    mask_s = np.sum(mask,axis=0)
    mask_s[mask_s==0] = 1
    print("sum:",np.sum(mask_s))
    return torch.from_numpy(mask,),torch.from_numpy(mask_s)

def random_masks(frames=8,size_h=256,size_w=256,mask_path=None):
    if mask_path is None:
        mask = np.random.randint(0,high=2,size=(frames,size_h,size_w)).astype(np.float32)
        # np.save("mask.npy",mask)
    else:
        mask = np.load(mask_path)
    mask_s = np.sum(mask,axis=0)
    mask_s[mask_s==0] = 1
    print("sum:",np.sum(mask_s))
    return torch.from_numpy(mask,),torch.from_numpy(mask_s)

def save_image(out,gt,image_name,show_flag=False):
    sing_out = out.transpose(1,0,2).reshape(out.shape[1],-1)
    sing_gt = gt.transpose(1,0,2).reshape(gt.shape[1],-1)
    result_img = np.concatenate([sing_out,sing_gt],axis=0)*255
    cv2.imwrite(image_name,result_img)
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)
def generate_masks(mask_path):
    mask = scio.loadmat(osp.join(mask_path,'mask.mat'))
    mask = mask['mask']
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0,dtype=np.float32)
    mask_s[mask_s==0] = 1
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    return mask, mask_s

def Logger(log_dir):
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s")

    localtime = time.strftime("%Y_%m_%d_%H_%M_%S")
    logfile = osp.join(log_dir,localtime+".log")
    fh = logging.FileHandler(logfile,mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def random_real_masks(frames=10,size_h=512,size_w=512,mask_path=None):
    mask_dict = scio.loadmat(mask_path)
    mask = mask_dict["mask"]
    h,w,f = mask.shape
    if frames is None:
        frames=np.random.randint(10,51)
    if size_h!=h or size_w!=w:
        h_begin = np.random.randint(0,h-size_h)
        w_begin = np.random.randint(0,w-size_w)
        f_begin = np.random.randint(0,f-frames+1)
        mask = mask[h_begin:h_begin+size_h,w_begin:w_begin+size_w,f_begin:f_begin+frames]
    else:
        mask = mask[:,:,0:frames]

    mask = mask.transpose(2,0,1)
    mask_s = np.sum(mask,axis=0)
    mask_s[mask_s==0] = 1
    print("sum:",np.sum(mask_s))
    return torch.from_numpy(mask,),torch.from_numpy(mask_s)
