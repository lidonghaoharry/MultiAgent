#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image, Imu
import sys
import os
from cv_bridge import CvBridge, CvBridgeError
from message_filters import TimeSynchronizer, Subscriber
import robustness as methods
from PIL import Image as PILImage
import numpy as np
import yaml
from utils import save_image, config_loader
import argparse
from queue import Queue
import torch
import rosbag
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
# import image_dehazer
import matplotlib.pyplot as plt

d = {'gaussian_noise': methods.gaussian_noise, 
        'shot_noise': methods.shot_noise,
        'impulse_noise': methods.impulse_noise,
        'speckle_noise': methods.speckle_noise,
        'gaussian_blur': methods.gaussian_blur,
        'glass_blur': methods.glass_blur, 
        'defocus_blur': methods.defocus_blur,
        'fog': methods.fog,
        'frost': methods.frost,
        'contrast': methods.contrast, 
        'brightness': methods.brightness,
        'saturate': methods.saturate, 
        'zoom_blur': methods.zoom_blur,
        'motion_blur': methods.motion_blur,
        'snow': methods.snow,
        'spatter': methods.spatter, 
        'jpeg_compression' : methods.jpeg_compression,
        'pixelate': methods.pixelate,
        'none': methods.none
        }

def image_repair(image, severity, noise_type):

    original = image
    noisy = d[noise_type](image, severity)
    print(original.shape, noisy.shape)
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)
    ax[0, 0].imshow(noisy)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Noisy')

    # ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1, channel_axis=None))
    # ax[0, 1].axis('off')
    # ax[0, 1].set_title('TV')
    # ax[0, 2].imshow(denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
    #             channel_axis=None))
    # ax[0, 2].axis('off')
    # ax[0, 2].set_title('Bilateral')

    wavelet_denoised = denoise_wavelet(noisy, channel_axis=None, rescale_sigma=True, method='BayesShrink')
    ax[0, 1].imshow(wavelet_denoised)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Wavelet denoising BayesShrink')

    alpha = 1.5
    enhanced_image = cv2.addWeighted(noisy, alpha, wavelet_denoised, 1 - alpha, 0)
    ax[0, 2].imshow(enhanced_image)
    ax[0, 2].axis('off')
    ax[0, 2].set_title('Enhanced Image')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = enhanced_image.astype(np.uint8)
    cl1 = clahe.apply(enhanced_image)

    ax[0, 3].imshow(cl1)
    ax[0, 3].axis('off')
    ax[0, 3].set_title('Apply CLAHE')

    Laplacian = cv2.Laplacian(wavelet_denoised, cv2.CV_64F)
    laplacian_8bit = cv2.convertScaleAbs(Laplacian)

    ax[1, 1].imshow(laplacian_8bit)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Laplacian')
    
    noise_int = noisy.astype(np.uint8)
    nlm_denoised = cv2.fastNlMeansDenoising(noise_int, None, 10, 7, 21)
    ax[1, 2].imshow(nlm_denoised) # filter strength, window size
    ax[1, 2].axis('off')
    ax[1, 2].set_title('Non-local means denoising ')

    # nlm_wavelet_denoised = denoise_wavelet(nlm_denoised, channel_axis=None, rescale_sigma=True, method='BayesShrink')
    # enhanced_image_nlm = cv2.addWeighted(nlm_denoised, alpha, nlm_wavelet_denoised, 1 - alpha, 0)
    # clahe_nlm = clahe.apply(enhanced_image_nlm)
    # ax[1, 3].imshow(clahe_nlm)


    ax[1, 0].imshow(original)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Original')

    fig.tight_layout()

    plt.show()

def brightness_repair(image, severity, noise_type):

    original = image
    noisy = d[noise_type](image, severity)
    print(original.shape, noisy.shape)
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)
    # noisy = noisy.clip(0, 255)
    print(noisy)
    ax[0, 0].imshow(noisy, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Noisy')

    threshold = 50
    average_in_hist = np.mean(image)
    print(f"average_in_hist: {average_in_hist}", f", threshold: {threshold}")

    if np.mean(noisy) < average_in_hist - threshold:
        print("Too dark")
        alpha = 1.5
        enhanced_image = cv2.addWeighted(noisy, alpha, noisy, 1 - alpha, 0)
        ax[0, 1].imshow(enhanced_image, cmap='gray')
        ax[0, 1].axis('off')
        ax[0, 1].set_title('Weighted')

        c = [.1, .2, .3, .4, .5][severity - 1]
        constant = 255 * c
        adjust_simple = cv2.add(noisy, constant)
        adjust_simple = adjust_simple.clip(0, 255)
        print(adjust_simple)
        ax[0, 2].imshow(adjust_simple, cmap='gray')
        ax[0, 2].axis('off')
        ax[0, 2].set_title('Simple adjust')

        noisy_uint = noisy.astype(np.uint8)
        adjust_hist = cv2.equalizeHist(noisy_uint)
        ax[0, 3].imshow(adjust_hist, cmap='gray')
        ax[0, 3].axis('off')
        ax[0, 3].set_title('Hist adjust')

        gamma = 0.5
        gamma_corrected = np.array(255 * (noisy / 255) ** gamma, dtype = 'uint8')
        ax[1, 1].imshow(gamma_corrected, cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 1].set_title('Gamma adjust')
    


    ax[1, 0].imshow(original, cmap='gray')
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Original')

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description="Read frames form file, apply perturbations, then publish to rostopic")
    # parser.add_argument("--agent_number", type=int, help="Agent number", default=0)
    # parser.add_argument("--severity", type=int, help="Perturbation severity [1,5]", default=1)
    # # parser.add_argument("--start_time", type=int, help="Start time", default=0)
    # parser.add_argument("--trail", type=int, help="Trail number", default=1)
    # parser.add_argument("--noise_type", type=str, help="Perturbation type", default="gaussian_noise")
    # parser.add_argument("--config", type=str, help="Config file", default="/home/lidonghao/ws/covins_ws/src/ablation_perturbation/test.yaml")
    # parser.add_argument("--experiment", type=str, help="Experiment name", default="test")
    # parser.add_argument("--overwrite_bag_only", action="store_true", help="Overwrite the bag file only")
    # parser.add_argument("--noise_freq", type=int, help="Perturbation frequency", default=100)
    # parser.add_argument("--noise_duration", type=int, help="Perturbation duration", default=5)

    # args = parser.parse_args()
    # with open(args.config, 'r') as f:
    #     configs = yaml.load(f, Loader=yaml.FullLoader)

    # # Perturbation standard deviation
    # dataset = configs['test']['dataset']
    # work_dir = configs['perturbation']['work_dir'] + "/results/{}/agn{}".format(dataset, args.agent_number)
    # perturbation = RosbagPerturbation(args, work_dir, configs)
    # if args.overwrite_bag_only:
    #     print("Overwriting the bag file only with frequency {} and duration {}".format(args.noise_freq, args.noise_duration))
    #     perturbation.overwrite_bag(frequency=args.noise_freq, duration=args.noise_duration)
    # else:
    #     perturbation.run()
    image_path = "/home/lidonghao/ws/covins_ws/src/ablation_perturbation/results/EuRoC/agn0/original_images/perturbed_image_20.jpg"
    image = img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noise_type = "brightness"
    if(noise_type == "gaussian_noise"):
        image_repair(image, 5, "gaussian_noise")
    elif(noise_type == "brightness"):
        brightness_repair(image, 5, "brightness")
    