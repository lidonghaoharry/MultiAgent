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
from perturb_utils import save_image, config_loader
import argparse
from queue import Queue
import torch
import rosbag
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
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
class RosbagPerturbation:
    def __init__(self, args, general_dir, configs):
        # args.agent_number, args.noise_type, args.severity, args.experiment

        self.perturbation_loader = config_loader(configs, 'perturbation')
        self.config = configs
        self.exp_name = args.experiment
        self.ag_n = args.agent_number  # agent number
        rospy.init_node("perturbation_n{}".format(self.ag_n), anonymous=True)
        self.bridge = CvBridge()
        self.process_buffer = Queue(maxsize=0)
        self.sleep = 0.05
        # Define a perturbation method
        
        self.nosie_type = args.noise_type
        print("Perturbation method: {}".format(self.nosie_type))
        if d.get(self.nosie_type) != None: 
            self.perturb_func = d[self.nosie_type]
        else:
            rospy.logerr("Invalid perturbation method. \n")
            sys.exit(1)
        
        if self.perturbation_loader.load_config("save_image"):
            original_imgs_dir = general_dir + "/original_images"

        # Perturbation severity
        self.lv = args.severity

        self.denoise = args.denoise
        

        print("Starting perturbation node for agent {} with severity of {}".format(self.ag_n, self.lv))


        purb_res_dir = general_dir + "/perturbed_results/{}/{}_n{}_lv{}_exp{}".format(self.exp_name,self.nosie_type, self.ag_n, self.lv, args.trail)
        # Save perturbed images
        if self.config['perturbation'].get('save_image') or self.config['perturbation'].get('save_perturbed_bag'):
            if not os.path.exists(purb_res_dir):
                os.makedirs(purb_res_dir)
            if self.config['perturbation'].get('save_image'):
                print("Saving perturbed images")
                self.original_imgs_dir = original_imgs_dir
                self.save_freq = configs['perturbation']['save_frequency'] # Save frequency
            else:
                self.save_freq = 100
            self.save_dir = purb_res_dir
            self.counter = 0
        self.timeout_duration = 20

    def image_callback(self, data):
        # print("data info: ", data.header)
        self.process_buffer.put(data)
        rospy.sleep(self.sleep)
    
    def imu_callback(self, data):
        if self.bag != None:
            self.bag.write("/imu{}".format(self.ag_n), data)

    
    def perturbate_msg(self, data, publish_perturbed_image=False):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # print("image encoding", data.encoding) # mono8
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            # print("Image shape: ", cv_image.shape) # (480, 752)
        except CvBridgeError as e:
            rospy.logerr(e)
            return None
        
        if self.counter % 100 == 0 and self.config['perturbation'].get('save_image'):
            print("Processing image {}".format(self.counter))
            self.save_image(cv_image, self.original_imgs_dir)
        
        # Execute the perturbation
        # torch_img = torch.from_numpy(cv_image)
        perturbed_image = self.perturbation(cv_image)

        # print("Perturbed image shape: ", perturbed_image.shape) # (480, 752)
        # print("Perturbed image type: ", perturbed_image.dtype) # float64
        if self.counter % 100 == 0 and self.config['perturbation'].get('save_image'):
            self.save_image(perturbed_image, self.save_dir)
        
        self.counter += 1

        try:
            perturbed_img_msg = self.bridge.cv2_to_imgmsg(perturbed_image, "mono8")
            perturbed_img_msg.header = data.header
        except CvBridgeError as e:
            print(e)
            return None
        
        if (publish_perturbed_image):
            # Publish the perturbed image to topic "/cam0/image_perturbed"
            self.perturbed_image_pub.publish(perturbed_img_msg)

        return perturbed_img_msg

    
    def save_image(self, image, path):
        file_name = os.path.join(path, f'perturbed_image_{self.counter}.jpg')
        cv2.imwrite(file_name, image)

    def run(self):
        print("{} Perturbation node for agent {} is running, with severity of {}".format(self.nosie_type, self.ag_n, self.lv))
        start_time = rospy.Time.now()
        # Subscribe to the rosbag topic
        self.image_sub = rospy.Subscriber("/cam0/image_raw{}".format(self.ag_n), Image, self.image_callback)
        self.imu_sub = rospy.Subscriber("/imu{}".format(self.ag_n), Imu , self.imu_callback)
        # self.imu_pub = rospy.Publisher("/imu{}".format(self.ag_n), Imu, queue_size=10)
        self.perturbed_image_pub = rospy.Publisher("/cam0/image_perturbed{}".format(self.ag_n), Image, queue_size=10)
        perturbed_bag_path = os.path.join(self.save_dir, "perturbed_images.bag")
        if self.config['perturbation'].get('save_perturbed_bag'):
            self.bag = rosbag.Bag(perturbed_bag_path, 'w')
        else:
            self.bag = None
        while not rospy.is_shutdown():
            if not self.process_buffer.empty():
                start_time = rospy.Time.now()
                data = self.process_buffer.get()
                perturbed_image = self.perturbate_msg(data, True)
            
            if rospy.Time.now() - start_time > rospy.Duration(self.timeout_duration):
                rospy.logwarn("Havn't received any message for %ss, Timeout", str(self.timeout_duration))
                if self.bag != None:
                    self.bag.close()
                rospy.signal_shutdown("Timeout")

            rospy.sleep(self.sleep)
        
        rospy.spin()
    
    def overwrite_bag(self, frequency=1000, duration=5):
        print("{} Perturbation node for agent {} is running, with severity of {}".format(self.nosie_type, self.ag_n, self.lv))
        ori_bag_path = os.path.join("/home/lidonghao/MultiAgent/data/EuRoC", "MH_0{}_easy.bag".format(self.ag_n+1))
        with rosbag.Bag(ori_bag_path, 'r') as source_bag, rosbag.Bag(os.path.join(self.save_dir, "perturbed_images.bag"), 'w') as target_bag:
            for topic, msg, t in source_bag.read_messages(topics=["/cam0/image_raw", "/imu0"]):
                if topic == "/cam0/image_raw":
                    if self.counter % frequency in range(duration) and self.counter > 500:
                        print("Perturbing image {}".format(self.counter))
                        perturbed_image = self.perturbate_msg(msg, publish_perturbed_image=False)
                    else:
                        # print("Not perturbing image {}".format(self.counter))
                        perturbed_image = msg
                        self.counter += 1
                    target_bag.write('/cam0/image_perturbed{}'.format(self.ag_n), perturbed_image, t)
                elif topic == "/imu0":
                    target_bag.write('/imu{}'.format(self.ag_n), msg, t)

    def learning_based_denoise(self, img, mode):
        noise_level_img = 80                 # set AWGN noise level for noisy image
        noise_level_model = noise_level_img  # set noise level for model
        x8 = True                           # default: False, x8 to boost performance
        border = 0                           # shave boader to calculate PSNR and SSIM
        n_channels = 1                       # set 1 for grayscale image, set 3 for color image
        model_pool = '/home/lidonghao/ws/covins_ws/src/ablation_perturbation/src/model_zoo'             # fixed
        model_name = mode

        model_path = os.path.join(model_pool, model_name+'.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        # ----------------------------------------
        # load model
        # ----------------------------------------

        from utils import utils_model
        from utils import utils_image as util
        if 'drunet' in model_name:
            from models.network_unet import UNetRes as net
            model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
            model.load_state_dict(torch.load(model_path), strict=True)
        elif 'ircnn' in model_name:
            from models.network_dncnn import IRCNN as net
            model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
            model25 = torch.load(model_path)
            former_idx = 0
        elif 'dncnn' in model_name:
            from models.network_dncnn import DnCNN as net
            model = net(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
            model.load_state_dict(torch.load(model_path), strict=True)
        else:
            raise ValueError('model is not selected!')
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

        if n_channels == 1:
            img_H = np.expand_dims(img, axis=2)  # HxWx1
        elif n_channels == 3:
            if img.ndim == 2:
                img_H = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
            else:
                img_H = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        img_L = util.uint2single(img_H)

        # Add noise without clipping
        # np.random.seed(seed=0)  # for reproducibility
        # img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        # util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        if(model_name == 'drunet_gray'):
            img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        if(model_name == 'drunet_gray'):
            if not x8 and img_L.size(2)//8==0 and img_L.size(3)//8==0:
                img_E = model(img_L)
            elif not x8 and (img_L.size(2)//8!=0 or img_L.size(3)//8!=0):
                img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
            elif x8:
                img_E = utils_model.test_mode(model, img_L, mode=3)
        else:
            img_E = model(img_L)

        img_E = util.tensor2uint(img_E)

        return img_E

    def denoise_image(self, cv_image, mode):
        if mode == "wavelet":
            img = denoise_wavelet(cv_image, channel_axis=None, rescale_sigma=True, method='BayesShrink')
            img = np.array(img).astype(np.uint8)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if mode in ["drunet_gray", "ircnn_gray", "dncnn3"]:
            # print("Denoising image with model: ", mode)
            img = self.learning_based_denoise(cv_image, mode)
        return img
    
    def perturbation(self, cv_image):
        # Apply perturbation to the image
            # cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        rng = np.random.default_rng()
        # perturbed_image = cv2.add(cv_image, cv2.randn(cv_image, 0, self.perturbation_std))
        if self.lv == 99:
            img = PILImage.fromarray(cv_image)
            img = self.perturb_func(img, rng.integers(1, 5, endpoint=True))
            if self.denoise:
                # Denoise the image
                img = self.denoise_image(img, "dncnn3") # drunet_gray, ircnn_gray, dncnn3
        else:
            img = PILImage.fromarray(cv_image)
            img = self.perturb_func(img, self.lv)
            if self.denoise:
                # Denoise the image
                img = self.denoise_image(img, "dncnn3") # drunet_gray, ircnn_gray, dncnn3
        img = np.array(img).astype(np.uint8)

        return img

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Read frames form file, apply perturbations, then publish to rostopic")
    parser.add_argument("--agent_number", type=int, help="Agent number", default=0)
    parser.add_argument("--severity", type=int, help="Perturbation severity [1,5]", default=1)
    # parser.add_argument("--start_time", type=int, help="Start time", default=0)
    parser.add_argument("--trail", type=int, help="Trail number", default=1)
    parser.add_argument("--noise_type", type=str, help="Perturbation type", default="gaussian_noise")
    parser.add_argument("--config", type=str, help="Config file", default="/home/lidonghao/ws/covins_ws/src/ablation_perturbation/test.yaml")
    parser.add_argument("--experiment", type=str, help="Experiment name", default="test")
    parser.add_argument("--overwrite_bag_only", action="store_true", help="Overwrite the bag file only")
    parser.add_argument("--noise_freq", type=int, help="Perturbation frequency", default=100)
    parser.add_argument("--noise_duration", type=int, help="Perturbation duration", default=5)
    parser.add_argument("--denoise", action="store_true", help="Denoise the image")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Perturbation standard deviation
    dataset = configs['test']['dataset']
    work_dir = configs['perturbation']['work_dir'] + "/results/{}/agn{}".format(dataset, args.agent_number)
    perturbation = RosbagPerturbation(args, work_dir, configs)
    if args.overwrite_bag_only:
        print("Overwriting the bag file only with frequency {} and duration {}".format(args.noise_freq, args.noise_duration))
        perturbation.overwrite_bag(frequency=args.noise_freq, duration=args.noise_duration)
    else:
        perturbation.run()
    