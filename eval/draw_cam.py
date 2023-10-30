import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import os
from torch.utils.data import DataLoader

#import model
from ..lib.Network import LDCNet
#import dataloader
from ..utils.dataloader import name_kvasir_test_dataset

#import util
from ..configs.test_config import config
from ..utils.metric import *
from ..utils.preprocess import *

def tensor2numpy(tesnor_img):
    clean = tesnor_img.clone().detach().cpu().squeeze(0)
    if len(clean.shape) == 2:#gray
        clean = np.float32(clean)
        clean = np.repeat(clean[:, :, None], 3, axis=-1)#转换成3通道
    else:#rgb
        clean = np.float32(clean).transpose(1, 2, 0)     # 跟换三通道 (C, H, W) --> (H, W, C)
    # clean = np.around(clean.mul(255))                     # 转换到颜色255 [0, 1] --> [0, 255]
    return clean

#USE GRAD CAM to visuallise network
def draw_CAM():
    #
    class OneLabelSegmentationTarget:
        def __init__(self,mask,batch_num):
            self.batch_num = batch_num
            self.mask = mask
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()
            
        def __call__(self, model_output):
            return (model_output[self.batch_num,:,:].squeeze() * self.mask).sum()

    from pytorch_grad_cam import GradCAM,ScoreCAM,AblationCAM


    os.environ['CUDA_VISIBLE_DEVICES'] = "0"#choose GPU
    gpu_id = "cuda:0"
    device = torch.device(gpu_id)

    model_path = "MODEL SAVE PATH"
    save_root = "PICTURE SAVE PATH"

    target_layers = [model.feature_extractor.layer3]#target layer
    # target_layers = [model.High_RFB]
    # target_layers = [model.ddf2]

    model = LDCNet().to(device)
    #get pth
    print("load model from "+ model_path)
    state_dict = torch.load(model_path, map_location=torch.device(gpu_id if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    test_dataset = name_kvasir_test_dataset(img_size=512)
    dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8, drop_last=True)
    
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    count = 0
    for img, gt, name in dataloader:
        img = img.to(device)
        result = model(img)
        result = result.cpu()#picture:[b,1,h,w];video:[b*t,1,h,w]
        mask = torch.gt(result,0.5)
        gt = gt.detach().cpu()#picture:[b,1,h,w];video:[b,t,1,h,w]
        for batch_num in range(mask.shape[0]):
            silce_mask = mask[batch_num,:,:,:].detach().squeeze()#[h,w]
            silce_name = name[batch_num]
            targets = [OneLabelSegmentationTarget(silce_mask,batch_num)]
            cam = GradCAM(model=model,target_layers=target_layers,use_cuda=torch.cuda.is_available())
            grayscale_cam = cam(input_tensor=img,
                                    targets=targets)[0, :]

            rgb_img = tensor2numpy(img[batch_num,:,:,:])
            gray_gt = tensor2numpy(gt[batch_num,:,:,:]) * 255
            gray_predmask = tensor2numpy(silce_mask) * 255

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            print("save image in "+os.path.join(save_root,silce_name))
            Image.fromarray(cam_image).save(os.path.join(save_root,silce_name))



if __name__ == "__main__":
    draw_CAM()




