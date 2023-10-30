import os

import numpy as np
import torch
from configs import config
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from utils.preprocess import *
# from preprocess import *
# from torchvision.transforms import Compose,RandomCrop,RandomHorizontalFlip,RandomVerticalFlip,Normalize,ToTensor
from torchvision.transforms import functional as F

# pretrain dataset
class Pretrain(Dataset):
    #img_dataset_name: data name list csv file
    #transform: image preprocessing
    def __init__(self, img_dataset_list, transform, get_name = False):
        self.dataset_root = config.img_dataset_root
        data_dir = config.img_dataset_root + img_dataset_list
        img_dataset = np.loadtxt(open(data_dir,"rb"), dtype=str, delimiter=",",skiprows=1)
        self.file_list =  img_dataset
        self.img_label_transform = transform
        self.get_name = get_name

    def __getitem__(self, idx):
        data_path = self.file_list[idx]
        # name = os.path.splitext(data_path[1])[0]
        name = data_path[1]
        data_path = os.path.join(data_path[0] ,data_path[1])

        img_path = os.path.join(self.dataset_root, "images", data_path)
        label_path = os.path.join(self.dataset_root, "labels", data_path)
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        img, label = self._process(img, label)
        if self.get_name:
            return img, label, name
        else:
            return img, label

    def _process(self, img, label):

        img, label = self.img_label_transform(img, label)
        return img, label

    def __len__(self):
        return len(self.file_list)


def get_pretrain_train_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize(config.size[0], config.size[1]),
        # RandomCrop((config.size[0], config.size[1])),
        Random_crop_Resize(20),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        ToTensor(),
        Normalize(data_mean, data_std)
    ])
    train_loader = Pretrain("train.csv", transform=trsf_main)

    return train_loader


def get_pretrain_test_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize(config.size[0], config.size[1]),
        ToTensor(),
        Normalize(data_mean, data_std)
    ])
    test_loader = Pretrain("test.csv", transform=trsf_main)

    return test_loader




def name_pretrain_test_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize(config.size[0], config.size[1]),
        ToTensor(),
        Normalize(data_mean, data_std)
    ])
    test_loader = Pretrain("test.csv", transform=trsf_main,get_name=True)
    return test_loader

class image_eval_para(Dataset):
    def __init__(self,len,size):
        self.len = len
        self.size = size
    def __getitem__(self, idx):
        img = torch.zeros(3,self.size[0],self.size[1])
        label = torch.zeros(1,self.size[0],self.size[1])
        return img, label
    def __len__(self):
        return len

class video_eval_para(Dataset):
    def __init__(self,len,size):
        self.time_clips = config.video_time_clips
        self.len = len
        self.size = size
    def __getitem__(self, idx):
        img = torch.zeros(self.time_clips,3,self.size[0],self.size[1])
        label = torch.zeros(self.time_clips,1,self.size[0],self.size[1])
        return img, label
    def __len__(self):
        return len

# def get_pretrain_train_dataset():
#     data_mean = torch.load("/home/yinyiyang/3d_Unet/data/data_Normalize_mean.pth")
#     data_std = torch.load("/home/yinyiyang/3d_Unet/data/data_Normalize_std.pth")
#     trsf_main = Compose_imglabel([
#         Resize(config.size[0], config.size[1]),
#         Random_crop_Resize(15),
#         Random_horizontal_flip(0.5),
#         toTensor(),
#         Normalize(data_mean, data_std)
#     ])
#     train_loader = Pretrain("train.csv", transform=trsf_main)

#     return train_loader


# def get_pretrain_test_dataset():
#     data_mean = torch.load("/home/yinyiyang/3d_Unet/data/data_Normalize_mean.pth")
#     data_std = torch.load("/home/yinyiyang/3d_Unet/data/data_Normalize_std.pth")
#     trsf_main = Compose_imglabel([
#         Resize(config.size[0], config.size[1]),
#         toTensor(),
#         Normalize(data_mean, data_std)
#     ])
#     test_loader = Pretrain("test.csv", transform=trsf_main)

#     return test_loader


# finetune dataset
class VideoDataset(Dataset):
    def __init__(self, video_dataset_list, transform=None, time_interval=1, get_name = False):
        super(VideoDataset, self).__init__()
        self.time_clips = config.video_time_clips
        self.video_train_list = []
        self.video_filelist = {}
        for video_name in video_dataset_list:
            self.video_filelist[video_name] = []
            video_root = os.path.join(config.video_dataset_root, "images", video_name)
            label_root = os.path.join(config.video_dataset_root, "labels", video_name)
            # print(video_root)
            # print(label_root)
            img_list = os.listdir(video_root)
            img_list.sort()#排序保证时间顺序
            for img in img_list:
                img_path = os.path.join(video_root, img)
                label_path = os.path.join(label_root, img)
                if os.path.isfile(label_path):
                    self.video_filelist[video_name].append((img_path, label_path))

        # print("time clip")
        # print(self.time_clips)
        for video_name in video_dataset_list:
            li = self.video_filelist[video_name]
            for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                batch_clips = []

                #判断数据集中帧是否连续
                # img_timelabel_li = []
                is_continut = True
                temp_img_path = os.path.split(li[begin][0])[-1]
                temp_img_path = os.path.splitext(temp_img_path)[0]
                temp_img_path = temp_img_path.split('_')[-1]
                img_timelabel = int(temp_img_path)  
                # img_timelabel_li.append(img_timelabel)   
                for t in range(1, self.time_clips):
                    temp_img_path = os.path.split(li[begin + time_interval * t][0])[-1]
                    temp_img_path = os.path.splitext(temp_img_path)[0]
                    temp_img_path = temp_img_path.split('_')[-1]
                    if abs(img_timelabel - int(temp_img_path)) >= 2:#放宽条件
                        is_continut = False
                        break
                    else:
                        img_timelabel = int(temp_img_path)


                    # img_timelabel_li.append(int(temp_img_path))
                # if not is_continut:
                #     # print("is continue")
                #     print(img_timelabel_li)

                #满足连续条件才加入数据集
                if is_continut:
                    for t in range(self.time_clips):
                        batch_clips.append(li[begin + time_interval * t])
                    self.video_train_list.append(batch_clips)
        self.img_label_transform = transform
        self.get_name = get_name

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        name_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            #得到文件名
            img_name = os.path.basename(img_path)
            # name_li.append(os.path.splitext(img_name)[0])
            name_li.append(img_name)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if IMG is not None:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
        if self.get_name:
            return IMG, LABEL, name_li
        else:
            return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)

# video test dataset
class TestVideoDataset(Dataset):
    def __init__(self, video_dataset_list, transform=None, time_interval=1, get_name = False):
        super(TestVideoDataset, self).__init__()
        self.time_clips = config.video_time_clips
        print(config.video_dataset_root)
        self.video_train_list = []
        self.video_filelist = {}
        for video_name in video_dataset_list:
            self.video_filelist[video_name] = []
            video_root = os.path.join(config.video_dataset_root, "images", video_name)
            label_root = os.path.join(config.video_dataset_root, "labels", video_name)

            #用于其他
            # video_root = os.path.join(config.video_dataset_root, video_name, "images")
            # label_root = os.path.join(config.video_dataset_root, video_name, "masks")
            img_list = os.listdir(video_root)
            # print(img_list)
            img_list.sort()#排序保证时间顺序
            for img in img_list:
                img_path = os.path.join(video_root, img)
                label_path = os.path.join(label_root, img)
                if os.path.isfile(label_path):
                    # print(label_path)
                    self.video_filelist[video_name].append((img_path, label_path))

        for video_name in video_dataset_list:
            li = self.video_filelist[video_name]
            for begin in range((len(li) - (self.time_clips - 1) * time_interval - 1) // self.time_clips):
            # for begin in range(len(li) - self.time_clips + 1):
                batch_clips = []

                #判断数据集中帧是否连续
                # is_continut = True
                # temp_img_path = os.path.split(li[begin * self.time_clips][0])[-1]
                # temp_img_path = os.path.splitext(temp_img_path)[0]
                # temp_img_path = temp_img_path.split('_')[-1]
                # img_timelabel = int(temp_img_path)  
                # # img_timelabel_li.append(img_timelabel)   
                # for t in range(1, self.time_clips):
                #     temp_img_path = os.path.split(li[begin + time_interval * t][0])[-1]
                #     temp_img_path = os.path.splitext(temp_img_path)[0]
                #     temp_img_path = temp_img_path.split('_')[-1]
                #     if abs(img_timelabel - int(temp_img_path)) >= 2:#放宽条件
                #         is_continut = False
                #         break
                #     else:
                #         img_timelabel = int(temp_img_path)
                # if is_continut:
                    #满足连续条件才加入数据集
                for t in range(self.time_clips):
                    batch_clips.append(li[(begin * self.time_clips) + time_interval * t])
                        # batch_clips.append(li[begin + t])
                        # temp_img_path = os.path.split(li[begin + time_interval * t][0])[-1]
                    # print(batch_clips)
                self.video_train_list.append(batch_clips)
        self.img_label_transform = transform
        self.get_name = get_name

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        name_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            #得到文件名
            img_name = os.path.basename(img_path)
            # name_li.append(os.path.splitext(img_name)[0])
            name_li.append(img_name)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if IMG is not None:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
        if self.get_name:
            return IMG, LABEL, name_li
        else:
            return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)
def get_video_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize(data_mean, data_std)
    ])
    print("video dataset")
    print(config.video_list)
    train_loader = VideoDataset(config.video_list, transform=trsf_main, time_interval=1)

    return train_loader

def get_video_train_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(data_mean, data_std)
    ])
    print("video dataset")
    train_list = torch.load("PATH TO train_class_name_list.pth",map_location='cpu')
    print(train_list)
    train_loader = VideoDataset(train_list, transform=trsf_main, time_interval=1)

    return train_loader

def get_video_test_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        toTensor_video(),
        Normalize_video(data_mean, data_std)
    ])
    print("video test dataset")
    test_list = torch.load("PATH TO test_class_name_list.pth",map_location='cpu')
    print(test_list)
    # train_loader = VideoDataset(test_list, transform=trsf_main, time_interval=1)
    train_loader = TestVideoDataset(test_list, transform=trsf_main, time_interval=1)

    return train_loader

def get_video_eval_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        toTensor_video(),
        Normalize_video(data_mean, data_std)
    ])
    print("video eval dataset")
    test_list = torch.load("/data2/yyy/PNS_net/test_class_name_list.pth",map_location='cpu')

    choose_num = 5
    eval_list = random.sample(list(test_list),choose_num)#随机选取n个样本
    # eval_list = test_list[0:5]
    print(eval_list)
    # train_loader = VideoDataset(eval_list, transform=trsf_main, time_interval=1)
    train_loader = TestVideoDataset(eval_list, transform=trsf_main, time_interval=1,get_name=False)

    return train_loader

def name_video_test_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        toTensor_video(),
        Normalize_video(data_mean, data_std)
    ])
    print("video test dataset with name")
    test_list = torch.load("PATH TP test_class_name_list.pth",map_location='cpu')


    train_loader = TestVideoDataset(test_list, transform=trsf_main, time_interval=1, get_name=True)
    return train_loader

def name_video_picture_dataset():
    data_mean = torch.load("./data_Normalize_mean.pth")
    data_std = torch.load("./data_Normalize_std.pth")
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        toTensor_video(),
        Normalize_video(data_mean, data_std)
    ])
    print("video test dataset with name")
    test_list = torch.load("PATH TO test_class_name_list.pth",map_location='cpu')



    train_loader = TestVideoDataset(test_list, transform=trsf_main, time_interval=1, get_name=True)
    return train_loader


class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, img_dataset_list, transform = None, get_name = False):
        self.dataset_root = config.img_dataset_root
        data_dir = config.img_dataset_root + img_dataset_list
        img_dataset = np.loadtxt(open(data_dir,"rb"), dtype=str, delimiter=",",skiprows=1)
        self.file_list =  img_dataset
        self.transforms = transform
        self.get_name = get_name

    # 根据idx对应读取待训练图片以及掩码图片
    def __getitem__(self, idx):
        # 根据idx针对img与mask组合路径

        data_path = self.file_list[idx]
        # img_name = os.path.splitext(data_path[1])[0]
        img_name = data_path[1]
        data_path = os.path.join(data_path[0] ,data_path[1])
        img_path = os.path.join(self.dataset_root, "images", data_path)
        label_path = os.path.join(self.dataset_root, "labels", data_path)
        # 根据路径读取三色图片并转为RGB格式
        img = Image.open(img_path).convert('RGB')
        # 根据路径读取掩码图片默认“L”格式
        mask = Image.open(label_path).convert('L')
        # 将mask转为numpy格式，h*w的矩阵,每个元素是一个颜色id
        mask = F.to_tensor(mask)
        mask = torch.gt(mask,0.5).long()
        mask = np.array(mask)

        # print(mask)
        # 获取mask中的id组成列表，obj_ids=[0,1]
        obj_ids = np.unique(mask)
        # print(obj_ids)
        # 列表中第一个元素代表背景，不属于我们的目标检测范围，obj_ids=[1]
        obj_ids = obj_ids[1:]
        # print(obj_ids)
        # obj_ids[:,None,None]:[[[1]],[[2]]],masks(2,536,559)
        # 为每一种类别序号都生成一个布尔矩阵，标注每个元素是否属于该颜色
        masks = mask == obj_ids[:, None, None]
 
        # 为每个目标计算边界框，存入boxes
        num_objs = len(obj_ids) # 目标个数N
        boxes = [] # 边界框四个坐标的列表，维度(N,4)
        for i in range(num_objs):
            pos = np.where(masks[i]) # pos为mask[i]值为True的地方,也就是属于该颜色类别的id组成的列表
            xmin = np.min(pos[1]) # pos[1]为x坐标，x坐标的最小值
            xmax = np.max(pos[1])
            ymin = np.min(pos[0]) # pos[0]为y坐标
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        # 将boxes转化为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # 初始化类别标签
        labels = torch.ones((num_objs,), dtype=torch.int64) # labels[1,1] (2,)的array
        masks = torch.as_tensor(masks, dtype=torch.uint8) # 将masks转换为tensor
 
        # 将图片序号idx转换为tensor
        image_id = torch.tensor([idx])
        # 计算每个边界框的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是人群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # iscrowd[0,0] (2,)的array
 
        # 生成一个字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_name"] = img_name
        # 变形transform
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        if self.get_name:
            return img, target, img_name
        else:
            return img, target
 
    def __len__(self):
        return len(self.imgs)

class MaskRCNN_eval_para(torch.utils.data.Dataset):
    def __init__(self, len, size):
        self.len = len
        self.size = size

    def __getitem__(self, idx):
        # 根据idx针对img与mask组合路径

        data_path = self.file_list[idx]
        # img_name = os.path.splitext(data_path[1])[0]
        img_name = data_path[1]
        data_path = os.path.join(data_path[0] ,data_path[1])
        img_path = os.path.join(self.dataset_root, "images", data_path)
        label_path = os.path.join(self.dataset_root, "labels", data_path)
        # 根据路径读取三色图片并转为RGB格式
        img = Image.open(img_path).convert('RGB')
        # 根据路径读取掩码图片默认“L”格式
        mask = Image.open(label_path).convert('L')
        # 将mask转为numpy格式，h*w的矩阵,每个元素是一个颜色id
        mask = F.to_tensor(mask)
        mask = torch.gt(mask,0.5).long()
        mask = np.array(mask)

        # print(mask)
        # 获取mask中的id组成列表，obj_ids=[0,1]
        obj_ids = np.unique(mask)
        # print(obj_ids)
        # 列表中第一个元素代表背景，不属于我们的目标检测范围，obj_ids=[1]
        obj_ids = obj_ids[1:]
        # print(obj_ids)
        # obj_ids[:,None,None]:[[[1]],[[2]]],masks(2,536,559)
        # 为每一种类别序号都生成一个布尔矩阵，标注每个元素是否属于该颜色
        masks = mask == obj_ids[:, None, None]
 
        # 为每个目标计算边界框，存入boxes
        num_objs = len(obj_ids) # 目标个数N
        boxes = [] # 边界框四个坐标的列表，维度(N,4)
        for i in range(num_objs):
            pos = np.where(masks[i]) # pos为mask[i]值为True的地方,也就是属于该颜色类别的id组成的列表
            xmin = np.min(pos[1]) # pos[1]为x坐标，x坐标的最小值
            xmax = np.max(pos[1])
            ymin = np.min(pos[0]) # pos[0]为y坐标
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        # 将boxes转化为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # 初始化类别标签
        labels = torch.ones((num_objs,), dtype=torch.int64) # labels[1,1] (2,)的array
        masks = torch.as_tensor(masks, dtype=torch.uint8) # 将masks转换为tensor
 
        # 将图片序号idx转换为tensor
        image_id = torch.tensor([idx])
        # 计算每个边界框的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是人群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # iscrowd[0,0] (2,)的array
 
        # 生成一个字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_name"] = img_name
        # 变形transform
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        if self.get_name:
            return img, target, img_name
        else:
            return img, target
 
    def __len__(self):
        return len(self.imgs)

def get_mask_train_dataset():
    import torchvision.transforms as T
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    transforms.append(T.RandomHorizontalFlip(0.5))    
    trsf_main = T.Compose(transforms)
    train_loader = MaskRCNNDataset("train.csv", transform=trsf_main)

    return train_loader

def get_mask_test_dataset():
    import torchvision.transforms as T
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float)) 
    trsf_main = T.Compose(transforms)
    test_loader = MaskRCNNDataset("test.csv", transform=trsf_main)
    return test_loader

def name_mask_test_dataset():
    import torchvision.transforms as T
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float)) 
    trsf_main = T.Compose(transforms)
    test_loader = MaskRCNNDataset("test.csv", transform=trsf_main,get_name=True)
    return test_loader


class Kvasir(Dataset):
    #img_dataset_name: data name list csv file
    #transform: image preprocessing
    def __init__(self, img_dataset_list, transform, get_name = False):
        self.dataset_root = "../dataset/Kvasir-SEG/"
        data_dir = "../dataset/Kvasir-SEG/" + img_dataset_list
        img_dataset = np.loadtxt(open(data_dir,"rb"), dtype=str, delimiter=",",skiprows=1)
        self.file_list =  img_dataset
        self.img_label_transform = transform
        self.get_name = get_name

    # def __getitem__(self, idx):
    #     data_path = self.file_list[idx]
    #     name = os.path.splitext(data_path)[0]
    #     img_path = os.path.join(self.dataset_root, "images", data_path)
    #     label_path = os.path.join(self.dataset_root, "masks", data_path)
    #     img = Image.open(img_path).convert('RGB')
    #     label = Image.open(label_path).convert('L')
    #     img, label = self._process(img, label)
    #     return img, label, name

    def __getitem__(self, idx):
        data_path = self.file_list[idx]
        # name = os.path.splitext(data_path)[0]
        name = data_path
        img_path = os.path.join(self.dataset_root, "images", data_path)
        label_path = os.path.join(self.dataset_root, "masks", data_path)
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        img, label = self._process(img, label)
        if self.get_name:
            return img, label, name
        else:
            return img, label

    def _process(self, img, label):

        img, label = self.img_label_transform(img, label)
        return img, label

    def __len__(self):
        return len(self.file_list)


def get_kvasir_train_dataset(img_size = 256):
    trsf_main = Compose_imglabel([
        # Resize(256, 256),
        Resize(img_size, img_size),
        Random_crop_Resize(20),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        ToTensor()
    ])
    train_loader = Kvasir("train_image.csv", transform=trsf_main)

    return train_loader


def get_kvasir_test_dataset(img_size = 256):
    trsf_main = Compose_imglabel([
        # Resize(256, 256),
        Resize(img_size, img_size),
        ToTensor()
    ])
    test_loader = Kvasir("test_image.csv", transform=trsf_main)
    return test_loader

def name_kvasir_test_dataset(img_size = 256):
    trsf_main = Compose_imglabel([
        # Resize(256, 256),
        Resize(img_size, img_size),
        ToTensor()
    ])
    test_loader = Kvasir("test_image.csv", transform=trsf_main, get_name=True)
    return test_loader

if __name__ == "__main__":
    get_kvasir_test_dataset(256)

