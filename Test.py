import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Resize as torch_resize
import time
import torch
import numpy as np
from tqdm import tqdm

#import model
from lib.Network import use_mid_ddf_PNSNet
from lib.Network import improve_use_mid_ddf_PNSNet
from lib.Network import LDCNet

from lib.Network import test_RFB_ddfPNSNet
from lib.Network import test_transformer_ddfPNSNet


#import dataloader
from utils.dataloader import name_pretrain_test_dataset,name_video_test_dataset,name_kvasir_test_dataset,get_video_eval_dataset

#import util
from configs.test_config import config
from utils.metric import *
from utils.preprocess import *



def safe_save(img, save_path):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    img.save(save_path)

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


class eval_Test:
    #model pasth: the path to model state dict
    #save path:the path to save result image
    def __init__(self, model,model_path,save_path,mode,batch_size = 5, save_num = 0, use_cam = False):
        #save path
        self.save_path = save_path
        print("result save in %s"%self.save_path)
        self.save_num = save_num
        print("save %d picture"%(self.save_num))
        # self.use_cam = use_cam
        # if self.use_cam:
        #     self.target_layer = [model.]
        try:
            os.mkdir(self.save_path)
        except OSError:
            pass
        self.log_path = os.path.join(self.save_path,"result_log.txt")
        if os.path.exists(self.log_path):
            print("Log file exists")
            # print(time.asctime(time.localtime(time.time())))
            print_and_save(self.log_path,"***********************************************************")
            print_and_save(self.log_path,time.asctime(time.localtime(time.time())))
        else:
            if not(os.path.exists(self.save_path)):
                print("make file for result")
                os.mkdir(self.save_path)
            train_log = open(self.log_path, "w")
            # train_log.write("\n")
            # print(time.asctime(time.localtime(time.time())))
            print_and_save(self.log_path,time.asctime(time.localtime(time.time())))
            train_log.close()

        # if batch_size <= 1:
        #     bs = 2
        #     print("batch size should bigger than 1")
        # else:
        #     bs = batch_size

        self.batch_size = batch_size
        self.mode = mode
        print_and_save(self.log_path,"dataset type "+self.mode)

        #get test dataset
        if mode == "pretrain":
            test_dataset = name_pretrain_test_dataset()
            self.img_num = len(test_dataset)

            self.dataloader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=8, drop_last=True)
            print_and_save(self.log_path,"batch size " +str(self.batch_size))
        elif mode == "video":
            test_dataset = name_video_test_dataset()
            # test_dataset = get_video_eval_dataset()
            self.img_num = len(test_dataset) * 5#video中每个data包含五张图片
            print(self.img_num)
            self.batch_size = int(self.batch_size / 5) #video batch size 调小
            if self.batch_size <= 1:
                self.batch_size = 1
            # self.batch_size = self.batch_size
            self.dataloader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=8, drop_last=True)
            print_and_save(self.log_path,"batch size "+str(self.batch_size))
        elif mode == "kvasir":
            test_dataset = name_kvasir_test_dataset(img_size=512)
            # test_dataset = torch.utils.data.Subset(test_dataset,range(16))
            self.img_num = len(test_dataset)
            # print('len of dataset is ' + str(self.img_num))
            self.dataloader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=8, drop_last=True)
            # print("batch size "+str(bs))
            print_and_save(self.log_path,"batch size "+str(self.batch_size))
        # elif mode == "maskrcnn":
        #     test_dataset = name_mask_test_dataset()
        #     self.img_num = len(test_dataset)
        #     self.dataloader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=8, drop_last=True)
        #     print_and_save(self.log_path,"batch size "+str(self.batch_size))
        # elif mode == "maskrcnn_kvasir":
        #     test_dataset = name_mask_kvasir_test_dataset()
        #     self.img_num = len(test_dataset)
        #     self.dataloader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=8, drop_last=True)
        #     print_and_save(self.log_path,"batch size "+str(self.batch_size))
        else:
            assert("worng type, please input pretarin or video")

        #get device
        # gpu_id = "cuda:" + config.gpu_id
        # self.device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
        # print_and_save(self.log_path,"use " + gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id#使用os选择gpu，防止冲突
        gpu_id = "cuda:0"
        self.device = torch.device(gpu_id)
        print_and_save(self.log_path,"use cuda:" + config.gpu_id)
        #get model
        self.model = model.to(self.device)

        #get pth
        print_and_save(self.log_path,"load model from "+ model_path)
        state_dict = torch.load(model_path, map_location=torch.device(gpu_id if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()


    def test(self):
        with torch.no_grad():
            dice = []
            acc = []
            sen = []
            spec = []
            prec = []
            rec = []
            f1 = []
            iou = []
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            use_time = 0
            save_num = 0

            img_resize = torch_resize([config.size[0],config.size[1]])
            for img, gt, name in tqdm(self.dataloader, desc="test:"):
                starter.record()
                result = self.model(img.to(self.device))
                ender.record()
                torch.cuda.synchronize() # 等待GPU任务完成
                use_time += starter.elapsed_time(ender) / 1000                

                # start_time = time.time()
                # # print(img.shape)
                # result = self.model(img.to(self.device))
                # end_time = time.time()
                # # use_time += end_time - start_time#记录这一批次使用时间
                            
                # print(result.shape)
                # result = result.squeeze().cpu()#picture:[b,h,w];video:[b,t,h,w]
                # result = torch.gt(result,0.5)
                # gt = gt.detach().squeeze()#[h,w]
                # print(result[0])
                if self.mode == "maskrcnn" or self.mode == "maskrcnn_kvasir":
                    #result: list for output dict which include box,mask,scores
                    #gt: dict which include box,mask.....

                    # #maskrcnn有时候找不到目标，使用zero向量表示mask
                    # temp_reslt = []
                    # for res in result:
                    #     # print(res)
                    #     if(res["masks"].shape[0] == 0):
                    #         print("empty mask rcnn output")
                    #         temp_reslt.append(torch.zeros(res["masks"].shape[1:]).to(self.device))
                    #     else:
                    #         # print(*res["masks"].shape[1:])
                    #         # print(torch.zeros(res["masks"].shape[1:]).to(self.device).shape)
                    #         # print(res["masks"][0].shape)
                    #         temp_reslt.append(res["masks"][0])
                    # result = temp_reslt

                    result = [res["masks"][0] for res in result]#提取出maskrcnn输出中的masks,因为maskrcnn由rcnn修改，其输出中含有多个可能mask
                    result = torch.cat(result,dim=0)
                    result = torch.unsqueeze(result,1)
                    # mask.resize((self.width, self.height), Image.NEAREST)
                    # result = result.reshape(self.batch_size,1,config.size[0],config.size[1])
                    # print(result.shape)
                    gt = gt["masks"]
                    # gt = gt.reshape(self.batch_size,1,config.size[0],config.size[1])
                    # print(gt.shape)
                
                result = result.cpu()#picture:[b,1,h,w];video:[b*t,1,h,w]
                #输出结果通过softmax和gt变为0,1
                # result = torch.softmax(result,dim = 1)
                result = torch.gt(result,0.5)
                # print(result.shape)
                
                gt = gt.detach().cpu()#picture:[b,1,h,w];video:[b,t,1,h,w]
                # print(gt.shape)
                # print(name)
                for batch_num in range(result.shape[0]):#每张图片需要单独计算,所以使用batch size进行循环
                    if self.mode == "video":#video mode
                        #result:[b*t,1,h,w]
                        #gt:[b,t,1,h,w]
                        #batch_num:b_num*time_size + t_num
                        #name:[t_size,batch_size]

                        slice_result = result[batch_num,:,:,:].squeeze()#[h,w]
                        time_size = int(result.shape[0] / self.batch_size)
                        # print(time_size)
                        t_num = batch_num % time_size #t num 
                        b_num = batch_num // time_size #b num
                        # print(t_num)
                        # print(b_num)
                        silce_name = name[t_num][b_num]
                        silce_gt = gt[b_num,t_num,:,:,:].squeeze()#[h,w]
                        #save image 
                        if save_num < self.save_num:#如果保存的图像数不够
                            # print(silce_name)
                            save_img = torch.tensor(slice_result,dtype=torch.float32)
                            save_image(save_img,os.path.join(self.save_path,silce_name))

                            save_num += 1
                        
                        #保存全部图片
                        save_img = torch.tensor(slice_result,dtype=torch.float32)
                        save_image(save_img,os.path.join(self.save_path,silce_name))

                        pred = slice_result.numpy()#[h,w]
                        target = silce_gt.numpy()#[h,w]
                        acc.append(accuracy(target, pred))
                        sen.append(sensitive(target, pred))
                        spec.append(specificity(target, pred))
                        prec.append(precision(target, pred))
                        rec.append(recall(target, pred))
                        dice.append(numpy_dice_coeff(pred, target))
                        f1.append(f1_score(target,pred))
                        iou.append(IntersectionOverUnion(target,pred))

                    else:#picture mode
                        
                        slice_result = result[batch_num,:,:,:].detach().squeeze()#[h,w]
                        silce_name = name[batch_num]
                        silce_gt = gt[batch_num,:,:,:].detach().squeeze()#[h,w]
                        # save image
                        if save_num < self.save_num:#如果保存的图像数不够
                            # if self.use_cam:
                            # else:
                            #     # name = name[0]#get name
                            #     save_img = torch.tensor(slice_result,dtype=torch.float32)
                            #     save_image(save_img,os.path.join(self.save_path,silce_name))
                            #     save_num += 1

                            save_img = torch.tensor(slice_result,dtype=torch.float32)
                            save_image(save_img,os.path.join(self.save_path,silce_name))
                            save_num += 1
                        # save all image
                        # save_img = torch.tensor(slice_result,dtype=torch.float32)
                        # save_img = img_resize(save_img.unsqueeze(0))
                        # save_image(save_img,os.path.join(self.save_path,silce_name)) 

                        pred = slice_result.numpy()
                        target = silce_gt.numpy()#[h,w]
                        acc.append(accuracy(target, pred))
                        sen.append(sensitive(target, pred))
                        spec.append(specificity(target, pred))
                        prec.append(precision(target, pred))
                        rec.append(recall(target, pred))
                        dice.append(numpy_dice_coeff(pred, target))
                        f1.append(f1_score(target,pred))
                        iou.append(IntersectionOverUnion(target,pred))
            
            print_and_save(self.log_path,"total use time is {}".format(int(use_time)))
            print_and_save(self.log_path,"avg FPS is {:.3f}".format(self.img_num / use_time))
            print_and_save(self.log_path,"total img num is %d"%self.img_num)
            # print_and_save(self.log_path,tp)
            dice = np.array(dice)
            acc = np.array(acc)
            sen = np.array(sen)
            spec = np.array(spec)
            prec = np.array(prec)
            rec = np.array(rec)
            f1 = np.array(f1)
            iou = np.array(iou)
            print_and_save(self.log_path,"mean")

            print_and_save(self.log_path,"avg dice:%f+-%f"%(np.mean(dice),np.std(dice)))
            # print(dice)
            print_and_save(self.log_path,"avg Iou:%f+-%f"%(np.mean(iou),np.std(iou)))

            print_and_save(self.log_path,"avg accuracy:%f+-%f"%(np.mean(acc),np.std(acc)))
            print_and_save(self.log_path,"avg recall:%f+-%f"%(np.mean(rec),np.std(rec)))
            print_and_save(self.log_path,"avg f1:%f+-%f"%(np.mean(f1),np.std(f1)))

            print_and_save(self.log_path,"avg sensitive:%f+-%f"%(np.mean(sen),np.std(sen)))
            print_and_save(self.log_path,"avg specificity:%f+-%f"%(np.mean(spec),np.std(spec)))
            print_and_save(self.log_path,"avg precision:%f+-%f"%(np.mean(prec),np.std(prec)))
            print_and_save(self.log_path,"len of result %d"%len(dice))

            print_and_save(self.log_path,"max")

            print_and_save(self.log_path,"max dice:%f"%(np.max(dice)))
            print_and_save(self.log_path,"max Iou:%f"%(np.max(iou)))

            print_and_save(self.log_path,"max accuracy:%f"%(np.max(acc)))
            print_and_save(self.log_path,"max recall:%f"%(np.max(rec)))
            print_and_save(self.log_path,"max f1:%f"%(np.max(f1)))

            print_and_save(self.log_path,"max sensitive:%f"%(np.max(sen)))
            print_and_save(self.log_path,"max specificity:%f"%(np.max(spec)))
            print_and_save(self.log_path,"max precision:%f"%(np.max(prec)))


            print_and_save(self.log_path,"dataloader count:%d"%len(self.dataloader))
if __name__ == "__main__":
    # test
    model = LDCNet()
    pth_path = "PATH TO WEIGHT"
    save_path = "./log"
    mode = "kvasir"
    ex = eval_Test(model,pth_path,save_path,mode,1,0)
    ex.test()

