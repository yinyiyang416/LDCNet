from thop import profile
from thop import clever_format
import torch
from torch.utils.data import DataLoader
from ..configs.test_config import config
import os
import time
import numpy as np
import tqdm
from ptflops import get_model_complexity_info


#import model
from ..lib.Network import LDCNet


#import data
from utils.dataloader import get_pretrain_test_dataset,get_video_test_dataset,image_eval_para,video_eval_para

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def eval_parm_layer(model,input, batch_size = 1):

    input = input.squeeze()
    flops, params = get_model_complexity_info(model, (3,256,256),as_strings=True,print_per_layer_stat=True)
    print("flops:"+flops)
    print("params:"+params)
    return flops, params


class test_ptflops(torch.nn.Module):
    def __init__(self):
        super(test_ptflops, self).__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=3, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        # x = torch.add(torch.zeros_like(x),x)
        return x

def eval_parm(model,input,repetitions = 300, batch_size = 1,use_ptflots = True):
    #计算模型参数量，同时进行预热
    if use_ptflots:
        if len(input.shape)  == 4:
            macs, params = get_model_complexity_info(model, (3,256,256),as_strings=True,print_per_layer_stat=True)
        else:
            macs, params = get_model_complexity_info(model, (5,3,256,256),as_strings=True,print_per_layer_stat=True)
    else:
        macs, params = profile(model, inputs=(input, ))
        macs, params = clever_format([macs, params], "%.3f")
    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        for rep in range(repetitions):
            # start = time.time()
            starter.record()
            _ = model(input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            # end2 = time.time()
            # use_time_2 = end2 - start
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            # print("test diff time ")
            # print(use_time_2*1000)
            # print(curr_time)
            timings[rep] = curr_time
    
    use_time_sum = timings.sum()
    use_time_avg = use_time_sum/repetitions
    
    FPS_row = repetitions * batch_size / use_time_sum * 1000
    return macs, params, str(use_time_avg), FPS_row

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    bs = 5

    image_set = get_pretrain_test_dataset()

    image_set = DataLoader(image_set, batch_size = bs, shuffle=False, num_workers=0, drop_last=True)
    for image_input,_ in image_set:
        break
    image_input = image_input.cuda()

    video_set = get_video_test_dataset()

    video_set = DataLoader(video_set, batch_size = 1, shuffle=False, num_workers=0, drop_last=True)
    for video_input,_ in video_set:
        break
    video_input = video_input.cuda()


    
    root_path = "./result/"
    log_path = os.path.join(root_path,"model_size_review_log.txt")
    if os.path.exists(log_path):
        print("Log file exists")
        print_and_save(log_path,"***********************************************************")
        print_and_save(log_path,"use cuda "+config.gpu_id)
        print_and_save(log_path,time.asctime(time.localtime(time.time())))
        print_and_save(log_path,"use ptflops")
    else:

        train_log = open(log_path, "w")
        print_and_save(log_path,time.asctime(time.localtime(time.time())))
        train_log.close()
    
    print_and_save(log_path,"image size: "+str(config.size[0])+str(config.size[1]))
    print_and_save(log_path,"video clips(image batch size): "+str(bs))

    #PNSnet pretrain
    model = LDCNet().cuda()
    macs, params, Latency, Fps = eval_parm(model,image_input,batch_size = bs)
    print_and_save(log_path,"model: LDCNet")
    print_and_save(log_path,"Macs: " + macs + "; Parms: " + params +"; Latency(ms): " + Latency+"; FPS: " + str(Fps))





