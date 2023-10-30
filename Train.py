import os
import logging
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from loguru import logger
from configs.train_config import config
from utils.loss import *

from lib.Network import LDCNet as Network



from utils.dataloader import get_pretrain_train_dataset,get_pretrain_test_dataset
from utils.dataloader import get_kvasir_train_dataset, get_kvasir_test_dataset
from utils.dataloader import get_mask_train_dataset,get_mask_test_dataset
from utils.utils import clip_gradient, adjust_lr
from utils.metric import dice_coeff
import numpy as np


def save_model(model,save_path,epoch):
    os.makedirs(os.path.join(save_path, "epoch_%d" % epoch), exist_ok=True)
    save_root = os.path.join(save_path, "epoch_%d" % epoch)
    torch.save(model.state_dict(), os.path.join(save_root, "pretrain_net.pth"))
    print("save model in %s"%save_root)

def save_best_model(model,save_path):
    torch.save(model.state_dict(), os.path.join(save_path, "best_pretrain_net.pth"))
    print("save model in %s"%save_root)


def train(train_loader, test_loader, model, optimizer, epoch, save_path, device, criterion, logger):
    """
    train function
    """
    global step
    model.to(device)
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        old_time = time.time()
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.to(device)
            gts = gts.to(device)
            preds = model(images)


            loss = criterion(preds.squeeze().contiguous(), gts.squeeze().float().contiguous().view(-1, *(gts.shape[2:])))
            loss.backward()

            # clip_gradient(optimizer, config.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 10 == 0 or i == total_step or i == 1:
                new_time = time.time()
                use_time = new_time - old_time
                # print('\ruse time[{}], Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}], Total_loss[{:.4f}]'.
                #       format(int(use_time), epoch, config.pretrain_epoches, i, total_step, loss.data), end="")
                old_time = new_time
                logger.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                    format(epoch, config.pretrain_epoches, i, total_step, loss.data))

        loss_all /= epoch_step

        count = 0
        dice_all = 0
        with torch.no_grad():
            model.eval()
            for i, (images, gts) in enumerate(test_loader, start=1):
                if(count == 500):
                    break
                images = images.to(device)
                # gts = gts.to(device).squeeze()
                preds = model(images)
                # preds = preds.squeeze()
                preds = model(images).detach().squeeze().cpu()
                gts = gts.detach().squeeze().cpu()   
                dice = dice_coeff(preds,gts)
                # dice = dice_coef_2d(preds,gts)
                dice_all += dice.cpu().numpy()
                count += 1
            dice_all /= count
        # print('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, eval_dice: {:.4f}'.format(epoch, config.pretrain_epoches, loss_all, dice_all))
        logger.info('[Testsc Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, eval_dice: {:.4f}'.format(epoch, config.pretrain_epoches, loss_all, dice_all))

        if epoch % 100 == 0:
            save_model(model,save_path,epoch)

        return loss_all.cpu().numpy(),dice_all

    except KeyboardInterrupt:
        logger.info('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_model(model,save_path,epoch+1)
        # torch.save(model.state_dict(), save_path + 'pretrain_net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        torch.cuda.empty_cache()
        raise


if __name__ == '__main__':



    expriment_name = "LDCNET"
    dataset_type = "kvasir"
    loss_type = "CE"

    model_path = None

    gpu_id = "cuda:" + config.gpu_id
    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False

    save_root = config.save_path
    save_path =  os.path.join(save_root,expriment_name)
    
    if not os.path.exists(save_path):
        print("save path is",save_path)
        os.makedirs(save_path)
    else:
        assert("save path exist,please check save path")
    #定义log保存位置
    log_path = os.path.join(save_path,"pretrain_log.log")
    # 添加log文件位置
    logger.add(log_path)
    logger.info("USE GPU %s"%gpu_id)
    # print("USE GPU %s"%gpu_id)


    #model
    print('Try to load model')
    model = Network().to(device)
    if model_path is not None:
        model.load_backbone(torch.load(model_path, map_location=torch.device('cpu')), logging)
        logger.info('load model from ', model_path)
        print("sussess load checkpoint")
 

    # RMSprop
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)
    logger.info("use RMSprop optimizer")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # logger.info("use AdamW optimizer")
    # from torch.optim.lr_scheduler import StepLR, ExponentialLR
    # from warmup_scheduler import GradualWarmupScheduler
    # # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optimizer, step_size=200, gamma=0.1)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_steplr)
    
    #loss fuction
    if loss_type == "CE":
        criterion = CrossEntropyLoss()
    elif loss_type == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "DICE_BCE":
        criterion = DiceBCELoss()
    else:
        Warning("wrong loss fuction name")
    

    # dataset
    # print('load data...')
    if dataset_type == "kvasir":
        train_loader = get_kvasir_train_dataset(config.size[0])
        test_loader = get_kvasir_test_dataset(config.size[0])
        logger.info("image reszie size is %d"%config.size[0])
        logger.info("use Kvasir dataset")
    elif dataset_type == "tatme":
        test_loader = get_pretrain_test_dataset()
        train_loader = get_pretrain_train_dataset()
        logger.info("use TATME dataset")
    elif dataset_type == "maskrcnn":
        test_loader = get_mask_test_dataset()
        train_loader = get_mask_train_dataset()
        logger.info("use MASK TATME dataset")
    else:
        Warning("worning dataset type")

    logger.info("train data number is %d"%train_loader.__len__())
    # print("train data number is %d"%train_loader.__len__())
    train_loader = data.DataLoader(dataset=train_loader,
                                   batch_size = config.pretrain_batchsize,
                                   shuffle=True,
                                   num_workers = 1,
                                   pin_memory=False,
                                   drop_last=True
                                   )
    total_step = len(train_loader)
    
    

    logger.info("test data number is %d"%test_loader.__len__())
    # print("test data number is %d"%test_loader.__len__())
    test_loader = data.DataLoader(dataset=test_loader,
                                   batch_size = 2,
                                   shuffle=False,
                                   num_workers = 1,
                                   pin_memory=False,
                                   drop_last=True)

    # logging
    # logging.basicConfig(filename=save_path + 'log.log',
    #                     format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    #                     level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logger.info("Network-Train")
    logger.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}'.format(config.pretrain_epoches, config.base_lr, config.video_batchsize, config.size, config.clip,
                                                         config.decay_rate, config.save_path, config.decay_epoch))



    step = 0
    best_loss = np.inf
    best_epoch = 0
    loss_list = []
    dice_list = []
    print("Start",expriment_name, "train...")
    epoch = 0
    print("start epoch",epoch)
    logger.info("start epoch",epoch)
    # for epoch in range(config.pretrain_epoches+1):
    while(epoch < config.pretrain_epoches + 1):
        epoch += 1
        # scheduler_warmup.step(epoch)
        # cur_lr = adjust_lr(optimizer, config.base_lr, epoch, config.decay_rate, config.decay_epoch)
        loss ,dice= train(train_loader, test_loader, model, optimizer, epoch, save_path, device, criterion, logger)
        if loss <= best_loss:
            best_epoch = epoch
            best_loss = loss
            logger.critical(f"saving checkpoint at {epoch}")
            save_best_model(model,save_path)
        print("train epoch done")
        # logger.info(f"| epoch : {epoch} | training done | loss: {loss} | dice: {dice}  best loss: {best_loss / (epoch + 1)} |")
        loss_list.append(loss)
        dice_list.append(dice)
        # print("save loss")
        np.save(os.path.join(save_path,"pretrain_loss_list.npy"),loss_list)
        # print("save dice")
        np.save(os.path.join(save_path,"pretrain_dice_list.npy"),dice_list)

    print("best epoch",best_epoch)
    save_model(model,save_path,epoch)
    print("save loss")
    np.save(os.path.join(save_path,"pretrain_loss_list.npy"),loss_list)
    print("save dice")
    np.save(os.path.join(save_path,"pretrain_dice_list.npy"),dice_list)
