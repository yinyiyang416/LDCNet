import argparse

parser = argparse.ArgumentParser()

# optimizer
parser.add_argument('--gpu_id', type=str, default='4', help='train use gpu')
parser.add_argument('--lr_mode', type=str, default="poly")
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--finetune_lr', type=float, default=1e-4)
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

# train schedule
parser.add_argument('--pretrain_epoches', type=int, default= 200)
parser.add_argument('--finetune_epoches', type=int, default= 100)
parser.add_argument('--log_inteval', type=int, default=50)

# data
parser.add_argument('--data_statistics', type=str,
                    default="utils/statistics.pth", help='The normalization statistics.')
parser.add_argument('--img_dataset_list', type=str,
                    default="train.csv")
parser.add_argument('--video_dataset_list', type=str,
                    default=["n088","n032","n010", "n078", "n081", "n055", "n077", "n072", "n063", "n074", "n085", "n009"])
parser.add_argument('--img_dataset_root', type=str,
                    default="/data1/zhouj/tatme_dataset/tatme_dataset/")
parser.add_argument('--video_dataset_root', type=str,
                    default="/data1/zhouj/tatme_dataset/tatme_dataset/")


parser.add_argument('--size', type=tuple,
                   default=(512, 512))


parser.add_argument('--pretrain_batchsize', type=int, default = 4)
parser.add_argument('--video_batchsize', type=int, default= 16)
parser.add_argument('--video_time_clips', type=int, default=5)
parser.add_argument('--video_testset_root', type=str, default="/data1/zhouj/tatme_dataset/tatme_dataset/images/")
parser.add_argument('--test_dataset_list', type=str,
                    default=["n076", "n071", "n047", "n021", "n051", "n002", "n027", "n043", "n065", "n020", "n014", "n029", "n005", "n030", "n040", "n068"])

# pretrain
parser.add_argument('--pretrain_state_dict', type=str, default="/data2/yyy/PNS_net/epoch_101/PNS_Pretrain.pth")
parser.add_argument('--save_path', type=str, default='/data2/yyy/PNS_net/')
config = parser.parse_args()
