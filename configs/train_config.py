import argparse

parser = argparse.ArgumentParser()

# optimizer
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
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
parser.add_argument('--img_dataset_root', type=str,
                    default="PATH TO DATASET")


parser.add_argument('--size', type=tuple,
                   default=(512, 512))

parser.add_argument('--save_path', type=str, default='./run')
config = parser.parse_args()
