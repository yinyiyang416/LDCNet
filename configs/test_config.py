import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
# data
parser.add_argument('--data_statistics', type=str,
                    default="utils/statistics.pth", help='The normalization statistics.')
parser.add_argument('--img_dataset_list', type=str,
                    default="train.csv")

parser.add_argument('--img_dataset_root', type=str,
                    default="PATH TO DATASET")

parser.add_argument('--size', type=tuple,
                   default=(512, 512))

parser.add_argument('--model_path', type=str,
                    default="PATH TO train model")
                    
parser.add_argument('--save_path', type=str, default='./run')
config = parser.parse_args()
