import os
import time
import argparse

parser = argparse.ArgumentParser(description='DevianceNet')

# Dataset
parser.add_argument('--train_folder_directory',
                    type=str,
                    default=r'/workspace/data/',
                    help='train directory')
parser.add_argument('--SEA_folder_directory',
                    type=str,
                    default=r'/workspace/data/',
                    help='SEA directory')
parser.add_argument('--DIA_folder_directory',
                    type=str,
                    default=r'/workspace/data/',
                    help='DIA directory')
parser.add_argument('--img_y',
                    type=int,
                    default=224,
                    help='height')
parser.add_argument('--img_x',
                    type=int,
                    default=224,
                    help='width')

# Environment
parser.add_argument('--experiment_description',
                    type=str,
                    help='Description of Experiment')
parser.add_argument('--cpu',
                    action='store_true',
                    default=False,
                    help='Description of Experiment')
parser.add_argument('--dropout',
                    type=float,
                    default=0.0,
                    help='learning rate')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size',
                    type=int,
                    default=7,
                    help='input batch size for training')

parser.add_argument('--lambda_1',
                    type=float,
                    default=0.5,
                    help='learning rate')
parser.add_argument('--lambda_2',
                    type=float,
                    default=0.15,
                    help='learning rate')
parser.add_argument('--h_1',
                    type=float,
                    default=300/330,
                    help='heinrich loss h1')
parser.add_argument('--h_2',
                    type=float,
                    default=29/330,
                    help='heinrich loss h2')
parser.add_argument('--h_3',
                    type=float,
                    default=1/330,
                    help='heinrich loss h3')

parser.add_argument('--frame_num',
                    type=int,
                    default=16,
                    help='input frame number')
parser.add_argument('--superpoint_type',
                    type=str,
                    default='con',
                    choices=('con', 'mul', 'None'),
                    help='superpoint attention type')
parser.add_argument('--superpoint_freeze',
                    action='store_true',
                    default=False,
                    help='Description of Experiment')
parser.add_argument('--non_local',
                    action='store_true',
                    default=False,
                    help='Description of Experiment')
parser.add_argument('--classifier_type',
                    type=str,
                    choices=('SEA', 'DIA', 'SEA_DIA'),
                    help='Classifier Type')
parser.add_argument('--korea_total',
                    action='store_true',
                    default=False,
                    help='seoul total')
parser.add_argument('--model',
                    type=str,
                    default='DevianceNet',
                    help='Classifier Type')
parser.add_argument('--partition',
                    type=int,
                    default=0,
                    help='train partition %')
parser.add_argument('--direction',
                    type=str,
                    default=False,
                    help='train partition %')


# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.0001,
                    help='learning rate')
parser.add_argument('--optimizer',
                    default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')

# Hardware
parser.add_argument('--seed',
                    type=int,
                    default=777,
                    help='random seed point')
parser.add_argument('--num_threads',
                    type=int,
                    default=4,
                    help='number of threads')

# Save
parser.add_argument('--score_min_CL',
                    type=float,
                    default=0.3,
                    help='SEA evaluation')
parser.add_argument('--score_min_DE',
                    type=float,
                    default=0.7,
                    help='DIA evaluation')
#superpoint
parser.add_argument('--num_kps',
                    type=int,
                    default=256,
                    help='top k keypoints')
parser.add_argument('--detection_threshold',
                    type=float,
                    default=0.0005,
                    help='Detection threshold')
parser.add_argument('--align_corners',
                    action='store_true',
                    default=False,
                    help='align corners')
parser.add_argument('--frac_superpoint',
                    type=float,
                    default=.5,
                    help='Detection threshold')
parser.add_argument('--nms_radius',
                    type=int,
                    default=9,
                    help='radius nms')

#test
parser.add_argument('--test_only',
                    action='store_true',
                    default=False,
                    help='test_only')
parser.add_argument('--test_metric',
                    type=str,
                    choices=('SEA','DIA'),
                    help='test metric')
parser.add_argument('--test_fold_num',
                    type=int,
                    default=None,
                    help='testing fold')
# pretrain
parser.add_argument('--weight_load_pth',
                    type=str,
                    default=None,
                    help='weight file')
parser.add_argument('--optimizer_load_pth',
                    type=str,
                    default=None,
                    help='optimizer file')
parser.add_argument('--resume',
                    action='store_true',
                    default=False,
                    help='retrain')


args = parser.parse_args()

now = time.localtime()
date = "%04d%02d%02d_%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
args.save_model_path = './save_model/' + date + '_' + args.experiment_description
args.output_path = './outputs/' + date + '_' + args.experiment_description
