#-*- coding: utf-8 -*-
import argparse
import os

def str2bool(v):
  return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--input_height', type=int, default=75)
data_arg.add_argument('--input_width', type=int, default=75)
data_arg.add_argument('--input_channel', type=int, default=3)
#data_arg.add_argument('--train_data_dir', type=str, default="/data1/jlogan/data/image")
#data_arg.add_argument('--train_mask_dir', type=str, default="/data1/jlogan/data/mask")
#data_arg.add_argument('--eval_data_dir', type=str, default="/data1/jlogan/data/image")
#data_arg.add_argument('--eval_mask_dir', type=str, default="/data1/jlogan/data/mask")
data_arg.add_argument('--train_data_dir', type=str, default="/mnt/bb/lot/data/image")
data_arg.add_argument('--train_mask_dir', type=str, default="/mnt/bb/lot/data/mask")
data_arg.add_argument('--eval_data_dir', type=str, default="/mnt/bb/lot/data/image")
data_arg.add_argument('--eval_mask_dir', type=str, default="mnt/bb/lot/data/mask")
data_arg.add_argument('--real_image_dir', type=str, default="../../segmentation-tensorflow/data/nuclei/real_75")

# Training
train_arg = add_argument_group('Training')
train_arg.add_argument('--batch_size', type=int, default=68, help='')
#train_arg.add_argument('--learner_learning_rate', type=float, default=0.03, help='')
train_arg.add_argument('--learner_learning_rate', type=float, default=0.003, help='')
train_arg.add_argument('--optimizer', type=str, default='moment', choices=['adam', 'moment', 'sgd'], help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--num_epochs', type=int, default=500, help='')
train_arg.add_argument('--image_count', type=int, default=11792, help='')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--model_dir', type=str, default='/home/lot/scratch/models/original')




###BELOW NOT USED###

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--kernel_dims', type=eval, default='[]', help='')
net_arg.add_argument('--stride_size', type=eval, default='[]', help='')
net_arg.add_argument('--channel_dims', type=eval, default='[]', help='')
net_arg.add_argument('--with_ref', type=str2bool, default=True, help='')
net_arg.add_argument('--learner_global_texture', type=str2bool, default=True, help='')
net_arg.add_argument('--with_batch_norm', type=str2bool, default=False, help='')
net_arg.add_argument('--refiner_dense_bias', type=str2bool, default=False, help='')
net_arg.add_argument('--resnet_refiner', type=str2bool, default=False, help='')

data_arg.add_argument('--data_set', type=str, default='nuclei')
data_arg.add_argument('--data_dir', type=str, default='data')
data_arg.add_argument('--max_synthetic_num', type=int, default=-1)



# Training / test parameters
train_arg.add_argument('--reg_scale_l1', type=float, default=0.01, help='')
train_arg.add_argument('--reg_scale_l2', type=float, default=0.1, help='')
train_arg.add_argument('--real_scale', type=float, default=0.0, help='')
train_arg.add_argument('--learner_adv_scale', type=float, default=0.0, help='')
train_arg.add_argument('--initial_K_g', type=int, default=1000, help='')
train_arg.add_argument('--initial_K_d', type=int, default=100, help='')
train_arg.add_argument('--initial_K_l', type=int, default=100, help='')
train_arg.add_argument('--max_step_d_g', type=int, default=100, help='')
train_arg.add_argument('--max_step_d_g_l', type=int, default=30000, help='')
train_arg.add_argument('--after_K_l', type=int, default=0, help='')
train_arg.add_argument('--K_d', type=int, default=0, help='')
train_arg.add_argument('--K_g', type=int, default=2, help='')
train_arg.add_argument('--K_l', type=int, default=4, help='')

train_arg.add_argument('--buffer_size', type=int, default=25600, help='')
train_arg.add_argument('--refiner_learning_rate', type=float, default=0.00001, help='')
train_arg.add_argument('--discrim_learning_rate', type=float, default=0.00001, help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')
train_arg.add_argument('--max_grad_norm', type=float, default=50, help='')
train_arg.add_argument('--pred_step_size', type=int, default=4, help='')
train_arg.add_argument('--pred_scaling', type=float, default=1.0, help='')
train_arg.add_argument('--pred_gkern_sig', type=float, default=0.5, help='')

# Misc
misc_arg.add_argument('--log_step', type=int, default=48, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--sample_dir', type=str, default='samples')
misc_arg.add_argument('--output_dir', type=str, default='outputs')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--debug', type=str2bool, default=False)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
misc_arg.add_argument('--max_image_summary', type=int, default=32)
misc_arg.add_argument('--sample_image_grid', type=eval, default='[10, 10]')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed
