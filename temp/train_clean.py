#!/usr/bin/env python
# coding: utf-8


# importing necessary modules and functions
import os
import random
import argparse
from mindspore import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.optim import Adam
from mindspore.nn.optim import SGD
from mindspore import Model, context
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from models.resnet import resnet18
import numpy as np
import argparse
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset as ds
from mindspore import dtype as mstype


def create_dataset(repeat_num=1, training=True, batch_size=64):
    """
    create data for next use such as training or inferring

    """
    
    if training == True:
        usage = 'train'
    else:
        usage = 'test'
    
    
    #importing the cifar10 dataset and assigning number of parallel workers as 8
    cifar_ds = ds.Cifar10Dataset('data/cifar-10-batches-bin/', usage = usage, num_parallel_workers = 8)

    resize_height = 32
    resize_width = 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = C.RandomHorizontalFlip() # flips the image horizontally 
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift) # rescales the input image
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalize the array with given mean and standard deviation
    changeswap_op = C.HWC2CHW() # transforms H,W,C to C,H,W
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)

    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_num)

    return cifar_ds


# for receiving arguments in CLI
ap = argparse.ArgumentParser()


ap.add_argument("-bsize", "--batchsize", default = 64,
   help="batch size")
ap.add_argument("-repeatNum", "--repeatNum", default = 1,
   help="repeat num")
ap.add_argument("-dir", "--savedir", default = 'new_check/',
   help="save directory")
ap.add_argument("-e", "--epoch", default = 100,
   help="no of epochs")
ap.add_argument("-opt", "--optimizer", default = 'Adam',
   help="optimizer")
ap.add_argument("-lr", "--learningRate", default = 0.01,
   help="learning rate")
ap.add_argument("-m", "--momentum", default = 0.9,
   help="momentum")
ap.add_argument("-wDecay", "--weightDecay", default = 1e-4,
   help="optimizer")

args = vars(ap.parse_args())



mode = 1
if mode == 1:
## PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
if mode == 2:
## Graph MODE ## Cant do since we are working with only one GPU
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", enable_graph_kernel=True)



batch_size = int(args['batchsize'])
repeat_num = int(args['repeatNum'])
num_classes = 10



from mindspore.train.callback import Callback

class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(acc["acc"])
            print('\n',acc, '\n')
            
            
class LossMonitor(Callback):
    def __init__(self, per_print_times=1):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            if cb_params.cur_step_num % 100 == 0:
                print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)



ckpt_save_dir = args['savedir']
eval_per_epoch = 1
epoch_size = args['epoch']
epoch_per_eval = {"epoch": [], "acc": []}



net = resnet18(num_classes)
ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
lr = args['learningRate']
weight_decay = args['weightDecay']
momentum = args['momentum']

optimizer = args['optimizer']

if optimizer == 'Momentum':
    opt = Momentum(net.trainable_params(),lr, momentum)
if optimizer == 'Adam':
    opt = Adam(net.trainable_params(),lr, weight_decay)
if optimizer == 'SGD':
    opt = SGD(net.trainable_params(),lr, weight_decay)
    
# quantizer = QuantizationAwareTraining(bn_fold=False)
# quant = quantizer.quantize(net)

model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

train = create_dataset()
eval_data = create_dataset(training=False)

loss_cb = LossMonitor()
time = TimeMonitor()

eval_cb = EvalCallBack(model, eval_data, eval_per_epoch, epoch_per_eval)
config_ck = CheckpointConfig(save_checkpoint_steps=15625, keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix="train_resnet_cifar10", directory=ckpt_save_dir, config=config_ck)


model.train(epoch_size, train, callbacks=[ckpoint_cb, loss_cb, eval_cb, time])


