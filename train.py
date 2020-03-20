# -*- coding: utf-8 -*-

from __future__ import print_function, division
from database import AllData,SingleData

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
from model import SelfMatchNet, SegMaskNet
from random_erasing import RandomErasing
import yaml
import math
from shutil import copyfile
from tqdm import tqdm
import models.SVHNet as svhnet

version =  torch.__version__
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='SelfMatchNet', type=str, help='output model name')
parser.add_argument('--train_data_root',default='/home/honglie.whl/data/VeRi/train.csv',type=str, help='training dir path')
parser.add_argument('--val_data_root',default='/home/honglie.whl/data/VeRi/train.csv',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=1, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
opt = parser.parse_args()

name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

transform_train_list = [
        transforms.Resize((234,234), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((234,234)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize((234,234), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]


data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_datasets = SingleData(opt.train_data_root, opt.batchsize, transform=transform_train_list)
val_datasets = SingleData(opt.val_data_root, opt.batchsize, transform=transform_val_list)


use_gpu = torch.cuda.is_available()

train_data_loader = DataLoader(dataset=train_datasets,
                                num_workers=3,
                                batch_size=opt.batchsize,
                                shuffle=True)
val_data_loader = DataLoader(dataset=val_datasets,
                                num_workers=3,
                                batch_size=opt.batchsize,
                                shuffle=True)

dataloaders = {
        'train':train_data_loader,
        'val':val_data_loader,
        }

since = time.time()
print(time.time()-since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    # warm_up = 0.1 # We start from the 0.1*lrRate
    # warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    for epoch in range(epochs):
        print('Steps {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_corrects = 0.0
            for data in tqdm(dataloaders[phase]):
                x,y,x_l,y_l,target = data
                now_batch_size,c,h,w = x.shape
                # if now_batch_size < opt.batchsize:
                #     continue
                x,y,x_l,y_l,target = x.cuda(),y.cuda(),x_l.cuda(),y_l.cuda(),target.cuda()
                optimizer.zero_grad()
                if phase == 'val':
                    with torch.no_grad():
                        out_x,_,_ = model(x,y)
                        _, preds = torch.max(out_x.data,1)
                        loss = criterion(out_x, x_l)
                else:
                    out_x,out_y,out_match = model(x,y)
                    _,preds_x = torch.max(out_x.data,1)
                    _,preds_match = torch.max(out_match.data,1)
                    loss1 = criterion(out_x, x_l)
                    loss2 = criterion(out_match,target)
                    loss = loss2
                    loss.backward()
                    # print('loss:',loss)
                    print(model.match.fc.weight.grad)
                    optimizer.step()
                running_loss1 += loss1.item() * now_batch_size
                running_loss2 += loss2.item() * now_batch_size
                # running_corrects += float(torch.sum(preds_x == x_l.data))
                running_corrects += float(torch.sum(preds_match == target.data))
                # print(f'batch_loss1:{loss1.item()},batch_loss2:{loss2.item()}')

            epoch_loss1 = running_loss1 / len(train_datasets)
            epoch_loss2 = running_loss2 / len(train_datasets)
            epoch_acc = running_corrects / len(train_datasets)

            print('{} Loss1: {:.4f} Loss2:{:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss1, epoch_loss2, epoch_acc))

            y_loss[phase].append(epoch_loss1+epoch_loss2)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(last_model_wts)
    # save_network(model, 'last')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

model = SelfMatchNet(776)
# model = svhnet.STNSVHNet((256,256),3,3,5,num_classes=576)

# print(model)

optimizer_ft = optim.SGD([
         {'params': model.parameters(), 'lr': 0.1*opt.lr}
     ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()

criterion = nn.CrossEntropyLoss()

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       epochs=100)

