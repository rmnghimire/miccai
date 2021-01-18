import gc
import os
import random
import shutil
import argparse
from model import UNet
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms
from tqdm import tqdm
from dataloader import random_seed, train_loader, val_loader, batch_size,unlabeled_set
from loss import DiceBCELoss
import models.network as models
from mean_teacher import losses,ramps
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchtoolbox.tools import mixup_data, mixup_criterion
from lib import transforms_for_rot, transforms_back_rot, transforms_for_noise, transforms_for_scale, transforms_back_scale, postprocess_scale,Cutout
from utils.utils import multi_validate, update_ema_variables

parser = argparse.ArgumentParser(description='PyTorch Miccai Training')
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Checkpoints
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=800,
                    help='Number of labeled data')
parser.add_argument('--val_iteration', type=int, default=80,
                    help='Number of labeled data')
parser.add_argument('--data', default='',
                    help='input data path')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--num-class', default=1, type=int)
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--unsupervised', action="store_true")
parser.add_argument('--wlabeled', action="store_true")
parser.add_argument('--patience', action="store_true")
parser.add_argument('--baseline', action="store_true")
parser.add_argument('--test_mode', action="store_true")
# lr
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--lr", default=0.03, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)

#
parser.add_argument('--consistency_type', type=str, default="mse")
parser.add_argument('--consistency', type=float,  default=10.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=400.0, help='consistency_rampup')

#
parser.add_argument('--initial-lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

from shutil import copyfile

def train(model,ema_model,train_dataloader, val_dataloader,unlabeled_loader, batch_size, num_epochs, learning_rate, patience, model_path, device,criterion,optimiser):
    """
    Function to train a u-net model for segmentation.
    model: U-Net model
    Args:
        train_dataloader: training set
        val_dataloader: validation set
        batch_size: batch_size for training.
        num_epochs: number of epochs to train.
        learning_rate: learning rate for the optimiser
        patience: number of epochs for early stopping.
        model_path: checkpoint path to store the model.
        device: CPU or GPU to train the model.

    Returns: A dictionary containing the training and validation losses.
    """

    # Loss Collection
    train_losses = []
    val_losses = []
    mean_squared_loss = []

    writer = SummaryWriter("runs/" + str(args.out.split("/")[-1]))
    writer.add_text('Text', str(args))
    count = 0

    for epoch in tqdm(range(1, args.epochs + 1)):
        current_train_loss = 0.0
        current_val_loss = 0.0
        Mean_square_error = 0.0

        #Test the Model
        if (epoch) % 50 == 0:
            val_loss, val_result = multi_validate(val_loader, model, criterion, epoch, use_cuda, args)
            test_loss, val_ema_result = multi_validate(val_loader, ema_model, criterion, epoch, use_cuda, args)

            step =  args.val_iteration * (epoch)

            writer.add_scalar('Val/loss', val_loss, step)
            writer.add_scalar('Val/ema_loss', test_loss, step)

            writer.add_scalar('Model/JA', val_result[0], step)
            writer.add_scalar('Model/AC', val_result[1], step)
            writer.add_scalar('Model/DI', val_result[2], step)
            writer.add_scalar('Model/SE', val_result[3], step)
            writer.add_scalar('Model/SP', val_result[4], step)


            writer.add_scalar('Ema_model/JA', val_ema_result[0], step)
            writer.add_scalar('Ema_model/AC', val_ema_result[1], step)
            writer.add_scalar('Ema_model/DI', val_ema_result[2], step)
            writer.add_scalar('Ema_model/SE', val_ema_result[3], step)
            writer.add_scalar('Ema_model/SP', val_ema_result[4], step)
            # scheduler.step()

            # save model
            big_result = max(val_result[0], val_ema_result[0])
            is_best = big_result > best_acc
            best_acc = max(big_result, best_acc)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_result[0],
                'best_acc': best_acc,
                'optimizer': optimiser.state_dict(),
            }, is_best)

        #Train model

        global global_step

        if args.consistency_type == 'mse':
            consistency_criterion = losses.softmax_mse_loss
        elif args.consistency_type == 'kl':
            consistency_criterion = losses.softmax_kl_loss
        else:
            assert False, args.consistency_type

        # switch to train mode
        model.train()
        ema_model.train()
        for batch_idx in range(args.val_iteration):
            try:
                inputs_x, targets_x, a = train_dataloader.next()
            except:
                labeled_train_iter = iter(train_dataloader)
                inputs_x, targets_x, a = labeled_train_iter.next()
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

            if args.unsupervised:
                try:
                    inputs_u = unlabeled_loader.next()
                except:
                    unlabeled_train_iter = iter(unlabeled_loader)
                    inputs_u, b = unlabeled_train_iter.next()

                inputs_u_noise = Cutout(inputs_u,1,16) #Applying Cutout regularization technique

                if use_cuda:
                    # targets_x[targets_x == 255] = 1
                    inputs_u = inputs_u.cuda()
                    inputs_u_noise = inputs_u_noise.cuda()

            #inputs_u_noise = torch.from_numpy(inputs_u_noise).float().to(device)
            iter_num = batch_idx + epoch * 100  # Validation iteration

            # Student model predictions
            output_x = model.forward(inputs_x)
            supervised_loss = criterion(output_x, targets_x)

            if args.unsupervised:
                # Teacher model predictions
                consistency_weight = get_current_consistency_weight(epoch)
                outputs_u1 = ema_model(inputs_u)
                outputs_u2 = ema_model(inputs_u_noise)

                consistency_dist = consistency_criterion(outputs_u1, outputs_u2)
                consistency_dist = torch.mean(consistency_dist)
                Unsupervised_loss = consistency_weight * consistency_dist
                Mean_square_error += Unsupervised_loss.item()
                #print(Mean_square_error)

                Total_loss = supervised_loss + Unsupervised_loss
            else:
                Total_loss = supervised_loss

            # print('*** TOTAL LOSS')
            # print(Total_loss)
            current_train_loss += Total_loss.item()

            optimiser.zero_grad()
            Total_loss.backward()
            optimiser.step()
            update_ema_variables(model, ema_model, 0.999, iter_num)

            writer.add_scalar('losses/train_loss', Total_loss, iter_num)
            writer.add_scalar('losses/train_loss_supervised', supervised_loss, iter_num)
            if args.unsupervised:
                writer.add_scalar('losses/train_loss_un', Unsupervised_loss, iter_num)
                writer.add_scalar('losses/consistency_weight', consistency_weight, iter_num)


        # model.train()
        # for features, labels, idx in train_dataloader:
        #     print(len(train_dataloader))
        #     optimiser.zero_grad()
        #     features, labels = features.to(device), labels.to(device)
        #     output = model.forward(features)
        #     loss = criterion(output, labels)
        #     loss.backward()
        #     optimiser.step()
        #     current_train_loss += loss.item()
        #
        #     del features, labels
        #     gc.collect()
        #     torch.cuda.empty_cache()

        #Evaluate model
        model.eval()
        with torch.no_grad():
            for features, labels, idx in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                output = ema_model.forward(features)
                loss = criterion(output, labels)
                current_val_loss += loss.item()

                # del features, labels
                # gc.collect()
                # torch.cuda.empty_cache()

        # Store Losses
        current_train_loss /= len(train_dataloader)
        train_losses.append(current_train_loss)

        current_val_loss /= len(val_dataloader)
        val_losses.append(current_val_loss)

        Mean_square_error /= len(train_dataloader)
        mean_squared_loss.append(Mean_square_error)

        print("Epoch: {0:d} -> Train Loss: {1:0.8f} Val Loss: {2:0.8f} Mean squared error Loss: {2:0.8f} ".format(epoch, current_train_loss,
                                                                                current_val_loss, mean_squared_loss))
        if ((epoch == 1) or (current_val_loss < best_val_loss)):
            best_val_loss = current_val_loss
            eq_train_loss = current_train_loss
            best_epoch = epoch
            count = 0

            # Save best model
            torch.save(ema_model.state_dict(), model_path)

        # Check for patience level
        if (current_val_loss > best_val_loss):
            count += 1
            if (count == patience):
                break

    # Save best parameters
    best_model_params = {'train_losses': train_losses,
                         'val_losses': val_losses,
                         'best_val_loss': best_val_loss,
                         'eq_train_loss': eq_train_loss,
                         'best_epoch': best_epoch}

    return best_model_params

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 10.0 * ramps.sigmoid_rampup(epoch, 400.0)
def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def main():
    output_dir = "experiment_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "polyp_Teacher_Student.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initiliase Model
    torch.manual_seed(random_seed)
    #model = UNet(n_channels = 3, n_classes = 1, bilinear = False).to(device)
    print("==> creating model")



    def create_model(ema=False):
        #model = models.DenseUnet_2d()
        model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)
    # Loss function
    criterion = DiceBCELoss().to(device)
    # Optimiser
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Hyperparameters
    num_epochs = args.epochs
    learning_rate = args.lr
    patience = args.patience

    #Resume
    # Resume
    # if args.resume:
    #     print('==> Resuming from checkpoint..' + args.resume)
    #     assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load(args.resume)
    #     best_acc = checkpoint['best_acc']
    #     print("epoch ", checkpoint['epoch'])
    #     model.load_state_dict(checkpoint['state_dict'])
    #     ema_model.load_state_dict(checkpoint['ema_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        val_loss, val_result = multi_validate(val_loader, ema_model, criterion, 0, use_cuda, mode='Valid Stats')
        print("val_loss", val_loss)
        print("Val ema_model : JA, AC, DI, SE, SP \n")
        print(", ".join("%.4f" % f for f in val_result))
        val_loss, val_result = multi_validate(val_loader, model, criterion, 0, use_cuda, mode='Valid Stats')
        print("val_loss", val_loss)
        print("Val model: JA, AC, DI, SE, SP \n")
        print(", ".join("%.4f" % f for f in val_result))
        return

    # Train model
    best_model_params = train(model, ema_model,train_loader, val_loader,unlabeled_set, batch_size, num_epochs,
                              learning_rate, patience, save_path, device,criterion,optimizer)

    print("Training complete.")

    # Delete model to free memory
    del model, best_model_params
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()