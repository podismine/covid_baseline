from cProfile import label
from genericpath import exists
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import logging
import random
import numpy as np
import warnings
from args import get_parser
from dataset.data import data_prefetcher, AllData
from models.i3d import I3D
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils import reduce_mean,adjust_learning_rate, AverageMeter, ProgressMeter

import torch.nn.functional as F

def initialize():
    # get args
    args = get_parser()

    # warnings
    warnings.filterwarnings("ignore")

    # logger
    logger = logging.getLogger(__name__)

    # set seed
    seed = int(1111)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # initialize logger
    logger.setLevel(level = logging.INFO)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    handler = logging.FileHandler("logs/%s.txt" % args.env_name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return args, logger

def main():
    config, logger = initialize()
    config.nprocs = torch.cuda.device_count()
    main_worker(config, logger)

def main_worker(config, logger):

    best_acc1 = 0.0

    # create model
    model = I3D(nr_outputs=2)
    ########################################################################################################
    torch.cuda.set_device(config.local_rank)
    model.cuda()
    config.batch_size = int(config.batch_size / config.nprocs)
    
    optimizer = torch.optim.Adam(model.parameters(),lr = config.lr,weight_decay = 0.00005)

    cudnn.benchmark = True

    # Data loading code
    train_data = AllData(config.data, train = True)
    val_data = AllData(config.data, train = False)


    train_loader = DataLoader(train_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = True)
    val_loader = DataLoader(val_data,config.batch_size,
                        shuffle=False,num_workers=2,pin_memory = True)

    for epoch in range(config.epochs):

        #adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, config, logger)
        
        acc1, acc2 = validate(val_loader, model, config, logger)
        #break
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)
        if not os.path.exists("./checkpoints/%s" % config.env_name):
            try:
                os.makedirs("./checkpoints/%s" % config.env_name)
            except:
                pass # multiple processors bug

        if is_best:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                }
            torch.save(state, './checkpoints/%s/%s_epoch_%s_%s' % (config.env_name, config.env_name, acc1, acc2))

def train(train_loader, model , optimizer, epoch, config, logger):
    losses = AverageMeter('Loss', ':.4e')
    loss_acc1 = AverageMeter('acc1', ':6.2f')
    loss_acc2 = AverageMeter('acc2', ':6.2f')

    progress = ProgressMeter(len(train_loader), [losses, loss_acc1, loss_acc2],
                             prefix="Epoch: [{}]".format(epoch), logger = logger)

    model.train()
    prefetcher = data_prefetcher(train_loader)
    images, target1, target2 = prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()

    while images is not None:
        
        out = torch.sigmoid(model(images))
        out_covid, out_severe = out[:, 0], out[:, 1] #.cpu().detach().numpy()

        loss1 = torch.nn.BCELoss()(out_covid, target1)
        loss2 = torch.nn.BCELoss()(out_severe, target2)

        loss = loss1 + loss2

        #auc = roc_auc_score(target1.cpu().detach().numpy(), out_covid.cpu().detach().numpy())
        acc1 = ((out_covid > 0.5) == target1).sum() / len(target1)
        acc2 = ((out_severe > 0.5) == target2).sum() / len(target2)

        losses.update(loss.item(), images.size(0))
        loss_acc1.update(acc1, images.size(0))
        loss_acc2.update(acc2, images.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if i % config.print_freq == 0:
            progress.display(i)

        i += 1

        images, target1, target2 = prefetcher.next()

    logger.info("[train acc1, acc2]: %.4f;%.4f" % (float(loss_acc1.avg), float(loss_acc1.avg)))


def validate(val_loader, model, config, logger):

    loss_acc1 = AverageMeter('acc1', ':6.2f')
    loss_acc2 = AverageMeter('acc2', ':6.2f')
    progress = ProgressMeter(len(val_loader), [loss_acc1, loss_acc2], prefix='Test: ', logger = logger)
    model.eval()

    with torch.no_grad():
        prefetcher = data_prefetcher(val_loader)
        images, target1, target2 = prefetcher.next()
        i = 0
        while images is not None:

            out = torch.sigmoid(model(images))
            out_covid, out_severe = out[:, 0], out[:, 1]
            loss1 = torch.nn.BCELoss()(out_covid, target1)
            loss2 = torch.nn.BCELoss()(out_severe, target2)

            loss = loss1 + loss2

            #acc1 = accuracy_score(out_covid.cpu().detach().numpy(), target1.cpu().detach().numpy())
            #print(out_covid)
            #print(target1)
            acc1 = ((out_covid > 0.5) == target1).sum() / len(target1)
            acc2 = ((out_severe > 0.5) == target2).sum() / len(target2)

            #print("######", ((out_covid > 0.5) == target1).sum())

            loss_acc1.update(acc1.item(), images.size(0))
            loss_acc2.update(acc2.item(), images.size(0))

            if i % config.print_freq == 0:
                progress.display(i)

            i += 1

            images, target1, target2 = prefetcher.next()

        logger.info("[val acc1, acc2]: %.4f;%.4f" % (float(loss_acc1.avg), float(loss_acc2.avg)))
        logger.info("\n")
    return loss_acc1.avg, loss_acc2.avg


if __name__ == '__main__':
    main()