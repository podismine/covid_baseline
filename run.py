from genericpath import exists
import os
import torch
import logging
import random
import numpy as np
import warnings
from args import get_parser
from dataset.data import data_prefetcher, AllData
import torch.distributed as dist
from models.sfcn import SFCN
from models.vgg import vgg16_bn
from models.dbn import DBN
from models.resnet import resnet18, resnet34, resnet50
from models.densenet import densenet121, densenet201
from apex import amp
from apex.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.utils import reduce_mean,adjust_learning_rate, AverageMeter, ProgressMeter, my_KLDivLoss
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
    torch.backends.cudnn.benchmark = False
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
    model_names = ["resnet18", "resnet50", "vgg", "dense121", "sfcn", "dbn"]
    models = [resnet18, resnet50,vgg16_bn, densenet121, SFCN, DBN]

    best_acc1 = 99.0

    dist.init_process_group(backend='nccl')
    # create model
    model = models[model_names.index(config.arch)](output_dim=88, mode = config.mode)
 
    torch.cuda.set_device(config.local_rank)
    model.cuda()

    config.batch_size = int(config.batch_size / config.nprocs)
    
    optimizer = torch.optim.Adam(model.parameters(),lr = config.lr,weight_decay = 0.00005)
    #optimizer.load_state_dict(checkpoint['optimizer'])    

    model, optimizer = amp.initialize(model, optimizer, opt_level=config.opt_level)
    #amp.load_state_dict(checkpoint['amp'])
    model = DistributedDataParallel(model)
    
    cudnn.benchmark = True

    # Data loading code
    train_data = AllData(config.data, train = True)
    val_data = AllData(config.data, train = False)

    if config.mode in [3,4,6,9]:
        all_predictions = torch.zeros((len(train_data)))#.cuda()
        all_predictions_kl = torch.zeros((len(train_data)))#.cuda()
    elif config.mode == 7:
        all_predictions = torch.zeros((len(train_data)))#.cuda()
        all_predictions_kl = torch.zeros((len(train_data), 22))#.cuda()
    else:
        all_predictions = torch.zeros((len(train_data), 22))#.cuda()
        all_predictions_kl = torch.zeros((len(train_data), 22))#.cuda()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)

    train_loader = DataLoader(train_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = False, sampler = train_sampler)
    val_loader = DataLoader(val_data,config.batch_size,
                        shuffle=False,num_workers=2,pin_memory = False, sampler = val_sampler)


    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)

        alpha_t = config.alpha * ((epoch + 1) / config.epochs)
        alpha_t = max(0, alpha_t)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, config, all_predictions,all_predictions_kl,alpha_t, logger)
            
        mae = validate(val_loader, model, config, logger)

        is_best = mae < best_acc1
        best_acc1 = min(mae, best_acc1)
        if not os.path.exists("./checkpoints/%s" % config.env_name):
            try:
                os.makedirs("./checkpoints/%s" % config.env_name)
            except:
                pass # multiple processors bug

        if is_best and config.local_rank == 0:
                state = {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'best_acc1': best_acc1,
                        'amp': amp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                torch.save(state, './checkpoints/%s/%s_epoch_%s_%s' % (config.env_name, config.env_name, epoch, best_acc1))

def train(train_loader, model, optimizer, epoch, config, all_predictions,all_predictions_kl, alpha_t, logger):
    losses = AverageMeter('Loss', ':.4e')
    loss_mae = AverageMeter('mae1', ':6.2f')

    progress = ProgressMeter(len(train_loader), [losses, loss_mae],
                             prefix="Epoch: [{}]".format(epoch), logger = logger)

    model.train()

    prefetcher = data_prefetcher(train_loader)
    images, target, yy, bc, indices = prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    while images is not None:
        
        lam = config.lam
        out = model(images)
        if config.mode % 3 == 0:
            mae = torch.nn.L1Loss()(out, target)
        else:
            prob = torch.exp(out)
            pred = torch.sum(prob * bc, dim = 1)
            mae = torch.nn.L1Loss()(pred, target)

        #------------------------------------------------- baseline ---------------------------------
        if config.mode == 0:
            # baseline mse
            loss = torch.nn.MSELoss()(out, target)
        
        elif config.mode == 1:
            # baseline dex
            loss = torch.nn.MSELoss()(pred, target)

        elif config.mode == 2:
            # baseline soft label
            loss = my_KLDivLoss(out, yy)
            
        #------------------------------------------------- PS-KD ---------------------------------
        elif config.mode == 3:
            # pskd + mse
            if epoch == 0:
                all_predictions[indices] = target.detach().cpu()
            soft_targets = Variable(((1 - alpha_t) * target.detach().cpu()) + (alpha_t * all_predictions[indices.detach().cpu()])).cuda()
            all_predictions[indices.detach().cpu()] = out.detach().cpu()
            loss = torch.nn.MSELoss()(out, soft_targets)

        elif config.mode == 4:
            # pskd + mse
            if epoch == 0:
                all_predictions[indices] = target.detach().cpu()
            soft_targets = Variable(((1 - alpha_t) * target.detach().cpu()) + (alpha_t * all_predictions[indices.detach().cpu()])).cuda()
            all_predictions[indices.detach().cpu()] = pred.detach().cpu()
            loss = torch.nn.MSELoss()(pred, soft_targets)

        elif config.mode == 5:
            if epoch == 0:
                all_predictions[indices] = yy.detach().cpu()
            soft_targets = Variable(((1 - alpha_t) * yy.detach().cpu()) + (alpha_t * all_predictions[indices.detach().cpu()])).cuda()
            all_predictions[indices.detach().cpu()] = prob.detach().cpu()
            loss = my_KLDivLoss(out, soft_targets)

        #------------------------------------------------- Ours - add pskd  ---------------------------------
        elif config.mode == 6:
            # ours + mse
            if epoch == 0:
                all_predictions[indices.detach().cpu()] = target.detach().cpu()
                all_predictions_kl[indices.detach().cpu()] = target.detach().cpu()

            soft_targets_1 = Variable(((1 - alpha_t) * target.detach().cpu()) + (alpha_t * all_predictions[indices.detach().cpu()])).cuda()
            soft_targets_2 = Variable(all_predictions_kl[indices.detach().cpu()]).cuda()

            all_predictions[indices.detach().cpu()] = out.detach().cpu()
            all_predictions_kl[indices.detach().cpu()] = out.detach().cpu()

            out[out<1e-4] = 1e-4

            loss = torch.nn.MSELoss()(out, target) + lam * torch.nn.MSELoss()(out, soft_targets_1) + alpha_t * my_KLDivLoss(out.log(), soft_targets_2)

        elif config.mode == 7:
            # ours + mse
            if epoch == 0:
                all_predictions[indices.detach().cpu()] = target.detach().cpu()
                all_predictions_kl[indices.detach().cpu()] = yy.detach().cpu()

            soft_targets_1 = Variable(((1 - alpha_t) * target.detach().cpu()) + (alpha_t * all_predictions[indices.detach().cpu()])).cuda()
            soft_targets_2 = Variable(all_predictions_kl[indices.detach().cpu()]).cuda()

            all_predictions[indices.detach().cpu()] = pred.detach().cpu()
            all_predictions_kl[indices.detach().cpu()] = prob.detach().cpu()

            loss = torch.nn.MSELoss()(pred, target) + lam * torch.nn.MSELoss()(pred, soft_targets_1) + alpha_t * my_KLDivLoss(out, soft_targets_2)

        elif config.mode == 8:
            if epoch == 0:
                all_predictions[indices] = yy.detach().cpu()
                all_predictions_kl[indices] = yy.detach().cpu()

            soft_targets_1 = Variable(((1 - alpha_t) * yy.detach().cpu()) + (alpha_t * all_predictions[indices.detach().cpu()])).cuda()
            soft_targets_2 = Variable(all_predictions_kl[indices.detach().cpu()]).cuda()

            all_predictions[indices.detach().cpu()] = prob.detach().cpu()
            all_predictions_kl[indices.detach().cpu()] = prob.detach().cpu()
            loss = my_KLDivLoss(out, yy) + lam * my_KLDivLoss(out, soft_targets_1) + alpha_t * my_KLDivLoss(out, soft_targets_2)

        torch.distributed.barrier() 

        reduced_loss = reduce_mean(loss, config.nprocs)
        reduced_mae = reduce_mean(mae, config.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        loss_mae.update(reduced_mae.item(), images.size(0))

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if i % config.print_freq == 0:
            progress.display(i)

        i += 1

        images, target, yy, bc,indices = prefetcher.next()

    logger.info("[train mae]: %.4f" % float(loss_mae.avg))


def validate(val_loader, model, config, logger):

    loss_mae = AverageMeter('mae1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [loss_mae], prefix='Test: ', logger = logger)
    model.eval()

    with torch.no_grad():
        prefetcher = data_prefetcher(val_loader)
        images, target, yy, bc,indices = prefetcher.next()
        i = 0
        while images is not None:

            out = model(images)

            if config.mode % 3 == 0:
                mae = torch.nn.L1Loss()(out, target) 
            else:
                prob = torch.exp(out)
                pred = torch.sum(prob * bc, dim = 1)
                mae = torch.nn.L1Loss()(pred, target) 
            
            torch.distributed.barrier()
            reduced_mae = reduce_mean(mae, config.nprocs)
            loss_mae.update(reduced_mae.item(), images.size(0))

            if i % config.print_freq == 0:
                progress.display(i)

            i += 1

            images, target, yy, bc, indices = prefetcher.next()

        logger.info("[val mae]: %.4f" % float(loss_mae.avg))
    return loss_mae.avg


if __name__ == '__main__':
    main()