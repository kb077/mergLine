import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_faces_dataset_3 import *
from pointMLP import *
from losses_and_metrics_for_mesh import *
import utils
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
add_log = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1910"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# set up logger

previous_check_point_path = './models_pointmlp_biou_alpha'
previous_check_point_name = 'latest_checkpoint.tar'

train_list="data/new_dataset_train.h5"
val_list="data/new_dataset_test.h5"

model_path = './models_pointmlp_biou_alpha/'
model_name = 'pointmlp_test_1'
checkpoint_name = 'latest_checkpoint.tar'

save_dir=model_path

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

logger, logger_results = utils.setup_logger(save_dir, checkpoint=False)

logdir = writer.get_logdir()
logger.info('logdir: '+ str(logdir))
use_visdom = False # if you don't use visdom, please set to False
# DGCNN specific
k = 20
# k = 30 in TSGCNet
emb_dims = 1024
dropout = 0.5
use_sgd = False
lr = 0.0001
momentum = 0.9
scheduler = 'step'
seed = 1

num_classes = 2
num_channels=24
epochs=1000
num_epochs = 1000
num_workers = 0
train_batch_size = 8
val_batch_size = 8
num_batches_to_print = 5
test_num_batches_to_print=1
use_amp = True
alpha =5
logger.info('alpha: '+ str(alpha))
if use_visdom:
    # set plotter
    global plotter
    plotter = utils.VisdomLinePlotter(env_name=model_name)

if not os.path.exists(model_path):
    os.mkdir(model_path)

torch.manual_seed(seed)

# set dataset
# we will set the patch size to be bigger because our meshes are ~30k cells

training_dataset = Mesh_Dataset(data_list_path=train_list,
                                num_classes=num_classes,
                                patch_size=13000)

val_dataset = Mesh_Dataset(data_list_path=val_list,
                           num_classes=num_classes,
                           patch_size=13000)

train_loader = DataLoader(dataset=training_dataset,
                          batch_size=train_batch_size,
                          shuffle=True,
                          num_workers=num_workers)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        shuffle=False,
                        num_workers=num_workers)

torch.cuda.is_available()

# set model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = pointMLP_seg(num_classes=num_classes, num_channels=num_channels)
model = nn.DataParallel(model)
model = model.to(device, dtype=torch.float)

opt = None
if use_sgd:
    logger.info("Use SGD")
    opt = optim.SGD(model.parameters(), lr=lr*100, momentum=momentum, weight_decay=1e-4)
else:
    logger.info("Use Adam")
    #opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    opt = optim.Adam(model.parameters(), lr=lr)
    #opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.5)

scheduler = None
if scheduler == 'cos':
    scheduler = CosineAnnealingLR(opt, epochs, eta_min=1e-3)
elif scheduler == 'step':
    scheduler = StepLR(opt, 120, 0.5, epochs)


scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

losses, mdsc, msen, mppv ,mbiou = [], [], [], [] ,[]
val_losses, val_mdsc, val_msen, val_mppv , val_mbiou = [], [], [], [],[]

best_val_dsc = 0.0
#if load_checkpt is True
# re-load
checkpoint=False
if checkpoint :
    checkpoint = torch.load(os.path.join(previous_check_point_path, previous_check_point_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_init = checkpoint['epoch']
    losses = checkpoint['losses']
    mdsc = checkpoint['mdsc']
    msen = checkpoint['msen']
    mppv = checkpoint['mppv']
    val_losses = checkpoint['val_losses']
    val_mdsc = checkpoint['val_mdsc']
    val_msen = checkpoint['val_msen']
    val_mppv = checkpoint['val_mppv']
    del checkpoint

    best_val_dsc = max(val_mdsc)
    print('previous best value: ', best_val_dsc)
#cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

logger.info('Training model...')
class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
class_weights_for_loss = class_weights
print('class_weights_for_loss: ', class_weights_for_loss)

# batch accumulation parameter
accum_iter = 16

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_mdsc = 0.0
    running_msen = 0.0
    running_mppv = 0.0
    running_mbiou = 0.0
    
    loss_epoch = 0.0
    mdsc_epoch = 0.0
    msen_epoch = 0.0
    mppv_epoch = 0.0
    mbiou_epoch = 0.0

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()

    end = time.time()


    for i_batch, batched_sample in enumerate(train_loader):
        # send mini-batch to device
        inputs = batched_sample['cells'].to(device, dtype=torch.float)
        labels = batched_sample['labels'].to(device, dtype=torch.long)
        one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)
        opt.zero_grad()

        outputs = model(inputs)
        op = outputs.contiguous().view(-1, num_classes)
        lbl = labels.view(-1)
        loss = F.nll_loss(op, lbl, weight=class_weights_for_loss)

        dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
        sen = weighting_SEN(outputs, one_hot_labels, class_weights)
        ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
        biou= weighting_BIoU(outputs, one_hot_labels,inputs)
        biou_metric = B_IoU (outputs, one_hot_labels,inputs)
        
        loss+=alpha*biou
        loss.backward()
        opt.step()
        
        running_loss += loss.item()
        running_mdsc += dsc.item()
        running_msen += sen.item()
        running_mppv += ppv.item()
        running_mbiou += biou_metric
        
        loss_epoch += loss.item()
        mdsc_epoch += dsc.item()
        msen_epoch += sen.item()
        mppv_epoch += ppv.item()
        mbiou_epoch += biou_metric
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % num_batches_to_print == num_batches_to_print-1:  
            logger.info('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7},biou: {8}, Batch Time:{9}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print, running_mbiou/num_batches_to_print, batch_time.avg))
            running_loss = 0.0
            running_mdsc = 0.0
            running_msen = 0.0
            running_mppv = 0.0
            running_mbiou =0.0 

    # record losses and metrics
    losses.append(loss_epoch/len(train_loader))
    mdsc.append(mdsc_epoch/len(train_loader))
    msen.append(msen_epoch/len(train_loader))
    mppv.append(mppv_epoch/len(train_loader))
    mbiou.append(mbiou_epoch/len(train_loader))
    
    if add_log is True:
        writer.add_scalar("Loss/train", loss_epoch/len(train_loader), epoch)
        writer.add_scalar("mdsc/train", mdsc_epoch/len(train_loader), epoch)
        writer.add_scalar("msen/train", msen_epoch/len(train_loader), epoch)
        writer.add_scalar("mppv/train", mppv_epoch/len(train_loader), epoch)
        writer.add_scalar("mbiou/train",mbiou_epoch/len(train_loader),epoch)

    if epoch+1 % 20 == 0:
        scheduler.step()
        if scheduler == 'cos':
            scheduler.step()
        elif scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5


    #reset
    loss_epoch = 0.0
    mdsc_epoch = 0.0
    msen_epoch = 0.0
    mppv_epoch = 0.0
    mbiou_epoch = 0.0
    
    running_mppv = 0.0
    # validation
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        running_val_mdsc = 0.0
        running_val_msen = 0.0
        running_val_mppv = 0.0
        running_val_mbiou =0.0
        
        val_loss_epoch = 0.0
        val_mdsc_epoch = 0.0
        val_msen_epoch = 0.0
        val_mppv_epoch = 0.0
        val_mbiou_epoch =0.0
        
        batch_time2 = utils.AverageMeter()
        end = time.time()

        for i_batch, batched_val_sample in enumerate(val_loader):
            inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
            labels = batched_val_sample['labels'].to(device, dtype=torch.long)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            outputs = model(inputs)
            op = outputs.contiguous().view(-1, num_classes)
            lbl = labels.view(-1)

            loss = F.nll_loss(op, lbl, weight=class_weights_for_loss)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            biou= weighting_BIoU(outputs, one_hot_labels,inputs)
            biou_metric= B_IoU(outputs, one_hot_labels,inputs)
            
            loss+=alpha*biou
            running_val_loss += loss.item()
            running_val_mdsc += dsc.item()
            running_val_msen += sen.item()
            running_val_mppv += ppv.item()
            running_val_mbiou += biou_metric
            
            val_loss_epoch += loss.item()
            val_mdsc_epoch += dsc.item()
            val_msen_epoch += sen.item()
            val_mppv_epoch += ppv.item()
            val_mbiou_epoch += biou_metric

            # measure elapsed time
            batch_time2.update(time.time() - end)
            end = time.time()

            if i_batch % test_num_batches_to_print == test_num_batches_to_print-1:  
                logger.info('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7},val_biou: {8}, Batch Time:{9}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/test_num_batches_to_print, running_val_mdsc/test_num_batches_to_print, running_val_msen/test_num_batches_to_print, running_val_mppv/test_num_batches_to_print, running_val_mbiou/test_num_batches_to_print, batch_time2.avg))
                running_val_loss = 0.0
                running_val_mdsc = 0.0
                running_val_msen = 0.0
                running_val_mppv = 0.0
                running_val_mbiou = 0.0

        # record losses and metrics
        val_losses.append(val_loss_epoch/len(val_loader))
        val_mdsc.append(val_mdsc_epoch/len(val_loader))
        val_msen.append(val_msen_epoch/len(val_loader))
        val_mppv.append(val_mppv_epoch/len(val_loader))
        val_mppv.append(val_mbiou_epoch/len(val_loader))
        
        if add_log is True:
            writer.add_scalar("Loss/val", val_loss_epoch/len(val_loader), epoch)
            writer.add_scalar("mdsc/val", val_mdsc_epoch/len(val_loader), epoch)
            writer.add_scalar("msen/val", val_msen_epoch/len(val_loader), epoch)
            writer.add_scalar("mppv/val", val_mppv_epoch/len(val_loader), epoch)
            writer.add_scalar("mbiou/val", val_mbiou_epoch/len(val_loader), epoch)

        # reset
        val_loss_epoch = 0.0
        val_mdsc_epoch = 0.0
        val_msen_epoch = 0.0
        val_mppv_epoch = 0.0
        val_mbiou_epoch =0.0

        if i_batch >= 1 :
            logger.info('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
        if use_visdom:
            plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
            plotter.plot('DSC', 'train', 'DSC', epoch+1, mdsc[-1])
            plotter.plot('SEN', 'train', 'SEN', epoch+1, msen[-1])
            plotter.plot('PPV', 'train', 'PPV', epoch+1, mppv[-1])
            plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])
            plotter.plot('DSC', 'val', 'DSC', epoch+1, val_mdsc[-1])
            plotter.plot('SEN', 'val', 'SEN', epoch+1, val_msen[-1])
            plotter.plot('PPV', 'val', 'PPV', epoch+1, val_mppv[-1])

    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'losses': losses,
                'mdsc': mdsc,
                'msen': msen,
                'mppv': mppv,
                'mbiou':mbiou,
                'val_losses': val_losses,
                'val_mdsc': val_mdsc,
                'val_msen': val_msen,
                'val_mppv': val_mppv,
                'val_mbiou':val_mbiou},
                model_path+checkpoint_name)

    # save the best model
    if best_val_dsc < val_mdsc[-1]:
        best_val_dsc = val_mdsc[-1]
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'mbiou':mbiou,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv,
                    'val_mbiou':val_mbiou},
                    model_path+'{}_best.tar'.format(model_name))

