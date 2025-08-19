import os
import argparse
import random
import math
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset

# model zoo
from SAM2MSnet import SAM2MSNet,LossNet,LossNetresnet50

torch.cuda.empty_cache()
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, default='Pweight/sam2_hiera_base_plus.pt',  #Pweight/sam2_hiera_tiny.pt  Pweight/sam2_hiera_small.pt   Pweight/sam2_hiera_base_plus.pt
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str,  default='dataset\\OSM\\train\\images\\',
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, default='dataset\\OSM\\train\\gt\\',
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, default='Save_Path_DSAM2FPN',
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=1, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()

use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def main(args):    
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    net = LossNetresnet50()
    net.eval()
    net.cuda()
    device = torch.device("cuda")
    #model = SAM2UNet(args.hiera_path)
    model = SAM2MSNet(args.hiera_path)
    #model = SAM2FPNPlus(args.hiera_path)
    #model = SAM2FPN(args.hiera_path)
    #model = DSAM2FPN(args.hiera_path)
    #model = SegFormer(num_classes=1, phi="b2", pretrained=False)     #   b0、b1、b2、b3、b4、b5
    #model = UNet(3,2)
    #model = UnetPlusPlus(num_classes=2, deep_supervision=False)
    #model = MSNet()
    #model = M2SNet()
    #model = DinkNet50()
    #model = SGCN_res50(num_classes=1)
    #model = MSMDFF_Net_base(3,init_block_SCMD=True,decoder_use_SCMD=False)
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    os.makedirs(args.save_path, exist_ok=True)
    global_step = 0

    for epoch in range(args.epoch):
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            # SAM2UNet,SAM2FPN,SAM32FPNPlus
            """ pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            print("第 "+ str(epoch+1)+"-"+str(i)+" loss值: ")
            print(loss) """
            # UNet Unet++ DinkNet50
            """ pred = model(x)
            loss = structure_loss(pred, target)
            print("第 "+ str(epoch+1)+"-"+str(i)+" loss值: ")
            print(loss) """

            # MSNet M2SNet SAM2MSNet SAM2FPN
            pred1 = model(x)
            loss2u = net(F.sigmoid(pred1), target)
            loss1u = structure_loss(pred1, target)
            loss = loss1u + 0.1 * loss2u
            print("第 "+ str(epoch+1)+"-"+str(i)+" loss值: ")
            print(loss,loss1u,loss2u) 
              
            torch.autograd.set_detect_anomaly(True)

            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                
        scheduler.step()
        
        torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2FPN-%d.pth' % (epoch + 1)))
        print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM2FPN-%d.pth'% (epoch + 1)))




# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main(args)