import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import re
from dataset import TestDataset
from SAM2MSnet import SAM2MSNet,LossNet

def save_model_structure(model, filename="model_structure.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        # 处理DataParallel包装
        model_to_analyze = model.module if isinstance(model, nn.DataParallel) else model
        
        def recursive_print(module, prefix='', depth=0):
            """递归打印模型结构"""
            # 生成缩进前缀
            indent = '│   ' * (depth-1) + '├── ' if depth > 0 else ''
            
            # 获取模块基本信息
            mod_str = f"{indent}{prefix}({module.__class__.__name__})"
            
            # 添加卷积层参数信息
            if isinstance(module, nn.Conv2d):
                mod_str += f" | in: {module.in_channels}, out: {module.out_channels}, kernel: {module.kernel_size}"
            elif isinstance(module, nn.BatchNorm2d):
                mod_str += f" | features: {module.num_features}"
            elif isinstance(module, nn.Linear):
                mod_str += f" | in: {module.in_features}, out: {module.out_features}"
            
            # 打印到控制台和文件
            print(mod_str)
            f.write(mod_str + '\n')
            
            # 递归处理子模块
            for name, child in module.named_children():
                recursive_print(child, prefix=f"{name}", depth=depth+1)

        # 开始递归打印
        print(f"\n{' Model Structure ':=^80}")
        f.write(f"{' Model Structure ':=^80}\n")
        recursive_print(model_to_analyze)
        print(f"{'':=^80}\n")
        f.write(f"{'':=^80}\n\n")

parser = argparse.ArgumentParser()
""" parser.add_argument("--checkpoint", type=str, default="save_path_5\SAM2-UNet2-16.pth",
                help="path to the checkpoint of sam2-unet") """
parser.add_argument("--checkpoint", type=str, default="MaSazhusai\\Massachusetts_Save_Path_SGCN\\SGCN-Massachusetts-50.pth",
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, default="dataset\\Massachusetts\\test\\images/",
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, default="dataset\\Massachusetts\\test\\gt/",
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, default="PRE_Massachusetts_SGCN_res50",  #lossnet 0.1权重
                    help="path to save the predicted masks")
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 1024)
model = SAM2MSNet()

model.to(device)
# 加载检查点后调用
save_model_structure(model, "model_architecture.txt")
""" print(model)
def unwrap_model(ckpt):
    state_dict={}
    for k,v in ckpt.items():
        if "module." in k:
            state_dict[k.replace('module.','')]=v
            if "Ddim.weight" in k:
                print("wei")
                print(k,v.tolist())
            if "Ddim.bia" in k:
                print("bia")
                print(k,v)
        else:
            state_dict[k] = v
    return state_dict

checkpoint = torch.load(args.checkpoint)

# 获取去除 "module." 前缀的 state_dict
weights = unwrap_model(checkpoint)
#print(weights)
model.load_state_dict(weights, strict=True) """
""" weight=unwrap_model(torch.load(args.checkpoint))
model.load_state_dict(torch.load(weight), strict=True) """

model = nn.DataParallel(model)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)

for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        print(image.dtype,image.shape)  # 打印出图像的数据类型

        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        #res,_,_= model(image)
        res= model(image)
        """ res,_,_= model(image)
        _,res,_= model(image) """
        # fix: duplicate sigmoid
        #res = torch.sigmoid(res)
        _,_,H1,W1 = res.shape
        if (H1,W1) != gt.shape:
            print(res.shape,gt.shape)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print("Saving " + name)
        name = name[:-4]
        imageio.imsave(os.path.join(args.save_path, name + ".png"), res)
