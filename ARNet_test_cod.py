import time

import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import imageio
import os, argparse
import cv2
from models.ARNet import ARNet
from data_cod import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='./TestDataset/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = ARNet()
model.load_state_dict(torch.load('./**************'))  ## weight path
model.cuda()
model.eval()


test_datasets = ['CHAMELEON']
for dataset in test_datasets:
    save_path = './test_maps/ARNet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    total_time = 0
    count = 0
    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        start_time = time.perf_counter()
        # res = model(image)
        res, res2, res3, e = model(image)
        end_time = time.perf_counter()
        count += 1
        total_time += end_time-start_time
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # res = torch.sigmoid(res)  # 添加这行代码
        
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # res = skeleton_postprocess(torch.from_numpy(res[None,None])) 
        # print("PG值范围:", res.min().item(), "-", res.max().item())
# 预期正常范围：0.0 ~ 1.0（添加Sigmoid后）



        #print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res*255)
#         res2 = F.upsample(res2, size=gt.shape, mode='bilinear', align_corners=False)
#         # res = torch.sigmoid(res)  # 添加这行代码
        
#         res2 = res2.sigmoid().data.cpu().numpy().squeeze()
#         res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
#         # res = skeleton_postprocess(torch.from_numpy(res[None,None])) 
#         # print("PG值范围:", res.min().item(), "-", res.max().item())
# # 预期正常范围：0.0 ~ 1.0（添加Sigmoid后）



#         #print('save img to: ',save_path+name)
#         cv2.imwrite(save_path + 'res2/'+ name, res*255)
        
#         res3 = F.upsample(res3, size=gt.shape, mode='bilinear', align_corners=False)
#         # res = torch.sigmoid(res)  # 添加这行代码
        
#         res3 = res3.sigmoid().data.cpu().numpy().squeeze()
#         res3 = (res3 - res3.min()) / (res3.max() - res3.min() + 1e-8)
#         # res = skeleton_postprocess(torch.from_numpy(res[None,None])) 
#         # print("PG值范围:", res.min().item(), "-", res.max().item())
# # 预期正常范围：0.0 ~ 1.0（添加Sigmoid后）



#         #print('save img to: ',save_path+name)
#         cv2.imwrite(save_path + 'res3/' + name, res*255)
    #     e = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=True)
    #     e = e.data.cpu().numpy().squeeze()
    #     e = (e - e.min()) / (e.max() - e.min() + 1e-8)
    #     imageio.imwrite(save_path+'edge/'+name, (e*255).astype(np.uint8))
    fps = count/total_time
    print('FPS:', fps)
    #print('Test Done!')
