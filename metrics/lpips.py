import lpips
import argparse
import os
from PIL import Image
import numpy as np
import torch


def normalize_tensor(t, dim=1):
    normed_tensors = []
    for i in range(t.size()[dim]):
        new_t = t[:, i, :, :]
        new_t_norm = (new_t - new_t.min())/(new_t.max()-new_t.min())
        normed_tensors.append(2*new_t_norm - 1)
    
    tt = torch.cat(normed_tensors, dim=0)

    return tt.unsqueeze(0)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_hr', type=str)
    parser.add_argument('--dataset_sr', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser

if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()

    device = args.device
    
    hr_imgs = os.listdir(args.dataset_hr)
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    lpips_alex_avg = 0
    lpips_vgg_avg = 0
    
    for img in hr_imgs:
        
        hr_img = torch.from_numpy(np.array(Image.open(args.dataset_hr + '/' + img))).permute(2,1,0).unsqueeze(0).float().to(device)
        sr_img = torch.from_numpy(np.array(Image.open(args.dataset_sr + '/' + img))).permute(2,1,0).unsqueeze(0).float().to(device)

        if sr_img.size() != hr_img.size():
            n, c, h, w = sr_img.size()
            hr_img = hr_img[:, :, :h, :w]
        
        hr_img = normalize_tensor(hr_img)
        sr_img = normalize_tensor(sr_img)
        alex_loss = loss_fn_alex(sr_img, hr_img)
        lpips_alex_avg += alex_loss.detach().item()

        vgg_loss = loss_fn_vgg(sr_img, hr_img)
        lpips_vgg_avg += vgg_loss.detach().item()
        
        del hr_img
        del sr_img
        del vgg_loss
        del alex_loss
        torch.cuda.empty_cache()
    
    print(args.dataset_sr)
    print('Avg AlexNet perceptual loss is: %.4f'%(lpips_alex_avg/len(hr_imgs)))
    print('Avg VGGNet perceptual loss is: %.4f'%(lpips_vgg_avg/len(hr_imgs)))

