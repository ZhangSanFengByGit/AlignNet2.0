# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 21:29:32 2018

@author: cvpr
"""


from AlignNet2 import AlignNet2

import torch
from torchvision import transforms
from PIL import Image
from Vgg16 import Vgg16

vgg = Vgg16().eval()
vgg = vgg.cuda()

to_img = transforms.ToPILImage()

transform = transforms.Compose([
        transforms.Resize([612,612]),
        transforms.RandomResizedCrop(512),
        transforms.RandomRotation([-5,5]),
        #transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform2 = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

img1 = Image.open('H:\\pedestrian_RGBT\\kaist-rgbt\\images\\set1\\V000\\visible\\I00000.jpg')
img2 = Image.open('H:\\pedestrian_RGBT\\kaist-rgbt\\images\\set1\\V000\\lwir\\I00000.jpg')

img1 = transform2(img1).unsqueeze(0).cuda()
img2_trans = transform(img2).unsqueeze(0).cuda()


alignTest = AlignNet2()
alignTest = alignTest.cuda()

#temp Area~~~~~~~~
alignTest.load_state_dict(torch.load('H:\\model_checkPoint_save_test\\ckpt_epoch_400_Sequence_id_0.plk'))

img1 = vgg(img1)
img2_trans = vgg(img2_trans)

out = alignTest(img1,img2_trans)

out = out.T_tran_image.cpu()
out = out.data[0]
out = to_img(out)

        
        
        
        