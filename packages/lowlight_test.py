import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import time
import packages.model as model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


 
def lowlight(img):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = img

	data_lowlight = (np.asarray(data_lowlight)/255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	# Change to number of epochs - 1
	DCE_net.load_state_dict(torch.load('snapshots/Epoch9.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)
	enhanced_image = enhanced_image[0]
	transform = torchvision.transforms.ToPILImage(mode='RGB')

	return transform(enhanced_image)

