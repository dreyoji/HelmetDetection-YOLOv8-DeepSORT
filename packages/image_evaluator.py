# pip install pyiqa
import pyiqa
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms 


class image_evaluator:
    def __init__(self):
        # Creates the metric niqe and psnr from pyiqa library
        self.niqe = pyiqa.create_metric('niqe')
        self.psnr = pyiqa.create_metric('psnr')

    def compute_niqe(self, img: Image):
        # Returns the niqe score of the image
        img = self.convert_to_grayscale(img)
        return self.niqe(img)


    def compute_psnr(self, source_img: Image, output_img: Image):
        # Returns the Peak Signal to Noise Ratio of the output image based on the source image
        return self.psnr(source_img, output_img)
    
    @staticmethod
    def tensor_to_Image(tensor_img: torch.Tensor):
        return transforms.ToPILImage(tensor_img)
    
    @staticmethod
    def convert_to_grayscale(img: Image):
        return ImageOps.grayscale(img)
    
    
