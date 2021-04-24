import io
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler


class MyHandler(BaseHandler):
    def __init__(self, *args, **kargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC)            
        ])
    
    def preprocess_one_image(self, req):
        """Process one single Image

        Args:
            req ([request]): single Image request
        """
        image = req.get("data")
        if image is None:
            image = req.get("body")
        image = Image.open(io.BytesIO(image)).convert("RGB")
        image= self.transform(image)
        image = np.array(image)
        img_lab = rgb2lab(image).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...]/ 50. -1
        ab = img_lab[[1, 2], ...] / 110. 
        
        return {'L': L, 'ab': ab}
    
    def preprocess(self, requests):
        """Process all the images from the requests 
           and batch them in a tensor.

        Args:
            requests ([type]): [description]

        Returns:
            [type]: [description]
        """
        images = []
        images.append([self.preprocess_one_image(req) for req in requests])
        images = torch.cat(images)
        
        return images
        
    def inference(self, data, *args, **kwargs):
        
        
        
