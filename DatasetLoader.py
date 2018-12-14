from __future__ import print_function
import torch.utils.data as data
from PIL import Image, ImageOps
import os
import torch
import numpy as np
from torchvision import transforms
import scipy.io as sio

class RandomColorFlip(object):    
    def __call__(self, img):        
        c = np.random.choice(3,3,np.random.random() < 0.5)            
        return img[c,:,:]


class DatasetLoader(data.Dataset):    
    
    def __init__(self, samples, num_comp=1, input_transform=None, no_images=False, no_shapes=False):

        self.input_transform  = input_transform
        self.num_comp = num_comp      
        self.samples  = samples
        self.no_images = no_images
        self.no_shapes = no_shapes
        pass


    def __getitem__(self, index):
        
        imagepath = self.samples[index][0]
        shapepath = self.samples[index][1]
        
        flipped = False#random.random() > 0.5 if self.flip else False
        
        if self.no_images:
            imgs = None
        else:
            imgs = self.input_transform(self._load_image(Image.open(imagepath),flipped))

        if self.no_shapes:
            shape = None
        else:
            if shapepath.endswith('.shl.mat'):
                shape = self._load_shl(shapepath)
            elif shapepath.endswith('.vox.mat'):
                shape = self._load_vox(shapepath)
            else:
                assert False, ('Could not determine shape representation from file name (%s has neither ".shl.mat" nor ".vox.mat").' % shapepath)

        if self.no_images:
            if self.no_shapes:
                return
            else:
                return shape
        else:
            if self.no_shapes:
                return imgs
            else:
                return imgs, shape


    def __len__(self):
        return len(self.samples)


    def _load_vox(self, path):
        d = sio.loadmat(path)
        return torch.from_numpy(d['voxel'])


    def _load_shl(self, path):
        d = sio.loadmat(path)
        return torch.from_numpy(np.array(d['shapelayer'], dtype=np.int32)[:,:,:6*self.num_comp]).permute(2,0,1).contiguous().float() 


    def _load_image(self, temp, flipped=False):   
        # Only process if image has transparency (http://stackoverflow.com/a/1963146)
        if temp.mode == 'RGBA':
            alpha = temp.split()[-1]
            bg = Image.new("RGBA", temp.size, (128,128,128) + (255,))
            bg.paste(temp, mask=alpha)
            im = bg.convert('RGB').copy()
            bg.close()
            temp.close()
        else:
            im = temp.copy()
            temp.close()
        return (im.transpose(Image.FLIP_LEFT_RIGHT) if flipped else im)
