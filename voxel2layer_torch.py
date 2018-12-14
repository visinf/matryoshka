import torch
import torch.nn.functional as tf
# r = torch.arange(0,128+2)
# id1,id2,id3 = torch.meshgrid([r,r,r])

import scipy.io as sio

def generate_indices(side, device=torch.device('cpu')):
    """ Generates meshgrid indices. 
        May be helpful to cache when converting lots of shapes.
    """
    r = torch.arange(0, side+2).to(device)
    id1,id2,id3 = torch.meshgrid([r,r,r])
    return id1.short(), id2.short(), id3.short()


def encode_shapelayer(voxel, id1=None, id2=None, id3=None):
    """ Encodes a voxel grid into a shape layer
        by projecting the enclosed shape to 6 depth maps.
        Returns the shape layer and a reconstructed shape.
        The reconstructed shape can be used to recursively encode
        complex shapes into multiple layers.
        Optional parameters id1,id2,id3 can save memory when multiple
        shapes are to be encoded. They can be constructed like this:
        r = np.arange(0,voxel.shape[0]+2)
        id1,id2,id3 = np.meshgrid(r,r,r, indexing='ij')
    """
    
    device = voxel.device

    side = voxel.shape[0]
    assert voxel.shape[0] == voxel.shape[1] and voxel.shape[1] == voxel.shape[2], \
        'The voxel grid needs to be a cube. It is however %dx%dx%d.' % \
        (voxel.shape[0],voxel.shape[1],voxel.shape[2])

    if id1 is None or id2 is None or id3 is None:
        id1, id2, id3 = generate_indices(side, device)
        pass

    # add empty border for argmax
    # need to distinguish empty tubes
    v = torch.zeros(side+2,side+2,side+2, dtype=torch.uint8, device=device)
    v[1:-1,1:-1,1:-1] = voxel
        
    shape_layer = torch.zeros(side, side, 6, dtype=torch.int16, device=device)
    
    # project depth to yz-plane towards negative x
    s1 = torch.argmax(v, dim=0) # returns first occurence
    # project depth to yz-plane towards positive x
    # s2 = torch.argmax(v[-1::-1,:,:], dim=0) #, but negative steps not supported yet
    s2 = torch.argmax(torch.flip(v, [0]), dim=0)
    s2 = side+1-s2 # correct for added border

    # set all empty tubes to 0    
    s1[s1 < 1] = side+2
    s2[s2 > side] = 0    
    shape_layer[:,:,0] = s1[1:-1,1:-1]
    shape_layer[:,:,1] = s2[1:-1,1:-1]
    
    # project depth to xz-plane towards negative y
    s1 = torch.argmax(v, dim=1)
    # project depth to xz-plane towards positive y
    # s2 = torch.argmax(v[:,-1::-1,:], dim=1) #, but negative steps not supported yet
    s2 = torch.argmax(torch.flip(v, [1]), dim=1)
    s2 = side+1-s2

    s1[s1 < 1] = side+2
    s2[s2 > side] = 0    
    shape_layer[:,:,2] = s1[1:-1,1:-1]
    shape_layer[:,:,3] = s2[1:-1,1:-1]
    
    # project depth to xy-plane towards negative z
    s1 = torch.argmax(v, dim=2)
    # project depth to xy-plane towards positive z
    # s2 = torch.argmax(v[:,:,-1::-1], dim=2) #, but negative steps not supported yet
    s2 = torch.argmax(torch.flip(v, [2]), dim=2)
    s2 = side+1-s2

    s1[s1 < 1] = side+2
    s2[s2 > side] = 0    
    shape_layer[:,:,4] = s1[1:-1,1:-1]
    shape_layer[:,:,5] = s2[1:-1,1:-1]
    
    return shape_layer


def decode_shapelayer(shape_layer, id1=None, id2=None, id3=None):
    """ Decodes a shape layer to voxel grid.
        Optional parameters id1,id2,id3 can save memory when multiple
        shapes are to be encoded. They can be constructed like this:
        r = np.arange(0,voxel.shape[0]+2)
        id1,id2,id3 = np.meshgrid(r,r,r)
    """

    device = shape_layer.device
    side   = shape_layer.shape[0]

    if id1 is None or id2 is None or id3 is None:
        id1, id2, id3 = generate_indices(side, device)
        pass

    x0 = tf.pad(shape_layer[:,:,0], (1,1,1,1), 'constant').unsqueeze(0)
    x1 = tf.pad(shape_layer[:,:,1], (1,1,1,1), 'constant').unsqueeze(0)
    y0 = tf.pad(shape_layer[:,:,2], (1,1,1,1), 'constant').unsqueeze(1)
    y1 = tf.pad(shape_layer[:,:,3], (1,1,1,1), 'constant').unsqueeze(1)
    z0 = tf.pad(shape_layer[:,:,4], (1,1,1,1), 'constant').unsqueeze(2)
    z1 = tf.pad(shape_layer[:,:,5], (1,1,1,1), 'constant').unsqueeze(2)

    v0 = (x0 <= id1) & (id1 <= x1)
    v1 = (y0 <= id2) & (id2 <= y1)
    v2 = (z0 <= id3) & (id3 <= z1)

    return (v0 & v1 & v2)[1:-1,1:-1,1:-1]


def encode_shape(voxel, num_layers=1, id1=None, id2=None, id3=None):
    """ Encodes a shape recursively into multiple layers.
    """

    device      = voxel.device
    side        = voxel.shape[0]
    shape_layer = torch.zeros(side, side, 6 * num_layers, dtype=torch.int16, device=device)
    decoded     = torch.zeros(side, side, side, dtype=torch.uint8, device=device)

    for i in range(num_layers):

        if i % 2 == 0:
            shape_layer[:,:,i*6:(i+1)*6] = encode_shapelayer(voxel - decoded, id1,id2,id3)
            rec = decode_shapelayer(shape_layer[:,:,i*6:(i+1)*6], id1, id2, id3)
            decoded += rec
        else:
            shape_layer[:,:,i*6:(i+1)*6] = encode_shapelayer(decoded - voxel, id1,id2,id3)
            rec = decode_shapelayer(shape_layer[:,:,i*6:(i+1)*6], id1, id2, id3)
            decoded -= rec
        
        # print('%.3f ' % (np.sum(np.logical_and(voxel, decoded)) / np.sum(np.logical_or(voxel, decoded))), end='')
        pass

    return shape_layer


def decode_shape(shape_layer, id1=None, id2=None, id3=None):
    """ Recursively decodes a shape from multiple shape layers.
    """

    device     = shape_layer.device
    side       = shape_layer.shape[0]
    num_layers = int(shape_layer.shape[2] / 6)
    decoded    = torch.zeros(side, side, side, dtype=torch.uint8, device=device)
    
    for i in range(num_layers):

        if i % 2 == 0:
            rec = decode_shapelayer(shape_layer[:,:,i*6:(i+1)*6], id1, id2, id3)
            decoded += rec
        else:
            rec = decode_shapelayer(shape_layer[:,:,i*6:(i+1)*6], id1, id2, id3)
            decoded -= rec
        pass

    return decoded

def shl2shlx(shape_layers):
    """ Modify shape layer representation for better learning.
        1. We reflect each odd depth map => background is always zeros, shape always positive depth.
        2. We transpose the first two depth maps => structures of shapes align better across depth maps.
    """
    side = shape_layers.shape[-1]
    shape_layers[:,::2,:,:] = side+2 - shape_layers[:,::2,:,:]
    shape_layers[:,:2,:,:]  = shape_layers[:,:2,:,:].clone().permute(0,1,3,2)
    return shape_layers


def shlx2shl(shape_layers):
    """ Undo the shape layer modifications for learning.
    """
    side = shape_layers.shape[-1]
    shape_layers[:,:2,:,:]  = shape_layers[:,:2,:,:].clone().permute(0,1,3,2)
    shape_layers[:,::2,:,:] = side+2 - shape_layers[:,::2,:,:]
    return shape_layers

    