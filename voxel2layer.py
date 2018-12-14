import numpy as np
import argparse
import os
import scipy.io as sio
from utils import convert_files

def generate_indices(side):
    """ Generates meshgrid indices. 
        May be helpful to cache when converting lots of shapes.
    """
    r = np.arange(0, side+2)
    id1,id2,id3 = np.meshgrid(r,r,r, indexing='ij')
    return id1, id2, id3


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
    
    side = voxel.shape[0]
    assert voxel.shape[0] == voxel.shape[1] and voxel.shape[1] == voxel.shape[2], \
        'The voxel grid needs to be a cube. It is however %dx%dx%d.' % \
        (voxel.shape[0],voxel.shape[1],voxel.shape[2])

    if id1 is None or id2 is None or id3 is None:
        id1,id2,id3 = generate_indices(side)
        pass

    # add empty border for argmax
    # need to distinguish empty tubes
    v = np.zeros((side+2,side+2,side+2), dtype=np.uint8)
    v[1:-1,1:-1,1:-1] = voxel
        
    shape_layer = np.zeros((side, side, 6), dtype=np.uint16)
    
    # project depth to yz-plane towards negative x
    s1 = np.argmax(v, axis=0) # returns first occurence
    # project depth to yz-plane towards positive x
    s2 = np.argmax(v[-1::-1,:,:], axis=0) # same, but from other side
    s2 = side+1-s2 # correct for added border

    # set all empty tubes to 0    
    s1[s1 < 1] = side+2
    s2[s2 > side] = 0    
    shape_layer[:,:,0] = s1[1:-1,1:-1]
    shape_layer[:,:,1] = s2[1:-1,1:-1]
    
    # project depth to xz-plane towards negative y
    s1 = np.argmax(v, axis=1)
    # project depth to xz-plane towards positive y
    s2 = np.argmax(v[:,-1::-1,:], axis=1)
    s2 = side+1-s2

    s1[s1 < 1] = side+2
    s2[s2 > side] = 0    
    shape_layer[:,:,2] = s1[1:-1,1:-1]
    shape_layer[:,:,3] = s2[1:-1,1:-1]
    
    # project depth to xy-plane towards negative z
    s1 = np.argmax(v, axis=2)
    # project depth to xy-plane towards positive z
    s2 = np.argmax(v[:,:,-1::-1], axis=2)
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

    side = shape_layer.shape[0]

    if id1 is None or id2 is None or id3 is None:
        id1,id2,id3 = generate_indices(side)
        pass

    x0 = np.expand_dims(np.pad(shape_layer[:,:,0], ((1,1), (1,1)), 'constant'), axis=0)
    x1 = np.expand_dims(np.pad(shape_layer[:,:,1], ((1,1), (1,1)), 'constant'), axis=0)
    y0 = np.expand_dims(np.pad(shape_layer[:,:,2], ((1,1), (1,1)), 'constant'), axis=1)
    y1 = np.expand_dims(np.pad(shape_layer[:,:,3], ((1,1), (1,1)), 'constant'), axis=1)
    z0 = np.expand_dims(np.pad(shape_layer[:,:,4], ((1,1), (1,1)), 'constant'), axis=2)
    z1 = np.expand_dims(np.pad(shape_layer[:,:,5], ((1,1), (1,1)), 'constant'), axis=2)

    v0 = np.logical_and(x0 <= id1, id1 <= x1)
    v1 = np.logical_and(y0 <= id2, id2 <= y1)
    v2 = np.logical_and(z0 <= id3, id3 <= z1)

    return np.logical_and(np.logical_and(v0, v1), v2)[1:-1,1:-1,1:-1]


def encode_shape(voxel, num_layers=1, id1=None, id2=None, id3=None):
    """ Encodes a shape recursively into multiple layers.
    """

    side = voxel.shape[0]
    shape_layer = np.zeros((side, side, 6 * num_layers), dtype=np.uint16)
    decoded = np.zeros((side, side, side), dtype=np.uint8)
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

    side       = shape_layer.shape[0]
    num_layers = int(shape_layer.shape[2] / 6)
    decoded    = np.zeros((side, side, side), dtype=np.uint8)
    
    for i in range(num_layers):

        if i % 2 == 0:
            rec = decode_shapelayer(shape_layer[:,:,i*6:(i+1)*6], id1, id2, id3)
            decoded += rec
        else:
            rec = decode_shapelayer(shape_layer[:,:,i*6:(i+1)*6], id1, id2, id3)
            decoded -= rec
        pass

    return decoded
    

def convert_voxel(file):
    """ Converts a mat file containing a voxel grid
        into a shape layer representation.
    """

    d = sio.loadmat(file)
    voxel = d['voxel']

    shape_layer = encode_shape(voxel, args.numlayers, id1, id2, id3)
    decoded     = decode_shape(shape_layer, id1, id2, id3)
    
    print('%s: %.3f ' % (file, np.sum(np.logical_and(voxel, decoded)) / np.sum(np.logical_or(voxel, decoded))))
    
    sio.savemat(file[:-8]+'.shl.mat', 
        {'shapelayer':shape_layer},
        do_compression=True)
    pass


# def convert_files(dir, recursive=True):
#     """ Traverse directory recursively to convert files.
#         If recursive==False, only files in the directory dir are converted.
#     """
#     files = sorted(os.listdir(dir))
#     for file in files:
#         path = os.path.join(dir, file)
#         if os.path.isdir(path):
#             if recursive:
#                 convert_files(path, recursive)
#                 pass
#             pass
#         elif path.endswith('.vox.mat') and not os.path.exists(path[:-8]+'.shl.mat'):
#             convert_voxel(path)
#             pass
#         pass
#     pass


id1,id2,id3 = generate_indices(128)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Encodes and decodes shape layers.')
    parser.add_argument('directory', type=str)
    parser.add_argument('-s', '--side', type=int, choices=[32, 64, 128, 256], default=128, help='side length of the layers.')
    parser.add_argument('-n', '--numlayers', type=int, default=1, help='number of shape layers to use for encoding.')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively traverses the directory and converts all .vox.mat files.')
    args = parser.parse_args()

    if args.side != 128:
        id1,id2,id3 = generate_indices(args.side)
        pass
        
    convert_files(args.directory, '.vox.mat', convert_voxel, args.recursive)
    pass
