Matryoshka Networks: Predicting 3D Geometry via Nested Shape Layers
===============================================================================

This source code release accompanies the paper  

**Matryoshka Networks: Predicting 3D Geometry via Nested Shape Layers**  
Stephan Richter and Stefan Roth. In CVPR 2018.  
[**Paper**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Richter_Matryoshka_Networks_Predicting_CVPR_2018_paper.pdf) [**Supplemental**](http://openaccess.thecvf.com/content_cvpr_2018/Supplemental/1524-supp.pdf)

Please cite our work if you use code or data from this repository.

Requirements and set up
-------------------------------------------------------------------------------
Clone the repository via `git clone https://bitbucket.org/visinf/projects-2018-matryoshka ./matryoshka`.
Assuming you have set up an Anaconda or Miniconda environment, the following 
commands should get you started:

```
conda create -y -n matryoshka python=3.7
source activate matryoshka
conda install -y numpy scipy pillow
conda install -y pytorch torchvision -c pytorch
```


General notes
-------------------------------------------------------------------------------
The shape layer representation will work the better the more consistent your
input shapes are wrt. occlusions and nesting of 3D shapes. Meshes from 
different sources will probably be not consistent and in this case fewer layers 
are likely to work better. Keep in mind that few layers can often reconstruct 
remarkably well. If mesh quality varies in the dataset (as in ShapeNet), you 
are probably better off using a single shape layer and increasing the number of 
inner residual blocks (`--block`) or number of inner feature channels (`--ngf`).

Datasets
-------------------------------------------------------------------------------
This version supports ShapeNet in 2 versions: as used in 3DR2N2[1], and as used
in PTN[2]. It also supports the highres car experiment from OGN[3]. To run it 
with the respective datasets, please check the [DatasetCollector.py](DatasetCollector.py). It commonly
expects only a base directory including sub directories for shapes and 
renderings. The renderings are expected to be 128x128 images (see below).

Adding a new dataset should be straightforward:

1. process images with [crop_images.py](crop_images.py).
2. convert binvox to voxel, voxel to shape layer with [voxel2layer](voxel2layer_torch.py).
3. write an adapter inheriting from [DatasetCollector](DatasetCollector.py), which collects samples
	
Input images
-------------------------------------------------------------------------------
The networks are built to process input images of 128x128 pixels. 
For convenience, we provide a script that crops images to this size. 
Consequently, the *DatasetCollector* assumes that images are named `*.128.png` to 
indicate this format. Please have a look at [crop_images.py](crop_images.py) and 
[DatasetCollector](DatasetCollector.py).

References
-------------------------------------------------------------------------------
[1] C. B. Choy, D. Xu, J. Gwak, K. Chen, and S. Savarese. 
    3D-R2N2: A unified approach for single and multi-view 3D object 
    reconstruction. ECCV 2016
	
[2] X. Yan, J. Yang, E. Yumer, Y. Guo, and H. Lee. 
    Perspective transformer nets: Learning single-view 3D object reconstruction
    without 3D supervision. NIPS 2016
	
[3] M. Tatarchenko, A. Dosovitskiy, and T. Brox. 
    Octree generating networks: Efficient convolutional architectures for
    high-resolution 3D outputs. ICCV 2017
