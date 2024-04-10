# OR_FashionParsing

## Get Data

You can download the data from the following links (Fashionpedia 2020):

[Training data](https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip)
[Validation and Test data](https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip)


### Get Semantic Segmentation of Fashion Images

[Train Instance Attributes](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json)
[Validation Instance Attributes](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json)

**Easiest way to get the data is to use the following command:**

```bash
python get_data.py
```

**Visualize masks**

Go to [notebooks/pycocoDemo.ipynb](notebooks/pycocoDemo.ipynb)

**File Structure**

```
.
├── datasets
│   └── fashion
│       ├── annotations
│       │   ├── train2020
│       │   └── val2020
│       ├── images
│       │   ├── test2020
│       │   ├── train2020
│       │   └── val2020
```

## Configuration Files
They can be found in the [config_files](config_files) folder. They must be copied into the mmsegmentation/configs/fashion folder.

To run them:
    
```bash
python tools/train.py configs/fashion/<architecture>/<config_file>.py
```

For example:

```bash
python tools/train.py configs/fashion/deeplabv3/fashion_deeplabv3_192x192.py
```

In our case, we register the dataset in the mmseg registry. This is done in the [mmseg/datasets/fashion.py](mmseg/datasets/fashion.py) file. Important to add the import in the [mmseg/datasets/__init__.py](mmseg/datasets/__init__.py) file.

Copy and paste both files in the mmseg/datasets folder.


## Issues with installations

- mmcv : error: [Errno 2] No such file or directory: '/usr/lib/cuda/bin/nvcc'

```bash
conda create --name openmmlab2 python=3.10.12 -y
conda activate openmmlab2
pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
mim install mmcv==2.0.0rc4

!rm -rf mmsegmentation
!git clone https://github.com/open-mmlab/mmsegmentation.git 
%cd mmsegmentation
!pip install -e .
```