_base_ = ['../../deeplabv3/deeplabv3_r18b-d8_4xb2-80k_cityscapes-512x1024.py']

cfg = _base_

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.crop_size = (192, 192)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 47
cfg.model.auxiliary_head.num_classes = 47

# Modify dataset type and path
cfg.dataset_type = 'FashionBG'
cfg.data_root = '../datasets/fashion/'

cfg.train_dataloader.batch_size = 64

# Information about transforms in https://mmcv.readthedocs.io/en/latest/api/transforms.html
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # If scale is a tuple, image_x follows U(ratio_range[0], ratio_range[1]) * scale[0] (analogous for y)
    dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(320, 240), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path='images/train2020', seg_map_path='annotations/train2020')
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
# cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path='images/val2020', seg_map_path='annotations/val2020')
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
# cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

cfg.val_evaluator = dict(iou_metrics=['mIoU','mDice'], type='IoUMetric', classwise=True, ignore_index=0)



cfg.test_dataloader = cfg.val_dataloader
# cfg.test_evaluator = cfg.val_evaluator

# Load the pretrained weights
cfg.load_from = 'deeplabv3_r18b-d8_512x1024_80k_cityscapes_20201225_094144-46040cef.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/tutorial'

cfg.train_cfg.max_iters = 200
cfg.train_cfg.val_interval = 200
cfg.default_hooks.logger.interval = 100
cfg.default_hooks.checkpoint.interval = 200

# Set seed to facilitate reproducing the result
cfg['randomness'] = dict(seed=0)

# Let's have a look at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')

work_dir = f'./work_dirs/schedule_{_base_.train_cfg.max_iters}/resol_{_base_.crop_size[0]}' #{_base_.load_from.split()[0]}/