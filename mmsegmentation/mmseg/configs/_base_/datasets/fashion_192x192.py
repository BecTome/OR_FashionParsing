data_root = '../datasets/fashion/'
dataset_type = 'FashionBG'
crop_size = (192, 192)
scale_size = (600,600)
ratio_range = (0.5,2.0)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=ratio_range,
        scale=scale_size,
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=scale_size, type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

#test_cfg = dict(type='TestLoop')
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='images/train2020', seg_map_path='annotations/train2020'),
        data_root='../datasets/fashion/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=ratio_range,
                scale=scale_size,
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='FashionBG'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))


val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val2020', seg_map_path='annotations/val2020'),
        data_root='../datasets/fashion/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=scale_size, type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='FashionBG'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], 
    type='IoUMetric',
    classwise=True,
    ignore_index=0)

test_dataloader = val_dataloader

# train_cfg = dict(max_iters=200, type='IterBasedTrainLoop', val_interval=200)
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], 
    type='IoUMetric',
    classwise=True,
    ignore_index=0)

test_evaluator = val_evaluator


# tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
# val_cfg = dict(type='ValLoop')


