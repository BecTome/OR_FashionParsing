cfg = dict(
    crop_size=(
        192,
        192,
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    data_root='../datasets/fashion/',
    dataset_type='FashionBG',
    default_hooks=dict(
        checkpoint=dict(by_epoch=False, interval=16000, type='CheckpointHook'),
        logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
        param_scheduler=dict(type='ParamSchedulerHook'),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        timer=dict(type='IterTimerHook'),
        visualization=dict(type='SegVisualizationHook')),
    default_scope='mmseg',
    env_cfg=dict(
        cudnn_benchmark=True,
        dist_cfg=dict(backend='nccl'),
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)),
    img_ratios=[
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
    ],
    load_from=
    './checkpoints/upernet_deit-s16_mln_512x512_160k_ade20k_20210621_161021-fb9a5dfb.pth',
    log_level='INFO',
    log_processor=dict(by_epoch=False),
    model=dict(
        auxiliary_head=dict(
            align_corners=False,
            channels=256,
            concat_input=False,
            dropout_ratio=0.1,
            in_channels=768,
            in_index=3,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=47,
            num_convs=1,
            type='FCNHead'),
        backbone=dict(
            act_cfg=dict(type='GELU'),
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            drop_rate=0.0,
            embed_dims=768,
            img_size=(
                512,
                512,
            ),
            in_channels=3,
            interpolate_mode='bicubic',
            mlp_ratio=4,
            norm_cfg=dict(eps=1e-06, type='LN'),
            norm_eval=False,
            num_heads=12,
            num_layers=12,
            out_indices=(
                2,
                5,
                8,
                11,
            ),
            patch_size=16,
            qkv_bias=True,
            type='VisionTransformer',
            with_cls_token=True),
        data_preprocessor=dict(
            bgr_to_rgb=True,
            mean=[
                135.43535295,
                125.5206132,
                122.8418554,
            ],
            pad_val=0,
            seg_pad_val=255,
            size=(
                192,
                192,
            ),
            std=[
                64.70508792,
                63.73913779,
                62.8355091,
            ],
            type='SegDataPreProcessor'),
        decode_head=dict(
            align_corners=False,
            channels=512,
            dropout_ratio=0.1,
            in_channels=[
                768,
                768,
                768,
                768,
            ],
            in_index=[
                0,
                1,
                2,
                3,
            ],
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=47,
            pool_scales=(
                1,
                2,
                3,
                6,
            ),
            type='UPerHead'),
        neck=dict(
            in_channels=[
                768,
                768,
                768,
                768,
            ],
            out_channels=768,
            scales=[
                4,
                2,
                1,
                0.5,
            ],
            type='MultiLevelNeck'),
        pretrained='pretrain/jx_vit_base_p16_224-80ecf9dd.pth',
        test_cfg=dict(mode='whole'),
        train_cfg=dict(),
        type='EncoderDecoder'),
    norm_cfg=dict(requires_grad=True, type='SyncBN'),
    optim_wrapper=dict(
        clip_grad=None,
        optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
        type='OptimWrapper'),
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    param_scheduler=[
        dict(
            begin=0,
            by_epoch=False,
            end=160000,
            eta_min=0.0001,
            power=0.9,
            type='PolyLR'),
    ],
    randomness=dict(seed=0),
    resume=False,
    test_cfg=dict(type='TestLoop'),
    test_dataloader=dict(
        batch_size=1,
        dataset=dict(
            ann_file=0,
            data_prefix=dict(
                img_path='images/val2020', seg_map_path='annotations/val2020'),
            data_root='../datasets/fashion/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(keep_ratio=True, scale=(
                    320,
                    240,
                ), type='Resize'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ],
            type='FashionBG'),
        num_workers=4,
        persistent_workers=True,
        sampler=dict(shuffle=False, type='DefaultSampler')),
    test_evaluator=dict(
        classwise=True,
        ignore_index=0,
        iou_metrics=[
            'mIoU',
            'mDice',
        ],
        type='IoUMetric'),
    test_pipeline=[
        dict(type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            320,
            240,
        ), type='Resize'),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs'),
    ],
    train_cfg=dict(
        max_iters=160000, type='IterBasedTrainLoop', val_interval=16000),
    train_dataloader=dict(
        batch_size=4,
        dataset=dict(
            ann_file=0,
            data_prefix=dict(
                img_path='images/train2020',
                seg_map_path='annotations/train2020'),
            data_root='../datasets/fashion/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    transforms=[
                        dict(limit=20, p=0.5, type='Rotate'),
                        dict(
                            p=0.5,
                            rotate_limit=20,
                            scale_limit=0.1,
                            shift_limit=0.1,
                            type='ShiftScaleRotate'),
                    ],
                    type='Albu'),
                dict(
                    keep_ratio=True,
                    ratio_range=(
                        0.5,
                        2.0,
                    ),
                    scale=(
                        320,
                        240,
                    ),
                    type='RandomResize'),
                dict(
                    cat_max_ratio=0.75,
                    crop_size=(
                        192,
                        192,
                    ),
                    type='RandomCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackSegInputs'),
            ],
            type='FashionBG'),
        num_workers=4,
        persistent_workers=True,
        sampler=dict(shuffle=True, type='InfiniteSampler')),
    train_pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            transforms=[
                dict(limit=20, p=0.5, type='Rotate'),
                dict(
                    p=0.5,
                    rotate_limit=20,
                    scale_limit=0.1,
                    shift_limit=0.1,
                    type='ShiftScaleRotate'),
            ],
            type='Albu'),
        dict(
            keep_ratio=True,
            ratio_range=(
                0.5,
                2.0,
            ),
            scale=(
                320,
                240,
            ),
            type='RandomResize'),
        dict(cat_max_ratio=0.75, crop_size=(
            192,
            192,
        ), type='RandomCrop'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PackSegInputs'),
    ],
    tta_model=dict(type='SegTTAModel'),
    tta_pipeline=[
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
    ],
    val_cfg=dict(type='ValLoop'),
    val_dataloader=dict(
        batch_size=1,
        dataset=dict(
            ann_file=0,
            data_prefix=dict(
                img_path='images/val2020', seg_map_path='annotations/val2020'),
            data_root='../datasets/fashion/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(keep_ratio=True, scale=(
                    320,
                    240,
                ), type='Resize'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ],
            type='FashionBG'),
        num_workers=4,
        persistent_workers=True,
        sampler=dict(shuffle=False, type='DefaultSampler')),
    val_evaluator=dict(
        classwise=True,
        ignore_index=0,
        iou_metrics=[
            'mIoU',
            'mDice',
        ],
        type='IoUMetric'),
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    visualizer=dict(
        name='visualizer',
        type='SegLocalVisualizer',
        vis_backends=[
            dict(type='LocalVisBackend'),
        ]),
    work_dir='./work_dirs/tutorial')
crop_size = (
    192,
    192,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        135.43535295,
        125.5206132,
        122.8418554,
    ],
    pad_val=0,
    seg_pad_val=255,
    std=[
        64.70508792,
        63.73913779,
        62.8355091,
    ],
    type='SegDataPreProcessor')
data_root = '../datasets/fashion/'
dataset_type = 'FashionBG'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=16000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = './checkpoints/upernet_deit-s16_mln_512x512_160k_ade20k_20210621_161021-fb9a5dfb.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=768,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=47,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_rate=0.0,
        embed_dims=768,
        img_size=(
            512,
            512,
        ),
        in_channels=3,
        interpolate_mode='bicubic',
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        norm_eval=False,
        num_heads=12,
        num_layers=12,
        out_indices=(
            2,
            5,
            8,
            11,
        ),
        patch_size=16,
        qkv_bias=True,
        type='VisionTransformer',
        with_cls_token=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            135.43535295,
            125.5206132,
            122.8418554,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            192,
            192,
        ),
        std=[
            64.70508792,
            63.73913779,
            62.8355091,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            768,
            768,
            768,
            768,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=47,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    neck=dict(
        in_channels=[
            768,
            768,
            768,
            768,
        ],
        out_channels=768,
        scales=[
            4,
            2,
            1,
            0.5,
        ],
        type='MultiLevelNeck'),
    pretrained='pretrain/jx_vit_base_p16_224-80ecf9dd.pth',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
name = 'vit'
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=160000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=0,
        data_prefix=dict(
            img_path='images/val2020', seg_map_path='annotations/val2020'),
        data_root='../datasets/fashion/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                320,
                240,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='FashionBG'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    classwise=True,
    ignore_index=0,
    iou_metrics=[
        'mIoU',
        'mDice',
    ],
    type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        320,
        240,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=16000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=0,
        data_prefix=dict(
            img_path='images/train2020', seg_map_path='annotations/train2020'),
        data_root='../datasets/fashion/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                transforms=[
                    dict(limit=20, p=0.5, type='Rotate'),
                    dict(
                        p=0.5,
                        rotate_limit=20,
                        scale_limit=0.1,
                        shift_limit=0.1,
                        type='ShiftScaleRotate'),
                ],
                type='Albu'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    320,
                    240,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    192,
                    192,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='FashionBG'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        transforms=[
            dict(limit=20, p=0.5, type='Rotate'),
            dict(
                p=0.5,
                rotate_limit=20,
                scale_limit=0.1,
                shift_limit=0.1,
                type='ShiftScaleRotate'),
        ],
        type='Albu'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            320,
            240,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        192,
        192,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
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
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=0,
        data_prefix=dict(
            img_path='images/val2020', seg_map_path='annotations/val2020'),
        data_root='../datasets/fashion/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                320,
                240,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='FashionBG'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    classwise=True,
    ignore_index=0,
    iou_metrics=[
        'mIoU',
        'mDice',
    ],
    type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/vit/schedule_160000/resol_192'
