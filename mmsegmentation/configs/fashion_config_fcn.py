_base_ = [
    '_base_/models/fcn_unet_s5-d16.py','_base_/datasets/fashion.py', 
    '_base_/schedules/schedule_1k.py', '_base_/fashion_runtime.py'
]

model = dict(
    decode_head=dict(num_classes=47),
    auxiliary_head=dict(num_classes=47),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=4,
# )
# load_from = '../checkpoints/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth'
work_dir = f'./work_dirs/fcn/schedule_{_base_.train_cfg.max_iters}/resol_{_base_.crop_size[0]}'