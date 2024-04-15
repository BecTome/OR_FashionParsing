_base_ = [
    '_base_/models/fashion.py','_base_/datasets/fashion.py', 
    '_base_/schedules/schedule_40k.py', '_base_/fashion_runtime.py'
]

work_dir = f'./work_dirs/{_base_.load_from.split()[0]}/schedule_{_base_.train_cfg.max_iters}/resol_{_base_.crop_size[0]}'