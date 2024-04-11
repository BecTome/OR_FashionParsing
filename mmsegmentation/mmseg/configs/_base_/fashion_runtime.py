
log_level = 'INFO'
log_processor = dict(by_epoch=False)
randomness = dict(seed=0)
resume = False

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

work_dir = './work_dirs/tutorial'