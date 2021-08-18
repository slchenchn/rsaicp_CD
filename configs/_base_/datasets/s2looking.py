'''
Author: Shuailin Chen
Created Date: 2021-06-14
Last Modified: 2021-08-17
	content: 
'''
# dataset settings
dataset_type = 'S2LookingDataset'
data_root = 'data/S2Looking/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
img_scale=(1024, 1024)
train_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='TimeShuffle'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortionMultiImages'),
    dict(type='GaussianBlur'),
    dict(type='RandomRotate', degree=180, prob=0.5),    
    dict(type='NormalizeMultiImages', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='NormalizeMultiImages', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img1_dir='train/Image1',
        img2_dir='train/Image2',
        ann_dir='train/label_index',
        pipeline=train_pipeline,
        if_visualize=False
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img1_dir='val/Image1',
        img2_dir='val/Image2',
        ann_dir='val/label_index',
        pipeline=test_pipeline
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img1_dir='Image1',
        img2_dir='Image2',
        # ann_dir='val/label_index',
        ann_dir='label_index',
        pipeline=test_pipeline
        ),
    )
