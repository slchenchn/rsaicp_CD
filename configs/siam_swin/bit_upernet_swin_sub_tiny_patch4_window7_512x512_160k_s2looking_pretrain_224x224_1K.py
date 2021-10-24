'''
Author: Shuailin Chen
Created Date: 2021-08-25
Last Modified: 2021-09-18
	content: 
'''

_base_ = [
    '../_base_/models/bit_upernet_swin_sub.py', '../_base_/datasets/s2looking.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    backbone=dict(
        merge_method='sub',
        pretrained=\
        'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth', # noqa
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        pretrain_style='official'
    ),
    decode_head=dict(
        num_classes=4
    ),
    auxiliary_head=dict(num_classes=4)
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8, workers_per_gpu=8)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


evaluation = dict(metric=['mFscore', 'mFscoreCD'])