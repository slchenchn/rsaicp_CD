'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-08-24
	content: sd
'''
_base_ = [
    '../_base_/models/siam_segformer_mit-b0_sub.py', '../_base_/datasets/s2looking.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    pretrained='pretrain/mit_b0.pth', decode_head=dict(num_classes=4))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
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

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    )

evaluation = dict(metric=['mFscore', 'mFscoreCD'])