'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-07-07
	content: 
'''
_base_ = [
    '../_base_/models/siam_hr18.py', '../_base_/datasets/s2looking.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[96, 192, 384, 768], channels=sum([48, 96, 192, 384])))


optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

evaluation = dict(metric=['mFscore', 'mFscoreCD'])