'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-07-16
	content: 
'''
_base_ = [
    './siam_hr18_512x512_40k_s2looking.py'
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[96, 192, 384, 768], channels=sum([96, 192, 384, 768])))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    )