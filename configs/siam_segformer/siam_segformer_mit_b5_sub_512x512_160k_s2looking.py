'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-08-22
	content: 
'''
_base_ = [
    './siam_segformer_mit_b0_512x512_160k_s2looking.py'
]


model = dict(
    pretrained='pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512])
)


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )