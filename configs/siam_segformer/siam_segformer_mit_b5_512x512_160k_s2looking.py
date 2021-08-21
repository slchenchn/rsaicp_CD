'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-08-21
	content: 
'''
_base_ = [
    './siam_segformer_mit_bo_512x512_160k_s2looking'
]


model = dict(
    backbone=dict(
            pretrained='checkpoints/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth',
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512])
)