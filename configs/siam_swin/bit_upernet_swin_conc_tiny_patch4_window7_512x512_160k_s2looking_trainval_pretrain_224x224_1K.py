'''
Author: Shuailin Chen
Created Date: 2021-08-25
Last Modified: 2021-09-09
	content: 
'''

_base_ = [
    './bit_upernet_swin_sub_tiny_patch4_window7_512x512_80k_s2looking_pretrain_224x224_1K.py'
]

model = dict(
    backbone=dict(
        merge_method='conc',
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536], 
        channels=1024, 
    ),
    auxiliary_head=dict(in_channels=768, channels=512)
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=4)
