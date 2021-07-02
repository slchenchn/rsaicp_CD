'''
Author: Shuailin Chen
Created Date: 2021-06-18
Last Modified: 2021-06-30
	content: 
'''
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='SiamUnet_conc',
        in_channels = 3,
        out_channels = 16,
		drop_p = 0,
        
        # depth=50,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # dilations=(1, 1, 2, 4),
        # strides=(1, 2, 1, 1),
        # norm_cfg=norm_cfg,
        # norm_eval=False,
        # style='pytorch',
        # contract_dilation=True
        ),
    # decode_head=dict(
    #     type='ASPPHead',
    #     in_channels=2048,
    #     in_index=3,
    #     channels=512,
    #     dilations=(1, 12, 24, 36),
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
        type='FCNHead',
        in_channels=16,
        channels=16,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
