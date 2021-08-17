'''
Author: Shuailin Chen
Created Date: 2021-08-17
Last Modified: 2021-08-17
	content: Bitemporal transformer, with relative positional embedding, encoder depth=1, decoder depth=8, add after the 4th stage of ResNet18n namely the 'base_transformer_pos_s4_dd8' of original implementation of the author
'''

norm_cfg = dict(type='BN', requires_grad=True)
# norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained='checkpoints/best_ckpt.pt',
    backbone=dict(
        type = 'BASE_Transformer',
        input_nc=3, 
        output_nc=4, 
        token_len=4, 
        resnet_stages_num=4,
        with_pos='learned', 
        enc_depth=1, 
        dec_depth=8,
    ),
    decode_head=dict(
        type='NewFCNHead',
        in_channels=2,
        channels=2,
        in_index=0,
        num_convs=0,
        concat_input=False,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        input_transform = None,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=1)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)