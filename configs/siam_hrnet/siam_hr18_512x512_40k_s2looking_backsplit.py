'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-08-18
	content: siamese HR18 with background splitting
'''
_base_ = [
    './siam_hr18_512x512_40k_s2looking.py'
]

model = dict(
    decode_head=dict(
        post_process=dict(
            type='SetConstValue',
            position=0,
            value=0.1,
        )
    )
)