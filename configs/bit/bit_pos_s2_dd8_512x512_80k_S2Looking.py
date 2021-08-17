'''
Author: Shuailin Chen
Created Date: 2021-08-17
Last Modified: 2021-08-17
	content: 
'''

_base_= [
    '../_base_/models/bit_pos_s4_dd8.py',
    '../_base_/datasets/s2looking.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    )

evaluation = dict(metric=['mFscore', 'mFscoreCD'])

