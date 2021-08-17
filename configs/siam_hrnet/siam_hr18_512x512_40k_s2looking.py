'''
Author: Shuailin Chen
Created Date: 2021-07-06
Last Modified: 2021-07-16
	content: 
'''
_base_ = [
    '../_base_/models/siam_hr18.py', '../_base_/datasets/s2looking.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]


evaluation = dict(metric=['mFscore', 'mFscoreCD'])