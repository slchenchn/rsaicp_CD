'''
Author: Shuailin Chen
Created Date: 2021-06-15
Last Modified: 2021-06-29
	content: 
'''
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=10)
evaluation = dict(interval=1, metric=['mFscore', 'mFscoreCD'] )
