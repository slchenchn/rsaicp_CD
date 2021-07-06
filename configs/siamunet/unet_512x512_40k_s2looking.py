'''
Author: Shuailin Chen
Created Date: 2021-06-18
Last Modified: 2021-07-05
	content: 
'''
_base_ = ['../_base_/models/siamunet_conc.py', 
		'../_base_/datasets/s2looking.py', 
		'../_base_/schedules/schedule_40k.py',
		'../_base_/default_runtime.py',
		]

model=dict(
	decode_head=dict(
		loss_decode=dict(
			# class_weight=[3.52291701e-04, 4.09197173e-02, 8.91839992e-02, 8.69543992e-01],
			# class_weight=[5e-04, 4e-02, 9e-02, 8e-01],
			loss_weight = 1,
		)
	)
)

optimizer = dict(_delete_=True,
				type='Adam', 
				lr=1e-3, 
			# momentum=0.9, weight_decay=0.0000
			)

# lr_config = dict(_delete_=True,
				# policy='fixed')
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-2, by_epoch=False)

evaluation = dict(metric=['mFscore', 'mFscoreCD'])
