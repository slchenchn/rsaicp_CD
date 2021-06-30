'''
Author: Shuailin Chen
Created Date: 2021-06-18
Last Modified: 2021-06-30
	content: 
'''
_base_ = ['../_base_/models/siamunet_conc.py', 
		'../_base_/datasets/s2looking.py', 
		'../_base_/schedules/schedule_100.py',
		'../_base_/default_runtime.py',
		]

model=dict(
	decode_head=dict(
		loss_decode=dict(
			# class_weight=[3.52291701e-04, 4.09197173e-02, 8.91839992e-02, 8.69543992e-01],
			# class_weight=[3e-04, 6e-02, 12e-02, 8e-01],
			# class_weight=[5e-04, 5e-02, 12e-02, 8e-01],
			class_weight=[5e-04, 5e-02, 12e-02, 8e-01],
		)
	)
)

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-3, by_epoch=False)