'''
Author: Shuailin Chen
Created Date: 2021-08-18
Last Modified: 2021-08-18
	content: 
'''

from torch import nn

from ..builder import build_post_process
from .decode_head import BaseDecodeHead


class BaseDecWithPostProcess(BaseDecodeHead):
    ''' Base class for decode head with post processing layer between original decode head the loss
    '''
    def __init__(self, *args, **kargs):
        post_process_args = kargs.pop('post_process', None)
        super().__init__(*args, **kargs)

        if post_process_args is None:
            self.post_process = nn.Identity()
        else:
            self.post_process = build_post_process(post_process_args)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        seg_logits = self.post_process(seg_logits)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs)
        seg_logits = self.post_process(seg_logits)

        return seg_logits
