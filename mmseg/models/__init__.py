'''
Author: Shuailin Chen
Created Date: 2021-06-13
Last Modified: 2021-07-06
	content: 
'''
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, build_backbone,
                      build_head, build_loss, build_segmentor)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .siambackbones import *

# __all__ = [
#     'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'build_backbone',
#     'build_head', 'build_loss', 'build_segmentor'
# ]
