'''
Author: Shuailin Chen
Created Date: 2021-10-23
Last Modified: 2021-10-24
	content: 
'''

from gpustat import GPUStatCollection
import time
import argparse

def wati_for_gpu(required_GB, interval=10):
    ''' Wait for available gpu, that is, exit until free gpu memory is greater than required.

    Args:
        required (int): required gpu memory in GB.
        interval (int): repeated interval in seconds. Default: 10
    '''

    if required_GB is not None:
        required_MB = required_GB * 1024

    gpu_stats = GPUStatCollection.new_query()
    mem_free = gpu_stats.gpus

    print()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--required', type=int, default=None)
    argparser.add_argument('--interval', type=int, default=10)

    args = argparser.parse_args()
    wati_for_gpu(args.required, args.interval)
