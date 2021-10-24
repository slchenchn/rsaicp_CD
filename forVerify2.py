'''
Author: Shuailin Chen
Created Date: 2021-10-24
Last Modified: 2021-10-24
	content: 
'''

import argparse


if __name__=='__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--a')
    argparser.add_argument('--b')
    argparser.add_argument('--c')

    arg = argparser.parse_args()

    print(f'a: {arg.a}, b: {arg.b}, c:{arg.c}')