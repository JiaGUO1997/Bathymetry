# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:06:19 2024

@author: Jia_G
"""

import numpy as np
import scipy.io as scio
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--command', help = 'the command used to run the progam')
    
    parser.add_argument('--task',help ='task to perform. Must load a pretrained model if task is "play" or "eval"',
                        choices = ['play','test','train','pretrain','study','vali', 'pretreatement', 'debug', 'rescheck', 't1est_point'],default='rescheck')
    parser.add_argument('--network_nomber', help = 'nomber of network choosen to be trained', default= '1')
    parser.add_argument('--lossfn', help = 'the loss function chossen to be used to train the network', 
                        choices = ['MSEpercent', 'MSELoss', 'L1loss', 'MSEcomb'], default='MSELoss')
    parser.add_argument('--type_input', help = 'the type of the input of the network', 
                        choices = ['n', 'residual1', 'residual4'], default='residual1')
    parser.add_argument('--size_image', default = '64')
    parser.add_argument('--epochs', default = '100')
    parser.add_argument('--nomber', default = '0')
    parser.add_argument('--show', default = 1)
    parser.add_argument('--num_model', default = '')
    parser.add_argument('--input_nomber', nargs='+', type=int, default = [])
    parser.add_argument('--factor', type=float, default = 1)
    parser.add_argument('--best', default='99')

    args = parser.parse_args()
    command = {'task' : args.task}
    
    try:
        command['lossfn'] = args.lossfn
    except:
        command['lossfn'] = 'MSELoss'
    command['input_nomber'] = args.input_nomber
    try:
        command['factor'] = args.factor
    except:
        command['factor'] = 1.0
    try:
        command['show'] = args.show 
    except:
        command['show'] = 0
    
    try:
        command['size_image'] = args.size_image
    except:
        command['size_image'] = '8'
    
    try:
        command['epochs'] = args.epochs
    except:
        command['epochs'] = '200'
        
    try:
        command['type_input'] = args.type_input
    except:
        command['type_input'] = ''
    
    try:
        command['network_nomber'] = args.network_nomber
    except:
        command['network_nomber'] = '1'
        
    try:
        command['nomber'] = args.nomber
    except:
        command['nomber'] = '0'
    command['num_model'] = args.num_model
    command['best'] = args.best

    scio.savemat(args.num_model + args.task + '.mat', command)