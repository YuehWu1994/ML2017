#!/usr/bin/env python
# -- coding: utf-8 --

import os
from termcolor import colored, cprint
from keras.utils.vis_utils import plot_model
from keras.models import load_model

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def main():
    """
    parser = argparse.ArgumentParser(prog='plot_model.py',
            description='Plot the model.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','easy','strong'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=80)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    """
    model_name = "0.67679.h5"
    model_path = os.path.join(base_dir, model_name)
    emotion_classifier = load_model(model_path)
    emotion_classifier.summary()
    plot_model(emotion_classifier,to_file='modelp2.png')

if __name__ == "__main__":
    main()