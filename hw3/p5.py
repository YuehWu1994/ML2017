#!/usr/bin/env python
# -- coding: utf-8 --

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np

base_dir = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw3/reportSimCode'

p5_dir = os.path.join(base_dir, 'p5')
if not os.path.exists(p5_dir):
    os.makedirs(p5_dir)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func, idx):
    
    step = 0.1
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data])
        input_image_data += grads_value * step   
    #input_image_data = np.sum(input_image_data)
    return input_image_data, np.abs(np.sum(input_image_data))

def main():
    model_name = "0.67177.h5"
    model_path = os.path.join(base_dir, model_name)
    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input
    name_ls = ["conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_5","conv2d_6"]
    #name_ls = ["dense_2","dense_3"]
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    for cnt, c in enumerate(collect_layers):
        # NUM_STEPS//RECORD_FREQ -> 1        
        filter_imgs = [[] for i in range(1)]
        #
        nb_filter = 32
        filter_imgs[0] = [[] for i in range(nb_filter)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])
            
            ###
            "You need to implement it."
            num_step = 20
            filter_imgs[0][filter_idx] = [[] for i in range(2)]
            tmp, tmp2 = grad_ascent(num_step, input_img_data, iterate, filter_idx)
            filter_imgs[0][filter_idx][0] = np.reshape(tmp,(1,48,48))
            filter_imgs[0][filter_idx][1] = tmp2
            ###

        for it in range(1):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16, 16, i+1)
                ax.imshow(np.reshape(filter_imgs[it][i][0],(48,48)), cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                #plt.xlabel(np.round(filter_imgs[it][i][1],3))
                plt.xlabel(np.round(filter_imgs[it][i][1],3))
                plt.tight_layout()
                # RECORD_FREQ = 1
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*1))
            #format(store_path)
            img_path = os.path.join(p5_dir, '{}-{}'.format(p5_dir, name_ls[cnt]))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            # RECORD_FREQ = 1
            fig.savefig(os.path.join(img_path,'e{}'.format(it*1)))

if __name__ == "__main__":
    main()