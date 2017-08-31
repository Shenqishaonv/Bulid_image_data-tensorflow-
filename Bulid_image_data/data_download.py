#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:42:46 2017

@author: Lily
"""
# load libraries
import os
import tarfile
import _pickle as cPickle 
import numpy as np
import urllib.request
import scipy.misc
#%%
'''
Step1: cifar_link:  the CIFAR-10 link
       data_dir:    the directory we will store the data in
        objects:    the ten categories to reference 
                   for saving the image
'''
cifar_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
data_dir = '/users/Lily/data'
# if not exit the direction make a new one
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
objects = ['airplane','automobile','bird', 'cat', 'deer', 'dog', 
           'frog', 'horse', 'ship', 'truck']

#%%
'''
Step 2: Download the CIFAR-10.tar and un-tar the file

'''
target_file = os.path.join(data_dir,'cifar-10-python.tar.gz')
if not os.path.isfile(target_file):  
    print('\nCIFAR-10 is not found. Downloading CIFAR data\n')
    print('\nThis may take a few minutes, please wait\n')
    filename, headers =urllib.request.urlretrieve(cifar_link, target_file)
#Extract into memory
tar = tarfile.open(target_file)
tar.extractall(path =  data_dir)
tar.close()
#%%
'''
Step3: Creat two folders: train_dir and Validation_dir
       each of folder has ten sub-folders for each category
'''
train_folder = 'train_dir'

if not os.path.isdir(os.path.join(data_dir, train_folder,)):
    for i in range(10):
        folder = os.path.join(data_dir, train_folder, objects[i])
        os.makedirs(folder)
    else:
        pass

test_folder = 'validation_dir'
if not os.path.isdir(os.path.join(data_dir, test_folder)):
    for i in range(10):
        folder = os.path.join(data_dir,test_folder,objects[i])
        os.makedirs(folder)
#%%  
'''
Step4: load image from memory and store them in an image dictionary
'''
def load_batch_from_file(file):
    file_conn = open(file,'rb')
    image_dictionary = cPickle.load(file_conn, encoding = 'latin1')
    file_conn.close()
    return image_dictionary
#%%
'''
Step5: after get the dictionary, save each of the files in the correct location 
'''
def save_images_from_dict(image_dict, folder = 'data_dir'):
    for ix,label in enumerate(image_dict['labels']):
        folder_path = os.path.join(data_dir, folder, objects[label])
        filename = image_dict['filenames'][ix]
        #transform image data
        image_array = image_dict['data'][ix]
        image_array.resize([3,32,32])
        #save image
        output_location = os.path.join(folder_path, filename)
        scipy.misc.imsave(output_location, image_array.transpose())
#%%
'''
Step6:
with the preceding functions, we can loop through the downloaded files
save each image to correct location
'''
data_location = os.path.join(data_dir, 'cifar-10-batches-py')
train_names = ['data_batch_'+str(x) for x in range(1, 6)]
test_names = ['test_batch']
# sort train images
for file in train_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_images_from_dict(image_dict, folder=train_folder)
#sort test images
for file in test_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dic = load_batch_from_file(file_location)
    save_images_from_dict(image_dict, folder=test_folder)
#%%
'''
Step7: Creat label file
'''
cifar_label_file = os.path.join(data_dir, 'cifar-10-labels.txt')
print('Writing labels file, {}'.format(cifar_label_file))
with open (cifar_label_file, 'w') as labels_file:
    for item in objects:
        labels_file.write("{}\n".format(item))
