
#%%

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
import random


#%%
def _find_image_files(data_dir, labels_file):#

  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.
    Assumes that the image data set resides in JPEG files located in
    the following directory structure.

        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg
        
      where 'dog' is the label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
      
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    matching_files = tf.gfile.Glob(jpeg_file_path)#A list of strings containing filenames that match the given pattern

    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels# 
#%%

def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#%%

def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train' 
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
    
    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)
    
    #images = np.array(images)
    #import pdb; pdb.set_trace()
    if np.shape(images)[0] != n_samples:#如果image的个数与标签的个数不一致 返回错误
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))
    
    
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i]) 
            # type(image) must be array  
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                            'label':int64_feature(label),
                            'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
    

#%%

def read_and_decode(tfrecords_file, batch_size):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file TFRecord文件的地址
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file 使用函数做一个file queue
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.
    
    image = tf.reshape(image, [28, 28])
    label = tf.cast(img_features['label'], tf.int32)    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = 2000)
    return image_batch, tf.reshape(label_batch, [batch_size])

    

  
#%% Convert data to TFRecord

test_dir = '/users/Lily/data/validation_dir'
save_dir = '/users/Lily/data/validation_dir'
labels_file = '/users/Lily/data/validation_dir/labels.txt'#label.txt: list of strings; each string is the class

#BATCH_SIZE = 25
#Convert test data: you just need to run it ONCE !
name_test = 'validation'
#images, labels = get_file(test_dir)
images, _ , labels = _find_image_files(test_dir, labels_file)
convert_to_tfrecord(images, labels, save_dir, name_test)








    











