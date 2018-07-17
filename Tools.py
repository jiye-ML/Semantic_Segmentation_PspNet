import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import numpy as np
from matplotlib import pyplot as plt
import cv2

import sys
import time
import os
import scipy.io as sio
from PIL import Image


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# colour map cityscapes
# label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
#                  # 0 = road, 1 = sidewalk, 2 = building
#     , (102, 102, 156), (190, 153, 153), (153, 153, 153)
#                  # 3 = wall, 4 = fence, 5 = pole
#     , (250, 170, 29), (219, 219, 0), (106, 142, 35)
#                  # 6 = traffic light, 7 = traffic sign, 8 = vegetation
#     , (152, 250, 152), (69, 129, 180), (219, 19, 60)
#                  # 9 = terrain, 10 = sky, 11 = person
#     , (255, 0, 0), (0, 0, 142), (0, 0, 69)
#                  # 12 = rider, 13 = car, 14 = truck
#     , (0, 60, 100), (0, 79, 100), (0, 0, 230)
#                  # 15 = bus, 16 = train, 17 = motocycle
#     , (119, 10, 32)]
# 18 = bicycle

# # voc
label_colours = [(0, 0, 0)
                 # 0=background
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def print_ckpt(ckpt_path):
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()

        for key in var_to_shape_map:
            print("tensor_name: ", key)
            # print(reader.get_tensor(key))
            pass
        pass

    pass


class Visualize:

    @staticmethod
    def _discrete_matshow_adaptive(data, labels_names=None, title=""):

        fig_size = [7, 6]
        plt.rcParams["figure.figsize"] = fig_size
        cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)
        mat = plt.matshow(data,
                          cmap=cmap,
                          vmin=np.min(data) - .5,
                          vmax=np.max(data) + .5)

        cax = plt.colorbar(mat,
                           ticks=np.arange(np.min(data), np.max(data) + 1))

        if labels_names:
            cax.ax.set_yticklabels(labels_names)

        if title:
            plt.suptitle(title, fontsize=15, fontweight='bold')

        fig = plt.gcf()
        fig.savefig('data/tmp.jpg', dpi=300)
        img = cv2.imread('data/tmp.jpg')
        return img

    @staticmethod
    def visualize_segmentation_adaptive(predictions, segmentation_class_lut, title="Segmentation"):

        # TODO: add non-adaptive visualization function, where the colorbar
        # will be constant with names

        unique_classes, relabeled_image = np.unique(predictions, return_inverse=True)

        relabeled_image = relabeled_image.reshape(predictions.shape)

        labels_names = []

        for index, current_class_number in enumerate(unique_classes):
            labels_names.append(str(index) + ' ' + segmentation_class_lut[current_class_number])

        im = Visualize._discrete_matshow_adaptive(data=relabeled_image, labels_names=labels_names, title=title)
        return im

    pass


class ImageTools:

    @staticmethod
    def _read_labelcolours(matfn):
        mat = sio.loadmat(matfn)
        color_table = mat['colors']
        shape = color_table.shape
        color_list = [tuple(color_table[i]) for i in range(shape[0])]

        return color_list

    @staticmethod
    def decode_labels(mask, num_images, num_classes):
        if num_classes == 150:
            color_table = ImageTools._read_labelcolours('./utils/color150.mat')
        else:
            color_table = label_colours

        shape = tf.shape(mask)
        # 将标签转换为可视化
        color_mat = tf.constant(color_table, dtype=tf.float32)
        onehot_output = tf.one_hot(mask[:num_images], depth=num_classes)
        onehot_output = tf.reshape(onehot_output, (-1, num_classes))
        pred = tf.matmul(onehot_output, color_mat)
        return tf.reshape(tf.cast(pred, tf.uint8), (num_images, shape[1], shape[2], 3))

    # 这个方法可以子啊 py_fn 中使用
    @staticmethod
    def decode_labels2(mask, num_images, num_classes):
        if num_classes == 150:
            color_table = ImageTools._read_labelcolours('./utils/color150.mat')
        else:
            color_table = label_colours

        n, h, w, c = mask.shape
        assert (n >= num_images),'Batch size {} should < number of images to save {}.'.format(n, num_images)
        # 将标签转换为可视化
        color_mat = np.array(color_table, dtype=np.float32)
        # 将255 转化为0
        mask[mask[:, :, :] > num_classes] = 0
        onehot_output = np.zeros((num_images, h, w, num_classes))  # 3个样本，4个类别
        onehot_output[:, :, :, mask[:num_images, : , :]] = 1  # 非零列赋值为1
        onehot_output = np.reshape(onehot_output, (-1, num_classes))
        pred = np.matmul(onehot_output, color_mat)
        return np.reshape(pred, (num_images, h, w, 3))

    # 最慢的方式
    @staticmethod
    def decode_labels3(mask, num_images=1, num_classes=21):
        n, h, w, c = mask.shape
        assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
            n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))  # Size is given as a (width, height)-tuple.
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = label_colours[k]
            outputs[i] = np.array(img)
        return outputs

    @staticmethod
    def prepare_label(input_batch, new_size, num_classes, one_hot=True):
        with tf.name_scope('encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)  # as labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3])  # reducing the channel dimension.
            if one_hot: input_batch = tf.one_hot(input_batch, depth=num_classes)
        return input_batch

    # 将图片升成4维， 然后变换尺寸
    @staticmethod
    def preprocess(image_tensor, h, w):
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image_tensor)
        image_tensor = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
        image_tensor -= IMG_MEAN
        input_batch = tf.expand_dims(image_tensor, 0)
        return tf.image.resize_nearest_neighbor(input_batch, [h, w])

    @staticmethod
    def inv_preprocess(imgs, num_images, img_mean):
        """Inverse preprocessing of the batch of images.
           Add the mean vector and convert from BGR to RGB.

        Args:
          imgs: batch of input images.
          num_images: number of images to apply the inverse transformations on.
          img_mean: vector of mean colour values.

        Returns:
          The batch of the size num_images with the same spatial dimensions as the input.
        """
        n, h, w, c = imgs.shape
        assert (n >= num_images),'Batch size {} should < number of images to save {}.'.format(n, num_images)
        outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
        for i in range(num_images):
            outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
        return outputs

    @staticmethod
    def load_img(img_path):
        if os.path.isfile(img_path):
            print('successful load img: {0}'.format(img_path))
        else:
            print('not found file: {0}'.format(img_path))
            sys.exit(0)

        filename, ext = img_path.split('/')[-1].split('.')

        img = None
        if ext.lower() == 'png':
            img = tf.image.decode_png(tf.read_file(img_path), channels=3)
        elif ext.lower() == 'jpg':
            img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
        else:
            print('cannot process {0} file.'.format(ext))

        return img, filename

    pass

if __name__ == '__main__':

    Tools.print_ckpt("checkpoints/model.ckpt-0")