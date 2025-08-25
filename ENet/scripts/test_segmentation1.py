#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import time
caffe_root = 'ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2


__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '24th May, 2017'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='/home/passwd123/zhu/ENet/final_model_weights/bn_conv_merged_model.prototxt', help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=False, default='/home/passwd123/zhu/ENet/final_model_weights/bn_conv_merged_weights.caffemodel', help='.caffemodel file')
    parser.add_argument('--colours', type=str, required=False, default='/home/passwd123/zhu/ENet/scripts/cityscapes19.png', help='label colours')
    parser.add_argument('--input_image_path', type=str, required=False, default='', help='input image path')
    parser.add_argument('--out_dir', type=str, required=False, default='', help='output directory in which the segmented images '
                                                                   'should be stored')
    parser.add_argument('--gpu', type=int, default=0, help='0: gpu mode active, else gpu mode inactive')

    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    
    input_a = os.listdir(args.input_image_path)
    
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['deconv6_0_0'].data.shape
    print(output_shape)
    totol = 0
    for input_index in input_a:
        input_image = args.input_image_path + input_index
       


        label_colours = cv2.imread(args.colours, 1).astype(np.uint8)
        input_image = cv2.imread(input_image, 1).astype(np.float32)

        input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.asarray([input_image])
        start = time.time()
        out = net.forward_all(**{net.inputs[0]: input_image})
        end = time.time()
        totol = totol + (end-start)
        prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)
        print("1",prediction.shape)
        prediction = np.squeeze(prediction)
        print("2",prediction.shape)
        prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
        print("3",prediction.shape)
        prediction = prediction.transpose(1, 2, 0).astype(np.uint8)
        print("4",prediction.shape)

        prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
        label_colours_bgr = label_colours[..., ::-1]
        cv2.LUT(prediction, label_colours_bgr, prediction_rgb)       
        print('Running time: {} Seconds'.format(end-start))

       #cv2.imshow("ENet", prediction_rgb)
       #key = cv2.waitKey(0)
        if args.out_dir is not None:
            #input_path_ext = input_image.split(".")[-1]
            #input_image_name = input_image.split("/")[-1:][0].replace('.' + input_path_ext, '')

            out_path_im = args.out_dir + input_index + '_enet' + '.png'
            #out_path_gt = args.out_dir + input_image_name + '_enet_gt' + '.' + input_path_ext

        cv2.imwrite(out_path_im, prediction_rgb)
        # cv2.imwrite(out_path_gt, prediction) #  label images, where each pixel has an ID that represents the class

    totol = totol / 220
    print('Running time ave============================>>>>>>>: {} Seconds'.format(totol))




