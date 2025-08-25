import caffe
import os
import numpy as np
import google.protobuf as pb
import google.protobuf.text_format


# project root
ROOT = '/home/passwd123/zhu/ENet/final_model_weights'

# choose your source model and destination model
WEIGHT = os.path.join(ROOT, 'bn_conv_merged_weights.caffemodel')
MODEL = os.path.join(ROOT, 'bn_conv_merged_model.prototxt')
DEPLOY_MODEL = os.path.join(ROOT, 'bn_conv_merged_model_deploy.prototxt')

# set network using caffe api
caffe.set_mode_gpu()
net = caffe.Net(MODEL, WEIGHT, caffe.TRAIN)
dst_net = caffe.Net(DEPLOY_MODEL, caffe.TEST)
with open(MODEL) as f:
    model = caffe.proto.caffe_pb2.NetParameter()
    pb.text_format.Parse(f.read(), model)

# go through source model 
for i, layer in enumerate(model.layer):
    if layer.type == 'Convolution':
        # extract weight and bias in Convolution layer
        name = layer.name
        print name
        if 'fc' in name:
            dst_net.params[name][0].data[...] = net.params[name][0].data
            dst_net.params[name][1].data[...] = net.params[name][1].data
            break
        w = net.params[name][0].data
        batch_size = w.shape[0]+
        try:
            b = net.params[name][1].data
        except:
            b = np.zeros(batch_size)

        # extract mean and var in BN layer
        bn = name+'/bn'
        print bn
        mean = net.params[bn][0].data
        var = net.params[bn][1].data
        scalef = net.params[bn][2].data
        if scalef != 0:
            scalef = 1. / scalef
        mean = mean * scalef
        var = var * scalef

        # extract gamma and beta in Scale layer
        scale = name+'/scale'
        print scale
        gamma = net.params[scale][0].data
        beta = net.params[scale][1].data

        # merge bn
        tmp = gamma/np.sqrt(var+1e-5)
        w = np.reshape(tmp, (batch_size, 1, 1, 1))*w
        b = tmp*(b-mean)+beta

        # store weight and bias in destination net
        dst_net.params[name][0].data[...] = w
        dst_net.params[name][1].data[...] = b

dst_net.save('bn_conv_merged_model_deploy.caffemodel')

