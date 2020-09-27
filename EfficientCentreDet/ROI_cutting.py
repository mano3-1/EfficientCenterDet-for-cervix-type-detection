import cv2, os
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,load_model
import pandas as pd
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
from PIL import Image
from efficientnet import tfkeras as efn

stride = 2
input_size = 512


def build_fpn(features,num_channels,wbifpn,kernel_size=2):
    p4,p5,p6,p7 = features
    #column1
    p6 = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p6)
    p5 = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p5)
    p4 = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p4)
    
    p7 = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p7)
    p7_resize = BatchNormalization()(p7)
    p7_resize = MaxPool2D((kernel_size,kernel_size))(p7_resize)
    if wbifpn:
        p6_td = Fuse()([p6,p7_resize])
    else:
        p6_td = Add()([p6,p7_resize])
    p6_td = Conv2D(num_channels,(3,3),kernel_initializer = 'glorot_uniform',activation='relu',padding='same')(p6_td)
    p6_td = BatchNormalization()(p6_td)
    p6_td = MaxPool2D((2,2),padding = 'same',strides = 1)(p6_td)
    p6_td_resize = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p6_td)
    p6_td_resize = BatchNormalization()(p6_td_resize)
    
    p6_td_resize = MaxPool2D((kernel_size,kernel_size))(p6_td_resize) 
    if wbifpn:
        p5_td = Fuse()([p5,p6_td_resize])
    else:
        p5_td = Add()([p5,p6_td_resize])
    p5_td = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p5_td)
    p5_td = BatchNormalization()(p5_td)
    p5_td = MaxPool2D((2,2),padding='same',strides = 1)(p5_td)
    p5_td_resize = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p5_td)
    p5_td_resize = BatchNormalization()(p5_td_resize)
    p5_td_resize = MaxPooling2D((kernel_size,kernel_size))(p5_td_resize)
    if wbifpn:
        p4_td = Fuse()([p4,p5_td_resize])
    else:
        p4_td = Add()([p4,p5_td_resize])
    p4_td = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p4_td)
    p4_td = MaxPool2D((2,2),padding='same',strides = 1)(p4_td)
    p4_U = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p4_td)
    p4_U = BatchNormalization()(p4_U)
    p5_U = UpSampling2D((kernel_size,kernel_size))(p4_U)
    if wbifpn:
        p5_U = Fuse()([p5,p5_td,p5_U])
    else:
        p5_U = Add()([p5,p5_td,p5_U])
    p5_U = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p5_U)
    p5_U = BatchNormalization()(p5_U)
    p6_U = UpSampling2D((kernel_size,kernel_size))(p5_U)
    if wbifpn:
        p6_U = Fuse()([p6,p6_td,p6_U])
    else:
        p6_U = Add()([p6,p6_td,p6_U])
    p6_U = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p6_U)
    p6_U = BatchNormalization()(p6_U)
    p7_U = UpSampling2D((kernel_size,kernel_size))(p6_U)
    if wbifpn:
        p7_U = Fuse()([p7,p7_U])
    else:
        p7_U = Add()([p7,p7_U])
    p7_U = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p7_U)
    p7_U = BatchNormalization()(p7_U)
    return (p4_U,p5_U,p6_U,p7_U)

class Fuse(tf.keras.layers.Layer):
    '''Fusion layer'''
    def __init__(self, epsilon=1e-4, **kwargs):
        super(Fuse, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=tf.keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(Fuse, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config
    
def create_model(input_shape ,wbifpn=False):
    '''model'''
    effnet = efn.EfficientNetB4(input_shape=input_shape,weights=None,include_top = False)
    p4 = effnet.get_layer('block2a_activation').output
    p5 = effnet.get_layer('block3a_activation').output
    p6 = effnet.get_layer('block4a_activation').output
    p7 = effnet.get_layer('block7a_activation').output
    features = (p7,p6,p5,p4)
    features = build_fpn(features,16,wbifpn)
    features = build_fpn(features,32,wbifpn)
    features = build_fpn(features,64,wbifpn)
    features = build_fpn(features,81,wbifpn)
    features = list(features)
    for i in range(1,4):
        feature_curr = features[i]
        feature_past = features[i-1]
        feature_past_up = UpSampling2D((2,2))(feature_past)
        feature_past_up = Conv2D(81,(3,3),padding='same',activation='relu',kernel_initializer='glorot_uniform')(feature_past_up)
        if wbifpn:
            feature_final = Fuse(name='final{}'.format(str(i)))([feature_curr,feature_past_up])
        else:
            feature_final = Add(name='final{}'.format(str(i)))([feature_curr,feature_past_up])
        features[i] = feature_final
    if stride == 2:
        features[-1] = UpSampling2D((2,2))(features[-1])
        features[-1] = Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='glorot_uniform')(features[-1])
    out = Conv2D(5,(3,3),activation='sigmoid',kernel_initializer='glorot_uniform',padding='same')(features[-1])
    zeros = tf.expand_dims(tf.zeros_like(out[...,0]),axis=-1)
    out_concat = tf.concat([zeros,out],axis = -1)
    prediction_model=tf.keras.models.Model(inputs=[effnet.input],outputs=out)
    model = Model(inputs = [effnet.input],outputs = out_concat)

    return model,prediction_model

def _ctdet_decode(hm, reg, wh, k=20, output_stride=2):
    bboxes = []
    scores = []
    hm = K.eval(_nms(tf.cast(hm,tf.float32)))
    hm = hm[0,:,:,0]
    w = hm.shape[1]
    hm_ = np.zeros_like(hm)
    hm_flat = hm.reshape(-1)
    for i in range(k):
        m = np.argmax(hm_flat)
        hm_[m//w][m%w] = hm_flat[m]
        hm_flat[m] = -1
    points = np.argwhere(hm_[:,:]!=0)
    
    for (y,x) in points:
        
        score = hm_[y,x]
        offy = reg[0,y,x,0]
        offx = reg[0,y,x,1]
        height = wh[0,y,x,0]*output_size
        width = wh[0,y,x,1]*output_size
        xc = x+offx
        yc = y+offy
        xmin = int((xc-(width/2)))
        ymin = int((yc-(height/2)))
        xmax = int((xc+(width/2)))
        ymax = int((yc+(height/2)))
        bboxes.append([xmin,ymin,xmax,ymax])
        scores.append(score)
    return scores,bboxes
def _nms(heat, kernel=5):
    hmax = K.pool2d(heat, (kernel, kernel), padding='same', pool_mode='max')
    keep = K.cast(K.equal(hmax, heat), K.floatx())
    return heat * keep

tsv1 = '/home/b170007ec/Type_1_bbox.tsv'
tsv2 = '/home/b170007ec/Type_2_bboxes.tsv'
tsv3 = '/home/b170007ec/Type_3_bbox.tsv'
tsv1 = pd.read_csv(tsv1,sep='/t',header=None,names=['bboxes'])
tsv2 = pd.read_csv(tsv2,sep='/t',header=None,names=['bboxes'])
tsv3 = pd.read_csv(tsv3,sep='/t',header=None,names=['bboxes'])

bboxes = []
img_dirs = []
for i,tsv in enumerate([tsv1,tsv2,tsv3]):
    for k in range(tsv.shape[0]):
        string = tsv.iloc[k].to_list()[0][7:]
        string = string.split(' ')
        img_dir = '/home/b170007ec/Datasets/Cervix Cancer/train/train/train/Type_'+str(i+1)+'/'+string[0]
        bbox = string[2:6]
        for l in range(4):
            bbox[l] = int(bbox[l])
        img_dirs.append(img_dir)
        bboxes.append(bbox)

_,model = create_model((input_size,input_size,3),False)
output_size = input_size//stride
model.load_weights('/home/b170007ec/Programs/Manoj/DETECTOR/best_model.h5')
destination = '/home/b170007ec/Datasets/Cervix Cancer/ROI'
for count,test_image in enumerate(img_dirs):
    img_name = test_image.split('/')[-1]
    Type = test_image.split('/')[-2]
    imagecv = cv2.imread(test_image)
    h,w = imagecv.shape[0],imagecv.shape[1]
    image = cv2.resize(imagecv,(input_size,input_size))
    
    output = model.predict(image.reshape(-1,input_size,input_size,3)/255)
    hm = output[:,:,:,0].reshape((1,output_size,output_size,1))
    reg = output[:,:,:,1:3]
    wh = output[:,:,:,3:]
    
    
    scores,detections = _ctdet_decode(hm, reg, wh, k=1, output_stride=2)
    for i in range(len(detections)):

        detection = detections[i]
        xmin = detection[0]*(w/output_size)
        
        xmin = xmin - (xmin//20)
        xmin = int(int(xmin>0)*xmin)
        if xmin>w:
            xmin = w
        ymin = detection[1]*(h/output_size)
        ymin = ymin - (ymin//20)
        ymin = int(ymin*int(ymin>0))
        if ymin>h:
            ymin = h
        xmax = detection[2]*(w/output_size)
        xmax = xmax+(xmax//20)
        xmax = int(xmax*int(xmax>0))
        if xmax>w:
            xmax = w
        ymax = detection[3]*(h/output_size)
        ymax = ymax + (ymax//20)
        ymax = int(ymax*int(ymax>0))
        
        if ymax>h:
            ymax = h
    img = Image.fromarray(imagecv)
    img = img.crop((xmin,ymin,xmax,ymax))
    img = img.save(os.path.join(destination,Type,img_name))
    print('{} of {} is saved in {}'.format(img_name,Type,os.path.join(destination,Type,img_name)))