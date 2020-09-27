import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
import os
import random
from random import shuffle
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
import seaborn as sns
from tensorflow.keras.models import Model
from PIL import Image
from scipy.ndimage.interpolation import rotate
import pickle
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import albumentations as A
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,roc_auc_score,confusion_matrix
random.seed(0)
tf.compat.v1.disable_eager_execution()
img_dirs = []
labels = []

root_dir = '/home/b170007ec/Programs/Manoj/CLASSIFIER/ROIS'

for file in os.listdir(root_dir):
    if file.split('_')[-2]!='.ipynb':
        file_dir = os.path.join(root_dir,file)
        img_dirs.append(file_dir)
        
        file_list = file.split('_')
        try :
            labels.append(int(file_list[-2])-1)
        except ValueError:
            labels.append(int(file_list[-3])-1)

root_dir = '/home/b170007ec/Datasets/Cervix Cancer/train'
list_ = ['additional_Type_1_v2','additional_Type_2_v2','additional_Type_3_v2','train/Type_1','train/Type_2','train/Type_3']
'''
#print(os.listdir(root_dir))
for i,folder in enumerate(os.listdir(root_dir)):
    present_dir = os.path.join(root_dir,folder)
    if folder.split('_')[-1] == '1' or folder.split('_')[-1]=='2' or folder.split('_')[-1]=='3':
        label = int(folder.split('_')[-1])
        
    elif folder.split('_')[-2] == '1' or folder.split('_')[-2]=='2' or folder.split('_')[-2]=='3':
        label = int(folder.split('_')[-2])
    else:
        print('shit! something went wrong!')
    for file in os.listdir(present_dir):
        if file.split('.')[-1] != 'ipynb_checkpoints':
            img_dirs.append(os.path.join(present_dir,file)  )
            labels.append(label-1)
print(len(img_dirs))      

temp = list(zip(img_dirs,labels)) 
shuffle(temp) 
img_dirs, labels = zip(*temp)
'''
test_dirs = img_dirs[:200]
test_labels = labels[:200]
img_dirs = img_dirs[200:]
labels = labels[200:]
data = (img_dirs,labels)


INPUT_SIZE = 512
BATCH_SIZE = 6
VALID_SPLIT = 0.1
EPOCHS = 50
ALPHA = 2.0
GAMMA = 3.0
LR = 0.0001

def metric(y_true,y_pred):
    log = -y_true*K.log(y_pred)
    log = K.sum(log,axis=-1)
    log = K.mean(log,axis=0)
    return log

def aug(max_angle=90):
    a = A.Compose([A.Rotate(limit = max_angle)])
    return a
#generator
class Generator(Sequence):
    def __init__(self,data,batch_size=BATCH_SIZE,input_size=INPUT_SIZE,is_train = True):
        self.img_dirs = data[0]
        self.labels = data[1]
        self.batch_size = batch_size
        self.input_size = input_size
        self.is_train = is_train
        if self.is_train:
            self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.img_dirs)/float(self.batch_size)))
    def on_epoch_end(self):
        if self.is_train:
            temp = list(zip(self.img_dirs,self.labels)) 
            shuffle(temp) 
            self.img_dirs, self.labels = zip(*temp)
    def __getitem__(self,idx):
        train_x = self.img_dirs[self.batch_size*idx:self.batch_size*(idx+1)]
        train_y =  self.labels[self.batch_size*idx:self.batch_size*(idx+1)]
        return self.generate(train_x,train_y)
    def generate(self,train_x,train_y):
        X = []
        Y = []
        for i,img in enumerate(train_x):
            y = np.zeros(3)
            y[train_y[i]] = 1
            if img.split('.')[-1]=='npy':
                roi = np.load(img)
            else:
                roi = cv2.imread(img)
                   
            #image = cv2.resize(image,(self.input_size,self.input_size))
            roi = cv2.resize(roi,(self.input_size,self.input_size))
            #img = np.concatenate([roi,image],axis=-1)
            X.append(roi/255)
            if self.is_train:
                data = {'image':roi}
                roi = aug()(**data)
                roi = roi['image']
                X.append(roi/255)
                Y.append(y)
            Y.append(y)
        return np.asarray(X), np.asarray(Y)
class prediction(Sequence):
    def __init__(self,data,batch_size=BATCH_SIZE,input_size=INPUT_SIZE):
        self.img_dirs = data
        self.batch_size = batch_size
        self.input_size = input_size
    def __len__(self):
        return int(np.ceil(len(self.img_dirs)/float(self.batch_size)))
    def on_epoch_end(self):
            pass
    def __getitem__(self,idx):
        train_x = self.img_dirs[self.batch_size*idx:self.batch_size*(idx+1)]
        return self.generate(train_x)
    def generate(self,train_x):
        X = []
        for i,img in enumerate(train_x):
            if img.split('.')[-1]=='npy':
                roi = np.load(img)
            else:
                roi = cv2.imread(img)   
            #image = cv2.resize(image,(self.input_size,self.input_size))
            roi = cv2.resize(roi,(self.input_size,self.input_size))
            #img = np.concatenate([roi,image],axis=-1)
            X.append(roi/255)
        return np.asarray(X)
def focal_loss(alpha,gamma):
    def loss_fn(y_true,y_pred):
        y_pred = K.clip(y_pred,1e-5,1-1e-5)
        loss = alpha*((1-y_pred)**gamma)*y_true*K.log(y_pred)
        loss = -K.sum(loss,axis=-1)
        return loss
    return loss_fn

#supporting blocks
def up_image(input_1,c=None):
    x = UpSampling2D((2,2))(input_1)
    x_ = Conv2D(c//4,(1,1),kernel_initializer='glorot_uniform',activation='relu')(x)
    return x

def build_model(ALPHA,GAMMA):
    encoder = efn.EfficientNetB4(include_top=False,input_shape = (INPUT_SIZE,INPUT_SIZE,3))    
    #x_ = up_image(encoder.output,c = 256)
    #x_ = up_image(x_,c = 128)
    #x_ = up_image(x_,c = 64)
    #x_ = Conv2D(32,(2,2),kernel_initializer = 'glorot_uniform',padding='same',activation='relu')(x_)
    #x_ = Conv2D(3,(1,1),kernel_initializer='glorot_uniform',padding='same',activation='sigmoid')(x_)
    #arbi = Model(encoder.input,x_)
    #arbi.load_weights('/home/b170007ec/Programs/Manoj/DAE/model2_dae.h5')
    x = encoder.output
    x = Dropout(0.5)(x)
    x = Conv2D(512,(1,1),padding = 'same',kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(256,(3,3),padding = 'same',kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(128,(1,1),padding = 'same',kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(3,activation='softmax')(x)
    model = Model(encoder.input,x)
    model.compile(optimizer=Adam(lr=LR),loss=focal_loss(ALPHA,GAMMA),metrics = ['acc',metric])
    return model


SEED = 10
skf = StratifiedKFold(n_splits=5,shuffle=True ,random_state=SEED) #originally 5 splits
img_dirs = np.asarray(img_dirs)
labels = np.asarray(labels)
pred_gen = prediction(test_dirs)
def test(model,weights):
    n = len(weights)
    predictions = np.zeros((len(test_dirs),3))
    for i,weight in enumerate(weights):
        model.load_weights(weight)
        print("weights loaded...")
        predictions = predictions+model.predict_generator(pred_gen,verbose=1)
    predictions = predictions/n
    pred_labels = []
    for i in range(predictions.shape[0]):
        pred_labels.append(np.argmax(predictions[i,:]))
    f = open("results.txt",'w+')
    f.write("accuracy:{}\n".format(accuracy_score(test_labels,pred_labels)))
    f.write("precision_score:{}\n".format(precision_score(test_labels,pred_labels,average=None)))
    f.write("recal_score:{}\n".format(recall_score(test_labels,pred_labels,average=None)))
    f.write("f1_score:{}\n".format(f1_score(test_labels,pred_labels,average=None)))
    #f.write("roc_auc_score:{}\n".format(roc_auc_score(test_labels,pred_labels,average=None)))
    f.close()
    matrix = confusion_matrix(test_labels,pred_labels)
    ax = plt.subplot()
    sns.heatmap(matrix,annot=True,ax = ax)
    ax.set_xlabel("prediction labels")
    ax.set_ylabel("true labels")
    plt.savefig("rconfusion matrix.png")
weights = []
for fold,(idxT,idxV) in enumerate(skf.split(img_dirs,labels)):
    print('#'*25)
    print('######FOLD'+str(fold))
    print('#'*25)
    train_dirs = img_dirs[idxT]
    train_labels = labels[idxT]
    val_dirs = img_dirs[idxV]
    val_labels = labels[idxV]
    train_gen = Generator((train_dirs,train_labels),is_train=True)
    val_gen = Generator((val_dirs,val_labels),is_train=False)
    STEPS_PER_EPOCH = len(train_dirs)//BATCH_SIZE
    model = build_model(ALPHA,GAMMA)
    name = '/home/b170007ec/Programs/Manoj/CLASSIFIER/'+'fold_'+str(fold)+'-{val_acc:03f}'+'.h5'
    early_stopping = EarlyStopping(monitor = 'val_acc', min_delta=0, patience = 6, verbose = 1)
    checkpoint = ModelCheckpoint(name,monitor = 'val_acc', save_best_only = True, verbose = 1, period = 1,mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1, epsilon=LR, mode='max')  
    history = model.fit_generator(train_gen,steps_per_epoch=STEPS_PER_EPOCH,validation_data = val_gen,epochs = EPOCHS,callbacks = [checkpoint,reduce_lr_loss,early_stopping])
    weights.append(name)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("accuracy_fold{}.jpg".format(fold))
    plt.clf()
# summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("loss_fold{}.jpg".format(fold))
    plt.clf()
model = build_model(ALPHA,GAMMA)
test(model,weights)
    