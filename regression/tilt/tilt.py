import keras
from keras import optimizers
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import shutil
import random


size=40

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

base_model = VGG16(weights="imagenet", include_top=False,
                                input_shape=(size, size, 3),backend=keras.backend,
                                          layers=keras.layers,
                                          models=keras.models,
                                          utils=keras.utils) 
# print(base_model.summary())
base_model.trainable=False
# for layer in base_model.layers[:15]: layer.trainable = False  

top_model = Sequential() 
top_model.add(Flatten(input_shape=base_model.output_shape[1:])) 

top_model.add(Dense(512, activation='relu')) 
top_model.add(Dropout(0.5)) 
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))  
top_model.add(Dense(2))  

print(top_model.summary())


model = Model(inputs=base_model.input, outputs=top_model(base_model.output)) 

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['acc'])


label = np.loadtxt(r'train.txt')
xpath=r'.\data\train\3_tilt'
filelist = os.listdir(xpath) 
images=[]
labels=label

val_label = np.loadtxt(r'val.txt')
val_xpath=r'.\data\val\3_tilt'
val_filelist = os.listdir(val_xpath) 
val_images=[]
val_labels=val_label

for file in filelist:
    filename=os.path.splitext(file)[0]
    x=filename.split('_')
    if(int(x[1])<1000):
        path=os.path.join(xpath,file)
        img=Image.open(path)
        img = img.resize((size, size),Image.ANTIALIAS)
        img = img.convert('RGB')
        img=np.array(img).astype(float)
        images.append(img)

for val_file in val_filelist:
    val_filename=os.path.splitext(val_file)[0]
    x=val_filename.split('_')
    if(int(x[1])<1000):
        val_path=os.path.join(val_xpath,val_file)
        val_img=Image.open(val_path)
        val_img = val_img.resize((size, size),Image.ANTIALIAS)
        val_img = val_img.convert('RGB')
        val_img=np.array(val_img).astype(float)
        val_images.append(val_img)

print(len(images))
print(len(labels))
def train_gen():
    i=0
    while True:
        n=0
        batch_images=[]
        batch_labels=[]
        for n in range(256):
            i=random.randint(0,3913)
            batch_images.append(images[i])
            batch_labels.append(labels[i])
        batch_images=np.array(batch_images)
        batch_labels=np.array(batch_labels)
        yield batch_images,batch_labels
        

def val_gen():
    j=0
    while True:
        n=0
        val_batch_images=[]
        val_batch_labels=[]
        for n in range(128):
            j=random.randint(0,1289)
            val_batch_images.append(val_images[j])
            val_batch_labels.append(val_labels[j])
        val_batch_images=np.array(val_batch_images)
        val_batch_labels=np.array(val_batch_labels)
        yield val_batch_images,val_batch_labels



checkpointer = ModelCheckpoint(filepath='tilt.h5', verbose=1, save_best_only=True)
history = model.fit_generator(
#    train_generator,
    generator=train_gen(),
    steps_per_epoch=16,
    epochs=500,
    callbacks=[checkpointer],
    validation_data=val_gen(),
    validation_steps=11
    )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

file=open('acc_1.txt','a')
file.write(str(acc));
file.close()
file=open('acc_val_1.txt','a')
file.write(str(val_acc));
file.close()
file=open('loss_1.txt','a')
file.write(str(loss));
file.close()
file=open('loss_val_1.txt','a')
file.write(str(val_loss));
file.close()

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

