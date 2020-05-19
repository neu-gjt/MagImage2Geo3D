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

size=40
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

train_dir = r'.\data\train'
validation_dir = r'.\ex4\data\val'

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
top_model.add(Dense(5, activation='softmax')) 


# top_model.load_weights("")  

model = Model(inputs=base_model.input, outputs=top_model(base_model.output)) 

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

# Adjust the format of training data
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    featurewise_center=True,  
    featurewise_std_normalization=True,
    samplewise_center=False,
    samplewise_std_normalization=False,
    zca_whitening=True

    )
test_datagen = ImageDataGenerator(
    # rescale=1./255,
    featurewise_center=True, 
    featurewise_std_normalization=True,
    samplewise_center=False,
    samplewise_std_normalization=False,
    zca_whitening=True)

#the generator of training data
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(size, size),
    batch_size=256,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(size, size),
    batch_size=128,
    class_mode='categorical')
print(train_generator.class_indices)

#useing checkpoint to save model
checkpointer = ModelCheckpoint(filepath='inceptionv3_1.h5', verbose=1, save_best_only=True)
history = model.fit_generator(
#    train_generator,
    generator=train_generator,
    steps_per_epoch=122,
    epochs=1000,
    callbacks=[checkpointer],
    validation_data=validation_generator,
    validation_steps=80
    )


#plot curve of accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
file=open('acc_inceptionv3_1.txt','a')
file.write(str(acc));
file.close()
file=open('acc_val_inceptionv3_1.txt','a')
file.write(str(val_acc));
file.close()
file=open('loss_inceptionv3_1.txt','a')
file.write(str(loss));
file.close()
file=open('loss_val_inceptionv3_1.txt','a')
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


