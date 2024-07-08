from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping
import keras
from time import time
from matplotlib import pyplot as plt

import os
import shutil
import random

#DATA SOURCE --------------------------------------------------

path_to_data = '/content/drive/My Drive/FSI/Imagenes'

batch_generator_size = 27
batch_validation_size = 5

train_data_dir =  path_to_data + '/entrenamiento'

folders = os.listdir(train_data_dir)
images = []
for i in range(len(folders)):
	this_class_images = os.listdir(train_data_dir + '/' + folders[i])
	chosen_images = []
	for i in range(batch_validation_size):
		chosen = random.randint(0,len(this_class_images)-1)
		chosen_images.append(this_class_images[chosen])
		this_class_images.pop(chosen)
	images.append(chosen_images)

validation_data_dir = path_to_data + '/validacion'

for i in range(len(images)):
	for j in range(batch_validation_size):
		shutil.move(str(train_data_dir)+'/'+str(folders[i])+'/'+str(images[i][j]),
		            str(validation_data_dir)+'/'+str(folders[i])+'/'+str(images[i][j]))

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
				horizontal_flip=True)

validation_datagen = ImageDataGenerator(
			rescale=1./255)

train_generator = train_datagen.flow_from_directory(
			train_data_dir,
			target_size=(150, 150),
			batch_size=batch_generator_size,
			class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
			validation_data_dir,
			target_size=(150, 150),
			batch_size=batch_validation_size,
			class_mode='categorical')

#MODEL --------------------------------------------------

model = Sequential()

model.add(
			Conv2D(32, kernel_size=(3, 3),
			activation='relu',
			input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(
				loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.Adadelta(),
				metrics=['accuracy'])

#TRAINING --------------------------------------------------

epochs = 100

es = EarlyStopping(
				monitor='val_accuracy',
				mode='max', verbose=1,
				patience=20,
				restore_best_weights=True)

history = model.fit_generator(
				train_generator,
				epochs=epochs,
				validation_data = validation_generator,
				callbacks = [es] )

#PLOT ----------------------------------------------------------

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Entrenamiento')
plt.xlabel('Épocas')
plt.legend(loc="lower right")
plt.show()

#SAVING ----------------------------------------------------------

for i in range(len(images)):
    for j in range(batch_validation_size):
        shutil.move(str(validation_data_dir)+'/'+str(folders[i])+'/'+str(images[i][j]),
                    str(train_data_dir)+'/'+str(folders[i])+'/'+str(images[i][j]))

model.save("Pa2.h5")