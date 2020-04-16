import csv 
import matplotlib.image as mping
import numpy as np
import sklearn 

# Open the csv file
lines = []
with open('../../../root/Desktop/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Split data into training set and validation set
from sklearn.model_selection import train_test_split
train_data, valid_data = train_test_split(lines, test_size=0.2)

from sklearn.utils import shuffle
def generator(lines, batch_size=32):
    num_line = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0, num_line, batch_size):
            batch_data = lines[offset:offset+batch_size]
            images = []
            angles = []
            # Extract steering angle data from csv file and read images from 3 cameras
            for line in batch_data:
                for camera in range(3):
                    source_path = line[camera]
                    file_name = source_path.split('/')[-1]
                    current_path = '../../../root/Desktop/data/IMG/' + file_name
                    image = mping.imread(current_path)
                    images.append(image)
                    if camera == 0:
                        angle = float(line[3])
                    if camera == 1:
                        angle = float(line[3]) + 0.15
                    else:
                        angle = float(line[3]) - 0.15
                    angles.append(angle)
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    angle_flipped = -angle
                    angles.append(angle_flipped)
                X_train = np.array(images)
                Y_train = np.array(angles)
                yield shuffle(X_train, Y_train)

train_generator = generator(train_data, batch_size=32)
valid_generator = generator(train_data, batch_size=32)

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Lambda, Flatten, Dense ,Cropping2D, Dropout, Conv2D
from keras.models import Model, Sequential
import math

# Use pre-trained VGG16 model as a start point
#input_height = 65
#input_width = 320
#vgg = VGG16(include_top=False, weights='imagenet', input_shape=(input_height, input_width, 3))

# freeze all layers weight
#for layer in vgg.layers:
    #layer.trainable = False

#input_shape = Input(shape=(160, 320, 3))
#normalize = Lambda(lambda x: x/255.0 - 0.5)(input_shape)
#crop_input = Cropping2D(cropping=((70, 25), (0,0)))(normalize)
#vgg16 = vgg(crop_input)
#flatten = Flatten()(vgg16)
#fc1 = Dense(2048, activation='relu')(flatten)
#d1 = Dropout(0.5)(fc1)
#fc2 = Dense(2048, activation='relu')(d1)
#d2 = Dropout(0.5)(fc2)
#prediction = Dense(1)(d2)

#model = Model(input=input_shape, outputs=prediction)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))   # Normalize inputs
model.add(Cropping2D(cropping=((35,25), (0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(80, (3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.8))   # dropout regularization
model.add(Dense(50))
model.add(Dense(1))


model.compile(optimizer='Adam', loss='mse')
model.fit_generator(train_generator,                          steps_per_epoch=math.ceil(len(train_data)/32),
  epochs=4, verbose=1, validation_data=valid_generator,
  validation_steps = math.ceil(len(valid_data)/32))

model.save('model.h5')


