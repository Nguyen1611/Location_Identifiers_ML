
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import Sequential
from keras.src.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.src.optimizers import RMSprop
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing

from tensorflow.keras.callbacks import EarlyStopping

import os

# list folder/directories in the path
os.listdir('qmind-school-of-ai-competition-2023')

# list directories in train path
os.listdir('qmind-school-of-ai-competition-2023/train')

# Augment train set only
train_data_generator = ImageDataGenerator(
    rescale=1 / 255,
    validation_split=0.15,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
batch_size = 128
train_batch = train_data_generator.flow_from_directory('qmind-school-of-ai-competition-2023/train',
                                                       target_size=(256, 256),
                                                       class_mode='categorical',
                                                       batch_size= batch_size
                                                       )


# Loading validation data
validation_data_generator = ImageDataGenerator(
    rescale=1 / 255,
    validation_split=0.15  # You can adjust the validation split as needed
)

validation_gen = validation_data_generator.flow_from_directory(
    'qmind-school-of-ai-competition-2023/test_all',
    target_size=(256, 256),
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'  # Specify that this is the validation subset
)

# the test files have to be retrieved in another way since they are not categorized into folders

# this is helper function to get all the paths to each file in the folder
def get_files(directory):
    files = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    return files


# this code gets all the files as image arrays
test = []
for file in get_files('qmind-school-of-ai-competition-2023/test_all'):
    img = cv2.imread(file)
    # sometimes cv2 does not read jpg files correctly, so when that happens, use pyplot and change to BGR
    # one of the images (89.jpg) has 4 colour channels, so we need to convert that to RGB
    if (img is None):
        # print(file)
        img = plt.imread(file)
        img = img[..., ::-1]
        if len(img.shape) > 2 and img.shape[2] == 4:
            # convert the image from RGBA2RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img_array = np.asarray(resized)
    test.append(img_array)

test = np.array(test, dtype=object)
test = test.astype('float32') / 255.

# CNN model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(256, 256, 3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

#set up early stop
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)

# train
model.fit(train_batch, validation_data=validation_gen, epochs=20, callbacks=[es])

# Predict on the test data
predictions = model.predict(test)

# interpret these predictions as class probabilities
predicted_class = np.argmax(predictions, axis=-1)  # will give same result as axis = 1

# get test ids
test_ids = []

for file in get_files('qmind-school-of-ai-competition-2023/test_all'):
    test_ids.append(os.path.splitext(os.path.basename(file))[0])

# encode prediction
classes = ['grocerystore', 'bar', 'casino', 'bedroom', 'kitchen', 'bookstore', 'deli', 'bakery', 'bowling',
           'winecellar']
le = preprocessing.LabelEncoder()
le.fit(classes)
classes_encoded = le.transform(classes)

# make a submission.csv

predicted_class = np.argmax(predictions, axis=-1)

submission = {"ID": np.array(test_ids).astype(int), "TARGET": predicted_class}
submisson_df = pd.DataFrame(submission)

# sort the data frame by ID
submisson_df = submisson_df.sort_values(by=['ID'])
submisson_df.to_csv('submission.csv', index=False)

train_metrics = model.evaluate(train_batch,verbose=1)
