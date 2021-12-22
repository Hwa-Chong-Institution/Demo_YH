# from os import close
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# from keras.models import load_model
# import pathlib
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import time as t

# print(tf.__path__)

# ds = tfds.load()

# datagen = ImageDataGenerator()
# dataset = datagen.flow_from_directory(
#     'Alphabet/'
# )

# print(f"abcbabcbabcbabcbabc{dataset}")

# t.sleep(100)

# (x_train, y_train), (x_test, y_test) = dataset

# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# x_test = tf.keras.utils.normalize(x_train, axis = 1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #output layer

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# # val_loss, val_acc = model.evaluate(x_test, y_test)
# # print(val_loss, val_acc)

# model.save('number_reader.model')
# new_model = tf.keras.models.load_model('number_reader.model')

# predictions = new_model.predict([x_test])
# print(np.argmax(predictions[0]))


# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()

#--------------------------------------------------------------------------------------------

# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# from keras.models import load_model
# import pathlib

# img_height = 28
# img_width = 28
# batch_size = 2

# model = tf.keras.Sequential()

# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #output layer

# ds_train = tf.keras.processing.image_dataset_from_directory(
#     'Alphabet/',
#     labels = 'inferred',
#     label_mode = "int",
#     colour_mode = 'grayscale',
#     batch_size = batch_size,
#     image_size = (img_height, img_width),
#     shuffle=True,
#     seed=123,
#     validation_split=0.1,
#     subset="training",
# )

# ds_validation = tf.keras.processing.image_dataset_from_directory(
#     'Alphabet/',
#     labels = 'inferred',
#     label_mode = "int",
#     colour_mode = 'grayscale',
#     batch_size = batch_size,
#     image_size = (img_height, img_width),
#     shuffle=True,
#     seed=123,
#     validation_split=0.1,
#     subset="validation",
# )

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.python.ops.gen_array_ops import lower_bound_eager_fallback

DATADIR = "Alphabet"
CATEGORIES = ["A", "B", "C"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        print(img_array)
        break
    break

IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))

random.shuffle(training_data)

for sample in training_data:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()
model.add(Conv2D(64), (3,3), input_shape = X.shape[1:])
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64), (3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1)) #output
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)