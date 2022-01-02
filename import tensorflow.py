from PIL.Image import WEB
from keras.engine.sequential import Sequential
from keras.saving.save import load_model
from matplotlib import cm
import tensorflow as tf
import tensorflow.keras as tk
import numpy
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import random
    
#LOAD DATA

directory = "Alphabet"

categories = ["A", "B", "C"]

training_data = []

image_size = 32

#loops through the directories for the images and converts them to arrays
for category in categories:
    path = os.path.join(directory, category)
    image_class = categories.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # Copy this when inputting code for predictions
        new_array = cv2.resize(img_array, (image_size,image_size))
        training_data.append([new_array, image_class]) #array has two columns (image, category)
        ##plt.imshow(new_array, cmap='gray')
        ##plt.show()
##print(training_data)

random.shuffle(training_data)

X = []
y = []

for image, category in training_data: #Splits training_data into images(X) and categories(Y)
    X.append(image)
    y.append(category)

X = numpy.array(X).reshape(-1, image_size, image_size, 1)
y = numpy.array(y)

print(y)

#SAVE DATA (PICKLE)

pickle_out = open("X.pickle", "wb") #Creates a write binary pickle file for the X array called X.pickle
pickle.dump(X, pickle_out) #dumps X array's data into X.pickle
pickle_out.close()

pickle_out = open("y.pickle", "wb") #Creates a write binary pickle file for the y array called y.pickle
pickle.dump(y, pickle_out) #dumps y array's data into y.pickle
pickle_out.close()

#LOAD DATA (PICKLE)

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

#NORMALIZE DATA

X = X/255.0

#DEFINE MODEL

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(64,(3, 3), input_shape = X.shape[1:])) #??
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3))) #??
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#define loss and optimizer
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

#fit the model

model.fit(X, y, batch_size=2, epochs=3)

# evaluate the model

model.evaluate(X, y, batch_size=4)

# save model

model.save("predict_model.model")

# load model

new_model = tk.models.load_model("predict_model.model")

# load prediction_data

prediction_data = cv2.imread(("A_prediction.jpeg"), cv2.IMREAD_GRAYSCALE)
prediction_data = cv2.resize(prediction_data, (image_size, image_size))
prediction_data = numpy.array(prediction_data).reshape(-1, image_size, image_size, 1)

# make a prediction

predictions = new_model.predict([prediction_data])
print(numpy.argmax(predictions[0]))