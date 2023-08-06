#importing the dependencies
# most important deep learning libraraies are tenflow, keras and pytouch

import numpy as np #for numpy arrays
import matplotlib.pyplot as plt #for the plots and graphs that we want
from matplotlib import pyplot
import seaborn as sns #for the plots and graphs that we want
import cv2 #open cv library for image processing tasks
from PIL import Image #also for image processing tasks
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix



#data collection part
#mnist data - image data from keras.datasets
#loading the mnist data from keras.dataset...(images datasets)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() #train is for traning data and test is for testing data
print(type(X_train)) #numpy n dimensional array

#shape of the numpy array
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
#output - (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
#meaning - X_train has 60,000 images and each image has a dimension of 28x28 and is a graysacle image
#Y_train stores the label of the images, similarly for x_test and y_test
#x_train: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data. Pixel values range from 0 to 255.
# y_train: uint8 NumPy array of digit labels (integers in range 0-9) with shape (60000,) for the training data.
# x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data. Pixel values range from 0 to 255.
# y_test: uint8 NumPy array of digit labels (integers in range 0-9) with shape (10000,) for the test data.

print(X_train[10].shape) #output- 28x28all the images have same dimensions in this dataaset, if not then we have to resize these images into one dimension
#displaying the image stored at index 10
plt.imshow(X_train[10])
plt.show()
#print the corresponding label
print(Y_train[10])
#image labels
print(Y_train.shape, Y_test.shape)
#unique values in y_train
print(np.unique(Y_train))
#unique values in y_test
print(np.unique(Y_test))

#scaling the images
X_train = X_train/255
X_test = X_test/255
print(X_train[50])

#building our neural network- normal neural network
#setting up the layers of the neural network
model = keras.Sequential([
    #layers of neural network

    keras.layers.Flatten(input_shape=(28,28)),#input layer #flatten function conerts all the dimensions of the image which were originally stored in the form a 2d matrix into a single line of dimensions
    keras.layers.Dense(50,activation='relu'),#hidden layer #no. of layers which we have is ...for example in this case 50
    keras.layers.Dense(50,activation='relu'),#hidden layer #another hidden layer
    keras.layers.Dense(10,activation='sigmoid')#output layer #10 means the number of classes, in this project we are identifying numbers, so total number of different digits are 10- (0,1,2,3,4,5,6,7,8,9) , so class will be 10 in this project
])

#compiling the neural network
model.compile(optimizer='adam', #to select an optimizer which finds the solution most accurately
              loss='sparse_categorical_crossentropy', #in this project we are working with numbers or we need to identify numbers, so we are using this loss value
              metrics=['accuracy'])

#training the neural network
model.fit(X_train,Y_train,epochs = 10) #epochs means how many times the neural network should go through the data, so we have put the value 10, so in this case the neural network will thorugh the data 10 times

#accuacy on the test data
#model evaluation
loss,accuracy = model.evaluate(X_test, Y_test) #model will take the value X_test and will give the prediction, now the predictions will be compared with the original value of label,i.e. with the Y_test , and will then give the accuracy of the model
print(accuracy) #output -0.9736999869346619 (not same in every run)

#making prediction for 1st data point
plt.imshow(X_test[0])
plt.show() #image will be printed that is stored at 0th index
print(Y_test[0]) #original value that is being displayed at the oth index, foe example image if 7 is being diplayed in the previous step, so the Y_test will output value 7
Y_pred = model.predict(X_test) #predict the label for all the images stored in X_test
print(Y_pred.shape) #output - (10000, 10) 10,000 since we have 10,000 total images in X_test and 10, since we have 10 distinct numbers

#converting the prediction to class label
label_for_first_test_image = np.argmax(Y_pred[0])
print(label_for_first_test_image) #output will be 7
#converting the prediction value to class label for all test data points
Y_pred_labels = [np.argmax(i) for i in Y_pred] #putting for loop
print(Y_pred_labels) #output is all the correct predictions for all the 10,000 stored images

#confusion matrix
conf_mat = confusion_matrix(Y_test, Y_pred_labels)
print(conf_mat)

#using seaborn and matplotlib
#we mention the width and height that we want for our plot
plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt = 'd', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

#using the model to predict values of the images we are inserting, the images are from mnist dataset , obviously because we using the the mnist algorithms
# input_image_path = r"C:\Users\anshi\Downloads\mnist image-8.jpg"
#
# input_image = cv2.imread(input_image_path)
# type(input_image) #numpy.ndarray
# print(input_image)
# print(input_image.shape) #the image is an rgb ,so we need to first convert it into grayscale and then resize it
# grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
# print(grayscale.shape)
# #resizing the image
# input_image_resize = cv2.resize(grayscale,(28,28))
# input_image_resize = input_image_resize/255
# image_reshape = np.reshape(input_image_resize, [1,28,28]) #[1,28,28] means that i am predicting for only 1 image , whose diension is 28,28
# input_prediction = model.predict(image_reshape)
# print(input_prediction)
# input_prediction_label = np.argmax(input_prediction)
# print(input_prediction_label)

#predicting system
input_image_path = input('Path of the image to be predicted: ') #image is taken by the usert
input_image = cv2.imread(input_image_path) #that image is read by the cv2 imread function
pyplot.imshow(input_image)
pyplot.show() #we will display the image to the user itself
grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY) #image is converted to grayscale image
input_image_resize = cv2.resize(grayscale,(28,28)) # then the image is resized to 28x28 px
input_image_resize = input_image_resize/255 #scaling the value between 0-1
image_reshape = np.reshape(input_image_resize, [1,28,28]) #reshaping the image , i.e. to tell the label that i am not predicting for 10 labels but for only 1 label
input_prediction = model.predict(image_reshape) #predicting the values
input_pred_label = np.argmax(input_prediction) # the above obtained prediction is in the form of probability , so converting it into a label
print('The Handwritten Digit is recognised as ', input_pred_label)


# C:/Users/anshi/Downloads/mnist 7.jpg




