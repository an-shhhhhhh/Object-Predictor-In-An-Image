# kaggle competitions download -c dogs-vs-cats
import tensorflow as tf
import numpy as np
import PIL
import cv2
import matplotlib.pyplot as plt
# filename = r'C:\Users\anshi\Downloads\butterfly.jpg'
# filename = r"C:\Users\anshi\Downloads\traffic-lights.jpg"
filename = r"C:\Users\anshi\Downloads\road_image.jpg"
# filename = r"C:\Users\anshi\Downloads\zoo.jpg"
# filename = r"C:\Users\anshi\Downloads\zebra.jpg"
from tensorflow.keras.preprocessing import image
img = image.load_img(filename, target_size = (224,224))
plt.imshow(img) #to display the image
plt.show()

#lets load the deep learning model
mobile = tf.keras.applications.mobilenet_v2.MobileNetV2() #mobilenet is a deep learning architecture for image classification
#we are using a pretrained architecture, so we are not building neural networks here

#we have 4 phases
# 1. creating a model
# 2. training a model
# 3. test or validate (in test case we provide labels)
# 4. predict (we can download any image and predict its objects)

#pre-proceessing of the image
from tensorflow.keras.preprocessing import image
img = image.load_img(filename, target_size = (224,224))
plt.imshow(img)
plt.show()
resized_img = image.img_to_array(img)
final_image = np.expand_dims(resized_img, axis = 0) # we need 4th dimension for prediction #Insert a new axis that will appear at the axis position in the expanded array shape. in this case we are inserting a new dimension at 0th axis
final_image = tf.keras.applications.mobilenet_v2.preprocess_input(final_image) #This function returns a Keras image classification model, optionally loaded with weights pre-trained on ImageNet.
print(final_image.shape)
predictions = mobile.predict(final_image)
# print(predictions) #it gives the output as all the values in the form of numerical probabilities predicted in terms of each 224 pixels in 4 dimensions
from tensorflow.keras.applications import imagenet_utils
results = imagenet_utils.decode_predictions(predictions) #converts the results into string values
print(results)










