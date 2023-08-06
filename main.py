#study about
# neural networks- see the saved tutorial in anshika.singgh chrome id

# deep learning -
# Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, enabling them to recognize a stop sign, or to distinguish a pedestrian from a lamppost. It is the key to voice control in consumer devices like phones, tablets, TVs, and hands-free speakers. Deep learning is getting lots of attention lately and for good reason. It’s achieving results that were not possible before.
# In deep learning, a computer model learns to perform classification tasks directly from images, text, or sound. Deep learning models can achieve state-of-the-art accuracy, sometimes exceeding human-level performance. Models are trained by using a large set of labeled data and neural network architectures that contain many layers.

# rgb and grayscale channels

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#loading the image through matplotlib.image module
img = mpimg.imread(r"C:\Users\anshi\Downloads\puppy.jpg")
print(type(img)) #returns the type of image or how is  it stored in the pycharm
print(img.shape) #output is 355, 474, 3.....first represnts the no. of rows of pixel matrix, second the columns, and third represnts the colour channel(in coloured channels...R,G,B so 3 coloured channels)
print(img) #stored in the form of 3 matrices , one for red, another for green and third one for blue, it is stored in the similar format
#how we can print numpy array as an image
img_plot = plt.imshow(img)
plt.show()

#resizing the image using pillow library or PIL library
from PIL import Image #pillow library
img = Image.open(r"C:\Users\anshi\Downloads\puppy.jpg")
img_resized = img.resize((200,200)) #we want to convert the image of the size of 200x200
img_resized.save('dog_image_resized.jpg') #this will save the resized image in the same directory as project is saved in.
#displaying the image from numpy array
img_res = mpimg.imread(r"C:\Users\anshi\PycharmProjects\imageProcessing\dog_image_resized.jpg")
img_res_plot = plt.imshow(img_res)
plt.show()
print(img_res.shape) #output will be (200,200,3)

#converting rgb images to grayscale images using opencv
import cv2
img = cv2.imread(r"C:\Users\anshi\Downloads\puppy.jpg")
type(img)
print(img.shape)
#converted rgb image to gray image using cv2.COLOR_RGB2GRAY function
graysacle_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #variable named grayscale which will store the value of rgb image converted into grayscale
print(type(graysacle_image)) #output stored as numpy array
print(graysacle_image.shape) #output is (355,474) ..the 3 is now removed, because the grayscale has only 1 channel
cv2.imshow("grayscale_image")
#saving the grayscale image using cv2
cv2.imwrite('dog_grayscale_image.jpg', graysacle_image)




