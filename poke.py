#dataset can be found here 
#https://www.kaggle.com/thedagger/pokemon-generation-one



#importing the required libraries
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from tensorflow.keras.layers import Conv2D, UpSampling2D,Dense
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img,array_to_img
from skimage.io import imshow
from skimage.transform import resize
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb

#rescaling the image for faster computation
train_datagen = ImageDataGenerator(

        rescale=1./255
)

#changing the current directory
os.chdir(r"C:\Users\Rishab\Downloads\pokemon-generation-one")
print(os.listdir())


#importing images from folders img-size =(64,64) and shuffling
train = train_datagen.flow_from_directory("dataset", 
                                          target_size=(64,64), 
                                          batch_size=64, 
                                          class_mode=None,
                                          shuffle=True)



X=[]
Y=[]

#putting the data into X and Y
#use 169 if my dataset is used else use 165
#In orrder to better the performance on porygon(as wanted) I have manually
#altered the dataset 
for i in range(165):#169 if porygon folder is copied 3 timesinto the dataset dataset is used
    print(i)
    for img in train[i]:
        try:
          lab = rgb2lab(img)
          X.append(lab[:,:,0]) 
          Y.append(lab[:,:,1:] / 128) 
        except:
         print('error{}'.format(i))

X = np.array(X)
X=X.reshape(X.shape+(1,))
Y = np.array(Y)
X.shape


#creating a convolutional model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2, input_shape=(64,64, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3) , activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'],)
model.summary()


#either fit hte model or load the model
model.fit(X,Y,validation_split=0.1,epochs=500, batch_size=64)


#OR
# model=load_model("fmodel4")
# model=load_model("fmodel5")

#OR
#preferable for porygon
#model=load_model("fmodel6")




#to evaluate change the locations of images based on ur computer
img1_color=[]
imgno1=r"C:\Users\Rishab\Desktop\dataset\Abra\2fd28e699b7c4208acd1637fbad5df2d.jpeg"
imgno2="C:/Users/Rishab/Downloads/pokemon-generation-one/dataset/Abra/e297786c64574fbbb264dc6274aa5864.jpg"
imgno3="C:/Users/Rishab/Downloads/pokemon-generation-one/dataset/Aerodactyl/ef7f7eeaa897478bbe1df7b0bb0522cb-39.jpg"
imgno4="C:/Users/Rishab/Downloads/pokemon-generation-one/dataset/Articuno/6c4329bebd9a453790ff8560a9965f53.jpg"
imgno5="C:/Users/Rishab/Downloads/pokemon-generation-one/dataset/Flareon/eb4b6891ed11438da3c4f6aee4f92d7c.jpg"
imgno6="C:/Users/Rishab/Downloads/pokemon-generation-one/dataset/Mewtwo/00000003.jpg"
imgno7="C:/Users/Rishab/Downloads/pokemon-generation-one/dataset/Mewtwo/00000052.png"
imgno8="C:/Users/Rishab/Downloads/pokemon-generation-one/dataset/Porygon/635b3d7b627c4c6aa6ec2f84ccaf9827.jpg"
#the image you want to make prediction on
currimg=imgno8

img  = load_img(currimg)
img_g  = load_img(currimg,color_mode="grayscale")
img1=img_to_array(load_img(currimg))
img1 = resize(img1 ,(64,64))
img1_color.append(img1)
img1_color = np.array(img1_color, dtype=float)
img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
img1_color = img1_color.reshape(img1_color.shape+(1,))
output1 = model.predict(img1_color)
output1 = output1*128
result = np.zeros((64, 64, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]
final = lab2rgb(result)



#to visualidse the results

#full grey image
imshow(np.array(img_g))

#predicted image
imshow(final)

#actual resized colored image
imshow(np.array(img.resize((64,64))))



#look at the performace of the model
model.evaluate(X,Y)
