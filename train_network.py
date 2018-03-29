import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import utils as np_utils
from lenet import LeNet
from imutils import paths
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse 
import random
import cv2
import os

#argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,help="path to output model")
ap.add_argument("-p","--plot",type=str,default="plot_1.png",help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 50
INIT_LR = 1e-3
BS = 32

print("[INFO] loading images...")
data = []
labels = []


imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = Image.open(imagePath)
    new_width  = 28
    new_height = 28
    image = image.resize((new_width, new_height), Image.ANTIALIAS) 
    image.load()
    image = np.asarray(image,dtype="int32")
    data.append(image/255.0)
       
    label = imagePath.split(os.path.sep)
    if "c1" in label[1]:
        label_ = 0
    if "c3" in label[1]:
        label_ = 1 
    if "c9" in label[1]:
        label_ = 2
    if "c10" in label[1]:
        label_ = 3

    labels.append(label_)
data = np.array(data)
labels = np.array(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=42)

trainY = np_utils.to_categorical(trainY,num_classes=4)
testY = np_utils.to_categorical(testY,num_classes=4)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

#initialize the model
print("[INFO] compiling model..")
model = LeNet.build(width=28,height=28,depth=3,classes=4)
opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

#train the network
print("[INFO] training network..")
H = model.fit_generator(aug.flow(trainX,trainY,batch_size=BS),validation_data=(testX,testY),steps_per_epoch = len(trainX)//BS,epochs=EPOCHS,verbose=1)


print("[INFO] serializing network")
model.save(args["model"])


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
