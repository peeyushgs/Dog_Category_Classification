from google.colab import drive

drive.mount("/content/drive")

# importing neccessary libraries

import os
import pandas as pd
import os, shutil, math, scipy, cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from PIL import Image
from PIL import Image as pil_image
from PIL import ImageDraw
from time import time
from glob import glob
from skimage.io import imread
from IPython.display import SVG
from scipy import misc, ndimage
from scipy.ndimage import zoom
from keras.utils.np_utils import to_categorical


def label_assignment(img, label):
    return label


def training_data(label, data_dir):
    for img in (os.listdir(data_dir)):
        label = label_assignment(img, label)
        path = os.path.join(data_dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (imgsize, imgsize))

        X.append(np.array(img))
        Z.append(str(label))


X = []
Z = []
imgsize = 150

supergroups = os.listdir('drive/MyDrive/dogs/dog_groups')
# supergroups= ['Herding', 'Hound', 'Non-Sporting', 'Sporting', 'Terrier', 'Toy', 'Working']

dog_supergroup = {}

for each_group in supergroups:
    dog_species_files = os.listdir('drive/MyDrive/dogs/dog_groups/' + each_group)
    species = []
    for each_species in dog_species_files:
        species.append(each_species[10:].strip())
        training_data(each_species[10:].strip(), 'drive/MyDrive/dogs/dog_groups/' + each_group + '/' + each_species)

    dog_supergroup[each_group] = species

print(len(set(Z)))


def key_dog(dog_supergroup, species):
    for each in dog_supergroup:
        for each_species in dog_supergroup[each]:
            if each_species == species:
                return each

fig, ax = plt.subplots(5, 2)
fig.set_size_inches(15, 15)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(Z))
        ax[i, j].imshow(X[l])
        ax[i, j].set_title(key_dog(dog_supergroup, Z[l]) + ': ' + Z[l])

plt.tight_layout()


label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Z)
Y = to_categorical(Y, 49)
X = np.array(X)
X = X / 255

#

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=69)

del X, Y

from keras.preprocessing.image import ImageDataGenerator

augs_gen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)

augs_gen.fit(x_train)

len(x_train)

