from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import cvlib as cv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob

#-------------------------------------------------TEST------------------------------------------------------------------
with tf.device('/cpu:0'):

    model_name = 'vgg.model'
    
    f1_dims = (96,96,3)
    
    image_files =  r"Sets\Test"
    data_path = os.path.join(image_files,'*g')
    files = glob.glob(data_path)
    
    error_count = 0
    
    # result image directories are created
    output_directory = 'testResults'
    if not os.path.exists( output_directory ):
        os.makedirs( output_directory )
    
    right_output_directory = output_directory + '/right'
    wrong_output_directory = output_directory + '/wrong'
    
    if not os.path.exists( right_output_directory ):
        os.makedirs( right_output_directory )
    
    if not os.path.exists( wrong_output_directory ):
        os.makedirs( wrong_output_directory )
    
    
    imge = []
    face = []
    confidence = []
    classes = ['man','woman']
    
    startX = 0
    startY = 0
    
    idx = 0
    label = ''
    
    face_crop = []
    
    # model is loaded from file
    model = tf.keras.models.load_model( model_name )
    
    for f1 in files:
        imge = cv2.imread(f1)
         
        # detect faces in the image
        face, confidence = cv.detect_face(imge)
    
        # get the first detected face
        f = face[ 0 ]
        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
    
        # draw rectangle over face
        cv2.rectangle(imge, (startX,startY), (endX,endY), (0,255,0), 2)
    
        # crop the detected face region
        face_crop = np.copy(imge[startY:endY,startX:endX])
    
        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
    
        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        print(conf)
        print(classes)
    
        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
    
        Y = startY - 10 if startY - 10 > 10 else startY + 10
    
        # write label and confidence above face rectangle
        cv2.putText(imge, "{}: {:.2f}%".format(label, conf[idx] * 100), (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
        # check if wrong decision
        if label != f1.split(os.path.sep)[-1].split()[ 0 ]:
            error_count = error_count + 1
    
            # save output image
            cv2.imwrite( wrong_output_directory + '/' + f1.split(os.path.sep)[-1], imge )
        else:
            # save output image
            cv2.imwrite( right_output_directory + '/' + f1.split(os.path.sep)[-1], imge )
    
    
    # display results
    total_number_of_images = len( files )
    if total_number_of_images > 0:
        print( 'Wrong prediction count = ' + str(error_count) + ' in a total number of ' + str(total_number_of_images) )
        print( 'Accuracy rate in % = ' + ( str(( total_number_of_images - error_count ) / total_number_of_images * 100 )) )
