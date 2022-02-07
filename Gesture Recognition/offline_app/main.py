import cv2 as cv

import numpy as np
import pandas as pd

import tensorflow as tf

import datetime
now = datetime.datetime.now()

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

import time


def landmark_to_data(landmarks):
    #coord_list=[]
    #for i in hand_landmarks:
    coord_list=np.array([])
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = landmark.x
        landmark_y = landmark.y
        landmark_z = landmark.z
        coord_list = np.append(coord_list,np.array([landmark_x,landmark_y,landmark_z]))
    return coord_list


def test_mode(image):
    dat_l=[]

    prevTime = 0

    dat = np.zeros((1,63))
    lab = np.array([])
    letlist=['0','0','0','0','0']
    
    # image = cv.imread(path)
    with mp_hands.Hands(
        min_detection_confidence=0.5,       #Detection Sensitivity
        min_tracking_confidence=0.5) as hands:
        
        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        #Classify hand gesture
        if results.multi_hand_landmarks!=None:
            for hand_landmarks in results.multi_hand_landmarks:
                #print(hand_landmarks)
                #print(landmark_to_data(hand_landmarks))
                sample = landmark_to_data(hand_landmarks)
                tf_sample=tf.convert_to_tensor(sample)
                tf_sample_good=tf.reshape(tf_sample,[1,63])
                #print(tf_sample_good)
                pred=model.predict(tf_sample_good)
                prediction = (np.max(pred)*100)
                letter=chr(np.argmax(pred)+97)
                # letlist.pop(0)
                # letlist.append(letter)
                
                return letter, prediction
        else:
            return "non", 0.9


model = tf.keras.models.load_model('saved_models/my_model')

""" To use this model call the function test_mode(path) and provide path to the image.
In return it will output the letter and prediction percentage. Load the 'model' on line 75 in the file which will call and import this 
file """

