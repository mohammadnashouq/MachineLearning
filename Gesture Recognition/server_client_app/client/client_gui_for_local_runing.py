import main as mn
import cv2, PySimpleGUI as sg
import requests

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import gtts
from time import sleep
import os
import pyglet
import playsound

#import jamspell

import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


from autocorrect import Speller
spell = Speller()

##################### Prepare Model ######################
# Prepare data generator for standardizing frames before sending them into the model.
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

def SLT_estimation(image):
	letter, conf = mn.test_mode(image)
	return letter, conf

def landmark_to_data(landmarks):
	coord_list=np.array([])
	for _, landmark in enumerate(landmarks.landmark):
		landmark_x = landmark.x
		landmark_y = landmark.y
		landmark_z = landmark.z
		coord_list = np.append(coord_list,np.array([landmark_x,landmark_y,landmark_z]))
	return coord_list

def test_mode(image):
	dat_l = []

	prevTime = 0

	dat = np.zeros((1, 63))
	lab = np.array([])
	letlist = ['0', '0', '0', '0', '0']

	with mp_hands.Hands(
			min_detection_confidence=0.5,  # Detection Sensitivity
			min_tracking_confidence=0.5) as hands:
		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		image.flags.writeable = False
		results = hands.process(image)
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				mp_drawing.draw_landmarks(
					image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
		currTime = time.time()
		fps = 1 / (currTime - prevTime)
		prevTime = currTime
		# Classify hand gesture
		if results.multi_hand_landmarks != None:
			for hand_landmarks in results.multi_hand_landmarks:
				sample = landmark_to_data(hand_landmarks)
				tf_sample = tf.convert_to_tensor(sample)
				tf_sample_good = tf.reshape(tf_sample, [1, 63])
			prediction = np.array(model.predict(tf_sample_good))
			predicted_class = classes[prediction.argmax()]  # Selecting the max confidence index.
			return predicted_class, prediction
		else:
			prediction = np.array(model.predict(tf_sample_good))
			predicted_class = classes[0]
			return predicted_class, prediction

###################### GUI Init ##########################
sg.theme('Dark Blue 3')

# Setting up the input image size and frame crop size.
IMAGE_SIZE = 200
CROP_SIZE = 400

camera_layout = [
				  [sg.Image(filename='', key='image')],
				  [sg.Button('START'), sg.Button('STOP')],
				]

output_layout = [
				  [sg.Text(size=(37,1), key='-OUTPUT-', font= ("Helvetica",22))],
				  [sg.Input(key='-IN-'), sg.Button('INPUT')],
				  [sg.Button('LISTEN'), sg.Button('CLEAR'), sg.Button('CANCEL')],
				]

layout = [[sg.Text('Sign Language Translator', size=(34, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
		  [sg.Frame('I am gesturing:', camera_layout, font='Helvetica 12')],
		  [sg.Frame('I am saying:', output_layout, font='Helvetica 12')]]

window = sg.Window('Demo Application - OpenCV Integration', layout, location=(800, 400))

################### Text to Speech ###################

def tts(text):
	tts = gtts.gTTS(text=text, lang='en')
	filename = "_.mp3"
	tts.save(filename)
	playsound.playsound(filename)
	os.remove(filename)

################### Load Correction Language model ###################
# Create a corrector
#corrector = jamspell.TSpellCorrector()
# argument is a downloaded model file path
#corrector.LoadLangModel('/en.bin')   # it must be 'en.bin' but for a reason not working in windows so I stoped it temporory

######################LOOP############################
cap = cv2.VideoCapture(0)  # Setup the camera as a capture device
outpuut_str = ""
pre_predicted_class = "Nothing"
repeated = 0
while True:  # The PSG “Event Loop”
	### Apply model ###
	# Capture frame-by-frame.
	ret, frame = cap.read()

	predicted_class,prediction = SLT_estimation(frame)

	# Preparing output based on the model's confidence.
	prediction_probability = prediction
	if predicted_class == "{":
		predicted_class = "space";
	if prediction_probability > 0.5:
		# High confidence.
		cv2.putText(frame, '{} - {:.2f}%'.format(predicted_class, prediction_probability),
					(10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)

	elif prediction_probability > 0.2 and prediction_probability <= 0.5:
		# Low confidence.
		cv2.putText(frame, 'Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability),
					(10, 450), 1, 2, (0, 255, 255), 2, cv2.LINE_AA)
	else:
		# No confidence.
		cv2.putText(frame, classes[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)

	if (predicted_class != pre_predicted_class):
		if repeated >= 3:
			if len(pre_predicted_class) == 1:
				if pre_predicted_class == "{":
					outpuut_str = outpuut_str + ' '
				else:
					outpuut_str = outpuut_str + pre_predicted_class
				print(outpuut_str)
			if pre_predicted_class == "space":
				# outpuut_str = corrector.FixFragment(outpuut_str)
				outpuut_str = spell(outpuut_str)
				outpuut_str = outpuut_str + ' '
				print(outpuut_str)
			if pre_predicted_class == "delete":
				outpuut_str = outpuut_str + '\b'
				print(outpuut_str)
		pre_predicted_class = predicted_class
		repeated = 0
	else:
		repeated += 1
		print(repeated)

	### Check buttons ###
	event, values = window.Read(timeout=20, timeout_key='timeout')
	if event == sg.WIN_CLOSED or event == 'CANCEL': break
	if event == 'Ok': print('You entered ', values[0])
	if event == 'INPUT': outpuut_str = values['-IN-']
	#if event == 'LISTEN': window['-OUTPUT-'].update(values['-IN-'])
	if len(outpuut_str) != 0:
		if event == 'LISTEN': tts(outpuut_str)
	if event == 'CLEAR': outpuut_str = ""

	window['-OUTPUT-'].update(outpuut_str)
	window['image'].Update(data=cv2.imencode('.png', frame)[1].tobytes())

# When everything done, release the capture.
cap.release()
cv2.destroyAllWindows()