from flask import Flask, jsonify, request,make_response
import requests
import glob, os
import os
#from flask_cors  import CORS, cross_origin
from pathlib import Path
app = Flask(__name__, static_url_path='')
import sys
api_prefex = "/api/gesture"
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from datetime import datetime as date_time
from datetime import date

Images_ALLOWED_EXTENSIONS = set(['jpeg','png','jpg','gif'])
Excel_ALLOWED_EXTENSIONS = set(['xlx','xlsx'])


app.config['CORS_HEADERS'] = '*'
app.config['DEBUG'] = True
_dirpath = str(Path(os.getcwd()))

_save_images_path = str(_dirpath) 
_save_excel_path = str(_dirpath) 

app.config['UPLOAD_FOLDER'] = _save_images_path
app.config['UPLOAD_Excel'] = _save_excel_path
log = "log"

import sys
from logging.config import dictConfig

sys.stdout = open('log/file.log', "a",encoding='utf8')

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout', 
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})



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

model = tf.keras.models.load_model('saved_models/my_model')

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


def test_mode(path):
    dat_l=[]

    prevTime = 0

    dat = np.zeros((1,63))
    lab = np.array([])
    letlist=['0','0','0','0','0']
    
    image = cv.imread(path)
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
                sample = landmark_to_data(hand_landmarks)
                tf_sample=tf.convert_to_tensor(sample)
                tf_sample_good=tf.reshape(tf_sample,[1,63])
                pred=model.predict(tf_sample_good)
                letter=chr(np.argmax(pred)+97)
                # letlist.pop(0)
                # letlist.append(letter)
                
                return letter



""" To use this model call the function test_mode(path) and provide path to the image.
In return it will output the letter (prediction). Load the 'model' on line 75 in the file which will call and import this 
file """



def Images_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Images_ALLOWED_EXTENSIONS

def Excel_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Excel_ALLOWED_EXTENSIONS

def edit_name_of_images(data):
    
    if "image" in data:
        kwargs["data"]["image"] = data["image"].split("\\")[-1]
        
    
def predect_hand_class_by_Image(path):
    
    return "Class A"
    
def predect_hand_class_by_Video(path):
    
    return "Class A"

def HandDetectByPhotoFun(Image):
    try:
    
        if file and Images_allowed_file(file.filename):
            
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_Excel'], filename))
            path = str(app.config['UPLOAD_Excel']) + str(filename)
            response = predect_hand_class_by_Image(path)
        
        
        return jsonify({"class":response})
    except Exception as e:
        return jsonify({"message":str(e)}),400



@app.route(api_prefex +'/HandDetectByPhoto',methods = ['post'])
def HandDetectByPhoto():
    try:
        
        if not 'file' in request.files:
            resp = jsonify({'message': 'No file part in the request'})
            resp.status_code = 400
            return resp
        
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message': 'No file selected for uploading'})
            resp.status_code = 400
            return resp
        
            
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_Excel'], filename +str(".jpg") ))
        path = str(app.config['UPLOAD_Excel']) + str(filename)+str(".jpg")
        response = test_mode(path)
        
        print("------------------------------- /HandDetectByPhoto Service has been requested --------------------------------------",date_time.now()," ","File Name: " ,path," request status": response.status_code,"  response: ", response)
    
        return jsonify({"class":response})
    except Exception as e:
        print("------------------------------- /HandDetectByPhoto Service has been requested --------------------------------------",date_time.now()," ","File Name: " ,path," request status": response.status_code,"  response: ", str(e))
    
        return jsonify({"message":str(e)}),400

@app.route(api_prefex +'/HandDetectByVideo',methods = ['post'])
def HandDetectByVideo():
    
    try:
        
        if not 'file' in request.files:
            resp = jsonify({'message': 'No file part in the request'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message': 'No file selected for uploading'})
            resp.status_code = 400
            return resp
        if file and Images_allowed_file(file.filename):
            
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_Excel'], filename))
            path = str(app.config['UPLOAD_Excel']) + str(filename)
            response = predect_hand_class_by_Video(path)
        
        
        return jsonify({"class":response})
    except Exception as e:
        return jsonify({"message":str(e)}),400


@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # Process the image frame
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

    

if __name__ == '__main__':
    
	app.run(host='0.0.0.0',debug=True) #run app on port 8080 in debug mode