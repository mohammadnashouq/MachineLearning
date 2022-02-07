from flask import Flask
from flask import Blueprint
import asyncio
#from flask_script import Manager, Server
from multiprocessing import Process
from flask_cors import CORS, cross_origin
from flask import Flask, jsonify, request,make_response,after_this_request
from pathlib import Path
import os
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['DEBUG'] = True
import time
import requests
req = requests.Session()
#app.config.from_pyfile('../config.py')
#app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'
#db.init_app(app)
from requests.adapters import HTTPAdapter
req.mount('https://',HTTPAdapter(pool_connections = 5000, pool_maxsize = 5000))
#github_adapter = HTTPAdapter(max_retries=5)
def Generate_image_data(Conversation_number):
    
    map_result = {}
    
    
    return map_result


def tryLoadTesting(external_chat,Conversation_number,image_path):
        
        _dirpath = Path(os.getcwd())
        _dirpath = str(Path(os.getcwd()))#[:-6]

        file = open(_dirpath + "/"+str(Conversation_number )+ '.log', "w",encoding='utf8')
        print("Thread number " + str(Conversation_number) +" starts")
        data = Generate_image_data(Conversation_number)
        start_time = time.time()
        files = {"file":open(image_path, 'rb')}
        
        response = requests.post(external_chat,files = files , timeout=700)
        finish_time = time.time()
        duration = finish_time - start_time 
            
        
        if response.status_code != 200:
        
            response1 = response.json()
            finish_time = time.time()
            file.write("duration@" + str(duration)  +  "@Thread number@" + str(Conversation_number) +"@finished with status_code@" + str(response.status_code) + "@message@" + str(response.text) +  "/n")
            file.close()
        else:
            
            response1 = response.json()
            if "class" in response1:
                    file.write(  "duration@" + str(duration)  +  "@Thread number@" + str(Conversation_number) +"@finished with status_code@" + str(response.status_code) + "@class@"+  str(response1["class"]) + "@finished" + "/n")
                    file.close()
                
                
         
                
                
                
    
@app.route('/tryLoadTesting', methods=['post'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def loadTesting():
    
    try:
   
        load_test_data = request.get_json('data')
        hand_detect_link = load_test_data["url"] + "/api/gesture/HandDetectByPhoto"
        Conversations_number = load_test_data["users_number"]
        image_path = load_test_data["image_path"]
        files = {"file":open(image_path, 'rb')}
        #data = {"file":files}
        response = req.post(hand_detect_link,files = files,timeout=700)
        print(response.text)
        for Conversation_number in range(Conversations_number):
            thread = Process(target=tryLoadTesting,args = (hand_detect_link,Conversation_number,image_path))
            thread.start()
            
        return jsonify({}),200
        
    except Exception as e:
        return jsonify({"message":str(e)}),400


        
@app.route('/getthreadingreport/<int:threads>', methods=['get'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def getthreadingreport(threads):
    
    try:
        from flask import send_from_directory
        import pandas as pd
        threads = threads
        times = []
        report = {"duration (Second)":[],"thread_number":[],"status":[],"reply":[]}
        for i in range(threads):
            _dirpath = Path(os.getcwd())
            _dirpath = str(Path(os.getcwd()))#[:-6]
            path = str(_dirpath)+  "/"+str(i) + '.log'
            
            with open(path) as f:
                mylist = f.read().replace('\n', '')
            splited_lines = mylist.split('@')
            print(i,splited_lines)
            if splited_lines == [""]:
                duration = 0
                thread_number = i
                status = "file is empty because you did not wait the test untill complete or you did not use the last version of stress_testing"
                reply = ""
            else:
                duration = splited_lines[1]
                
                thread_number = splited_lines[3]
                status = splited_lines[5]
                
                reply = splited_lines[7]
                
            report["duration (Second)"].append(duration) 
            report["thread_number"].append(thread_number) 
            report["status"].append(status)
            report["reply"].append(reply)
                        
        
        
        df1 = pd.DataFrame(report)
        directory = str(_dirpath) + "/"
        file_name = "report" + ".xlsx"
        df1.to_excel(directory + file_name)     
        
        return send_from_directory(directory,file_name, as_attachment=True)
        
    except Exception as E:
        return jsonify({"message":str(E)}),400



    
if __name__ == '__main__':
    app.run(debug=True,threaded=True)
    