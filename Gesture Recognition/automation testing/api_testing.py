api_url = "http://160f-104-196-169-130.ngrok.io"
api_prefex = "/api/gesture"
HandDetectByPhoto = api_url + api_prefex +'/HandDetectByPhoto'
import requests
def test_HandDetectByPhoto(path):
    files = open(path,'rb')
    file = {"file":files}
    print("HandDetectByPhoto")
    res = requests.post(HandDetectByPhoto,files=file)
    print(res)
    res_json = res.json()
        
    if res.status_code   == 200:
        if "class" in res_json:
            if res_json["class"] == "u":
                print("test_HandDetectByPhoto passed successfully")
                return 0
                
    print("test_HandDetectByPhoto need to be fixed",str(res_json["message"]))
    return 1
    
                    
        
path = "11.jpg"     
test_HandDetectByPhoto(path)