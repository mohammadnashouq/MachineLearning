import requests
import cv2

cap=cv2.VideoCapture(0)
api_link = "http://127.0.0.1:5000/api/HandDetectByPhoto"
i =0
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30,30)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
while True:
    i = i + 1
    
    name = "frame%d.jpg"%i
    ret,frame1=cap.read()
    result, frame = cv2.imencode('.jpg', frame1, encode_param)
    files = {'file': frame}
    cv2.imshow('original_photo',frame1)
    r = requests.post(api_link, files=files)
    res = r.json()
    print(res)
    cv2.imshow('original_photo',frame1)
    cv2.putText(frame1,res["class"],
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
    cv2.imshow('processed_image',frame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()

out.release()

cv2.destroyAllWindows()