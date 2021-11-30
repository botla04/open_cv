import cv2
from datetime import *
import time
from pygame import mixer

period = 8
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX
mixer.init()
sound = mixer.Sound('alarm.wav')

while True:
    # date time
    k_date = str(datetime.today())
    k_date1 = k_date[:10]
    k_time = k_date.replace(":", "-")


    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.3, 7)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = recognizer.predict(roi_gray)
        print(id)
        if (conf < 50):
            if (id == 1):
                myid = 'id1'
                sound.stop()

            elif (id == 2):
                myid = 'id2'
                sound.stop()

        else:
            myid = 'Unknown'
            cv2.putText(img, str(myid) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
            cv2.imwrite("unknown/" + k_time[:19] + '.jpg', img)
            sound.play()

            break

        cv2.putText(img, str(myid) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
        cv2.imwrite("known/" + myid + k_time[:19] + '.jpg', img)

    cv2.imshow('frame', img)

#    print(myid)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()