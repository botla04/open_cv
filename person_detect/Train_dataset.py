import os
import cv2
import numpy as np
from PIL import Image


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):

    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

    #create empth face list
    faceSamples=[]

    #create empty ID list
    Ids=[]

    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')

        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')

        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])

        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)

        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('dataSet')
s = recognizer.train(faces, np.array(Ids))
print("Successfully trained")
recognizer.write('trainer.yml')


'''
k_date = str(datetime.today())
            k_date1 = k_date[:10]

            k_time = k_date.replace(":", "-")
            cv2.imwrite("unknown/" + k_time[:19] + '.jpg', img)
            e_mails = ["gusidi.somasekhar@gmail.com"]
            for m in e_mails:
                fromaddr = "somasekhar.halten@gmail.com"
                msg = MIMEMultipart()
                msg['From'] = fromaddr
                msg['To'] = m
                msg['Subject'] = "person is detected"
                body = ["un authorised person detected"]
                msg.attach(MIMEMultipart(body[0], 'plain'))
                filename = "unknown/" + k_time[:19] + '.jpg'
                attachment = open(filename, "rb")
                p = MIMEBase('application', 'octet-stream')
                p.set_payload(attachment.read())
                encoders.encode_base64(p)
                p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
                # # attach the instance 'p' to instance 'msg'
                msg.attach(p)
                #
                # # creates SMTP session
                s = smtplib.SMTP('smtp.gmail.com', 587)
                s.ehlo()
                #
                # # start TLS for security
                s.starttls()
                #
                # # Authentication
                s.login(fromaddr, "somaSekharGmail@5651")  # pwd
                #
                # # Converts the Multipart msg into a string
                text = msg.as_string()
                #
                # # sending the mail
                s.sendmail(fromaddr, m, text)
                print("sent")
                #
                # # terminating the session
                s.quit()
'''


