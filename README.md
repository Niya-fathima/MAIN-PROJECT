# MAIN-PROJECT
Kmct - college of engineering
                                                              Title :WILDLIFE THREAT EARLY DETECTION AND WARNING SYSTEM

                                                              
ABSTRACT :

In regions abundant with wild animals such as elephants, tigers, and wild pigs, the delicate balance between human activity and wildlife presence poses a constant threat. Failure to
promptly detect oncoming wild animals can lead to life-threatening encounters, emphasizing the pressing need for an advanced surveillance and alert system. Traditional monitoring
methods, often manual or reliant on conventional cameras, fall short in providing timely alerts and accurate species identification. This project addresses these challenges by int-
roducing an intelligent surveillance system that leverages TensorFlow and Convolutional Neural Networks (CNN) for enhanced monitoring capabilities.

The proposed system employs continuous video recording from strategically placed cameras, focusing on agricultural land and hilly terrains, to detect and predict the movement of 
wild animals. Successive frame differencing identifies moving objects, while TensorFlow extracts unique key features from a wildlife dataset. To overcome the critical challenge of 
species identification,the system utilizes Haar features to characterize moving objects. The collected data forms a robust training set, enabling accurate identification of elephan-
ts, tigers, and wild pigs in real-time images captured by the cameras.

This project is specifically tailored to address the challenges of human-wildlife interaction, emphasizing its relevance and applicability to this specific locale.The system aims to 
provide a targeted solution that aligns with the unique characteristics and requirements of the local environment. The project’s design and implementation take into consideration the 
specificneeds of the region, making it a bespoke solution that directly contributes to the conservation efforts and safety concerns within this geographical area. Through collaborat-
ion with the Govt of kerala, the project aims to enhance the effectiveness of wildlife monitoring, response mechanisms, and overall conservation strategies in this particular forest 
region.

Distinct modules for administrators, forest officers, and users have been integrated into the system. Administrators manage division and station information, animal databases, and 
complaint handling. Forest officers play an active role in monitoring, forwarding complaints, and receiving alerts, while users can register, lodge complaints,and access information 
about animals and notifications. This project not only represents an innovation in wildlife monitoring but also a comprehensive solution that combines technological advancements with
practical applications. The objective is to minimize the risks associated with human-animal interaction, providing timely alerts and fostering safer coexistence between humans and
wildlife in vulnerable regions. Through this intelligent surveillance system, we aim to reduce response times, mitigate potential dangers, and contribute to the conservation efforts 
of both human lives and wild species.




CODE:



Camera module:


from tkinter import Tk, Label, Entry, Button
# import MySQLdb
import requests
import datetime
from wild_app.newcnn2 import predictcnn
import numpy as np
import cv2
import time

import time
import cv2
import os
import pymysql

# from wild_app.models import *

IMG_DIR = './static/sample'
from keras.engine.saving import load_model
import numpy as np
from keras.preprocessing import image
import os
# from src import database
sdThresh = 20
font = cv2.FONT_HERSHEY_SIMPLEX
#TODO: Face Detection 1
def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist
cv2.namedWindow('frame')
cv2.namedWindow('dist')
#capture video stream from camera source. 0 refers to first camera, 1 referes to 2nd and so on.
cap = cv2.VideoCapture(0)
# cap1 = cv2.VideoCapture(1)
_, frame1 = cap.read()
_, frame1 = cap.read()
_, frame1 = cap.read()
_, frame1 = cap.read()
_, frame1 = cap.read()
# _, frame2 = cap.read()
facecount = 0
flag=0
imgcount=0
imgname=1
cam="4"

# labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
labelsPath = os.path.sep.join([r"C:\Users\Anwar\Downloads\wildlfe (2)\wildlfe\wild_app\yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([r"C:\Users\Anwar\Downloads\wildlfe (2)\wildlfe\wild_app\yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join([r"C:\Users\Anwar\Downloads\wildlfe (2)\wildlfe\wild_app\yolo-coco", "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
(W, H) = (None, None)
ret, first_frame = cap.read()
ret, first_frame = cap.read()
ret, first_frame = cap.read()
ret, first_frame = cap.read()
ret, first_frame = cap.read()

# Convert the first frame to grayscale
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Compute the total number of pixels in the frame
total_pixels = first_frame_gray.shape[0] * first_frame_gray.shape[1]

# Initialize variables to store the cumulative absolute difference
total_diff = 0

# Loop through the video frames
frame_count = 0
while(True):
    _, frame3 = cap.read()
    # _,frame1=cap1.read()
    rows, cols, _ = np.shape(frame3)
    # cv2.imshow('dist', frame3)
    dist = distMap(frame1, frame3)
    frame_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the current frame and the first frame
    frame_diff = cv2.absdiff(frame_gray, first_frame_gray)

    # Compute the mean absolute difference (MAD) between corresponding pixels
    diff_value = np.sum(frame_diff) / total_pixels
    print(diff_value,"diff_value====================")
    print(diff_value,"diff_value====================")
    print(diff_value,"*********************************")
    if diff_value>40:
        # print(dist,"===============")
        # apply Gaussian smoothing
        mod = cv2.GaussianBlur(dist, (9,9), 0)
        # apply thresholding
        _, thresh = cv2.threshold(mod, 100, 255, 0)
        # calculate st dev test
        _, stDev = cv2.meanStdDev(mod)
        # cv2.imshow('dist', mod)
        if W is None or H is None:
            (H, W) = frame3.shape[:2]
        print(round(stDev[0][0],0),"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        cv2.putText(frame1, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        a = stDev
        blob = cv2.dnn.blobFromImage(frame3, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4,
                                0.5)
        status=True
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]

                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                           confidences[i])
                if LABELS[classIDs[i]] == "person":
                    print("person detected")
                    status = False
        if status:
            print("a",a)
            # #print(a[0][0])
            # #print(type(a[0][0]))

            # cv2.imwrite("F:\driver final\driver final\driver\src\static\sample", frame1)

            # cv2.imwrite(r"C:\\Users\\HP\\Downloads\\early warning new\\early warning\\src\\static\\tested\\"+str(imgname)+".jpg", frame3)
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = "static/tested/" + date + ".jpg"
            ani = date + ".jpg"
            cv2.imwrite(r"C:\Users\Anwar\Downloads\wildlfe (2)\wildlfe\static\tested\\"+str(date)+".jpg", frame3)

            path = "static/tested/"+str(imgname)+".jpg"
            img=str(date)+".jpg"
            # img="sdre.jpg"

            res = predictcnn(os.path.join(r"C:\Users\Anwar\Downloads\wildlfe (2)\wildlfe\static\tested", img))
            print("Res",res)
            # ob=notification_table()
            # ob.camera=camera_table.objects.get(id=1)
            # ob.notification=image
            # ob.date=datetime.datetime.now()
            # ob.save()

            # if res==0:
            #     animal="elephant"
            # elif res==1:
            #     animal="lion"
            # elif res==2:
            #     animal="tiger"
            # elif res==3:
            #     animal="wolf"
            # elif res==4:
            #     animal="zebra"
            # con = pymysql.connect(host='localhost', port=3306, user='root', password='123456789', db='forestfiredetection')
            # cmd = con.cursor()
            if res!=5:
                if status:
                    print(res,"==================================")
                    fn=datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png"
                    cv2.imwrite(r"C:\Users\Anwar\Downloads\wildlfe (2)\wildlfe\media/"+fn,frame3)
                    res=requests.get("http://127.0.0.1:8000/insertnotification?fn="+fn+"&cid=1")
                    print(res,"kkkkkkkkkkkkkkk")

            imgcount+=1
            imgname+=1
            print(imgcount,"++++++++++++++++++++++++++++")
    # if imgcount==30:
    #         imgname=0
    #         imgcount=0
    #         flag=0
    #
    #         print("okkkkkkkkkkkkkkkkkkkkkk")
    #     # qry="insert into attension values(null,%s,curdate(),curtime(),%s,'pending')"
    #     # values=(uid,str(ddd[0]))
    #     # database.iud(qry, values)
    #     #TODO: Face Detection 2
    cv2.imshow('dist', frame3)
    cv2.imshow('frame', frame1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()




