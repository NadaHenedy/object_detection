import numpy as np
import cv2
import time
data=open("coco.names").read().strip().split()# we read the names of objects by .read and strip it is to remove the white spaces from the start and end of string and split it to be as a list
print(data)
colors_data=np.random.randint(0,255,size=(len(data),3),dtype="uint8")# we create random integer numbers from 0 to 255 the array of colors then we create a matrix with the size of data as rows and 3 columes with type uint 8 that is means correct number not decimal
darknet=cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")# to read network model stored in darknet model files , we give it cfg file description of network architecture that is yolov3 , and darknet model that is the weights file with learned network that is yolov3 weights
layer_names=darknet.getLayerNames()# gives names of layers to get boxes and segments and indexes of layers
capture=cv2.VideoCapture("C:/Users/LENOVO/PycharmProjects/trainig/venv/drive-download-20221222T181847Z-001/videos/street2.mp4")# we get the video that we will work on it
try:#to catch some exception might be happened when taking layer names
    layer_names=[layer_names[z[0]-1] for z in darknet.getUnconnectedOutLayers()]# we make for loop for every layer in output layers give me layer name at each index , getUnconnectedOutLayers() is used for obtaining indexes of unconnected output layers
except IndexError :
    layer_names = [layer_names[z - 1] for z in darknet.getUnconnectedOutLayers()]#  we make for loop for every layer in output layers give me layer name
while True:
    _,image=capture.read()# we want to read image that will be taken from video , _ is for true or false
    height,weidhts=image.shape[:2]# to initialize the height and wights of capture image from video and the maximum index is 2 so height will be at index 0 and widht at index 1
    transform = cv2.dnn.blobFromImage(image,1/255 , (416,416),swapRB=True , crop=False)# we transform image to binary image  and give it the image and scale factor will be 1/255 , size of image , we swap RB channels because open cv assume image as BGR and we didn't want to crop image
    darknet.setInput(transform)# set image in darknet as binary image
    start=time.perf_counter()# we begin timer to our performance
    output=darknet.forward(layer_names)# takes layer name and it gives a list of layer output
    time_taken=time.perf_counter()-start# calculate the time taken to this operation done
    print(time_taken)
    box=[]# initialize box
    confidence=[]# initialize confidence interval
    id_class=[]# initialize id of object
    for i in output:# we make an iteration for every layer in output
       for j in i:# we make for loop for every j takes every thing that has been detcted by i
           detcet_score=j[5:]# takes the values that has been detcted , so we start to take the index 5 because it is out of box
           id_classes=np.argmax(detcet_score)# we want to get the highest score from the dected scores
           confidence_level=detcet_score[id_classes]# it is the confidence level so we put the highest confidence level i get
           if confidence_level>0.5:
               boxes=j[:4]*np.array([weidhts,height,weidhts,height])# we get dimensions of box that has inside it object and multiply it with array to get the dimensions of box detction
               (c_x,c_y,w,h)=boxes.astype("int")# the box of detcted object dimension
               x=int(c_x-(w/2))# get dimension of x axis of box
               y=int(c_y-(h/2))# get dimension of y axis of box
               box.append([x,y,int(w),int(h)])# we add the box size to a list of box
               confidence.append(float(confidence_level))# we add to the list the confidence level to each box
               id_class.append(id_classes)# to add the name of class to each box
    indexes=cv2.dnn.NMSBoxes(box,confidence,0.5,0.5)# it is a list that removes the boxes with low score and keep boxes with high score for each object with low and high threshold
    if len(indexes)>0:# if there is a box
        for k in indexes.flatten():# we will make for loop for indexes and turn it to row
            x,y,weidhts,height=box[k][0] , box[k][1],box[k][2] , box[k][3]# we initialize dimensions on box by every index 0 for x dimension , index 1 for y dimension , index 2 for weidth , index 3 for height
            color = [int(C) for C in colors_data[id_class[k]]]# specify color for each object so we make for loop of colors  and give it id class to give every id color different
            cv2.rectangle(image,(x,y),(x+weidhts,y+height),color=color,thickness=1)# we do rectangle to the image ,so we give to it the image that he capture it and gie it the x,y_axis , x+weidths , y+heights to colse the rectangle , color to the box , thickness for box
            name_confidence=f"{data[id_class[k]]}: {confidence[k]:.2f}"# to print on every rectangle the name of object and confidence
            (text_width,text_height)=cv2.getTextSize(name_confidence,cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,thickness=1)[0]# the height and width of text that will be written above box ,  so it will take the text of name_confidence and get the type of font and font scale and thickness of font , at index 0 to get the values
            coordiantes=((x,y-5),(x+text_width+2,y-5-text_height))# specify the coordinates of box
            copy_image=image.copy()# take a copy form image
            cv2.rectangle(copy_image,coordiantes[0], coordiantes[1], color=color, thickness=cv2.FILLED)# we make rectangle to the copy image
            image=cv2.addWeighted(copy_image,0.6,image,0.4,0)# to merge between the 2 images by add weighted we give alpha to copy image and beta to image they are the brightness of image and gama 0
            cv2.putText(image,name_confidence,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0),thickness=1)# add text to the merge image
    cv2.imshow("image",image)
    if ord("q")==cv2.waitKey(1):
       break
capture.release()
cv2.destroyAllWindow()






