#Crop Face from Video and Split DataSet
import cv2 
import uuid
import glob
import pandas as pd
from imageio import imread,imsave
from skimage.transform import resize
from tqdm import tqdm
import dlib
import os
from mtcnn.mtcnn import MTCNN
import glob
import shutil
import numpy as np
detector = MTCNN() 

train_paths = glob.glob("video/*")
df_name_video = []
for i,train_path in tqdm(enumerate(train_paths)):
    name_video = train_path.split("\\")[-1]    
    df_name_video.append(name_video)   
for video in df_name_video:
    name_dic=video.split('.')[0]    
    dic="./images/"+name_dic+"/"     
    try: 
        os.mkdir(dic)        
    except:
        pass 
    img_name=dic       
    counter=0
    i=0 
    cap = cv2.VideoCapture("./video/"+video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            i+=1
            if(i%3==0):
                imgname=img_name+"/"+name_dic+"_0{}.jpg".format(str(counter))
                cv2.imwrite(imgname, frame)
                cv2.imshow('frame', frame)
                counter+=1             
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
###################################################
#
#           Split DataSet
#           80% Train
#           20% Test
#           Remove All user have under 3 picture
###################################################
train_paths = glob.glob("./images/*")

number_in_array=[]
direc=[]
# Filter image
for i,train_path in tqdm(enumerate(train_paths)):
    name = train_path.split("\\")[-1]    
    dir=train_path+"/*"
    dir=glob.glob(dir)
    dir_name=train_path.replace("\\","/")
    lenght=len(dir)
    if(lenght<4):
        shutil.rmtree(dir_name)
# Creat list Dataset to split
for i,train_path in tqdm(enumerate(train_paths)):
    name = train_path.split("\\")[-1]    
    dir=train_path+"/*"
    dir=glob.glob(dir)
    dir_name=train_path.replace("\\","/")  

    direc.append(dir_name)

    lenght=len(dir)
    test_sample=int(np.round(lenght*0.2))
    if test_sample==0:
        test_sample=1    
    number_in_array.append(test_sample)
# Split Dataset
data = list(zip(number_in_array, direc))
dataset=pd.DataFrame(data,columns=["Sample","Directory"])  
for samp,direc in dataset.iterrows():
    src_dir = direc.values[1]
    dst_dir = "./test/"      
    count=0    
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        count+=1
        if(count<=number_in_array[samp]):
            shutil.move(jpgfile, dst_dir)