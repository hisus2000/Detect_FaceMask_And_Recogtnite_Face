# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import glob
import shutil
import numpy as np
import tensorflow as tf

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

dataset=datasets.ImageFolder('./data_lfw') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = [] # list of cropped faces from photos folder
name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True) 
    if face is not None and prob>0.90: # if face detected and porbability > 90%
        emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
        # emb=tf.math.l2_normalize(emb)
        embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
        name_list.append(idx_to_class[idx]) # names are stored in a list    
# for i in embedding_list:
#     embedding_list[i]=tf.math.l2_normalize(embedding_list[i])
#Caculate mean each face   
meo=embedding_list     
df=pd.DataFrame(columns=["tensor","name"])
df["name"]=name_list
df["tensor"]=embedding_list
number_people=df.groupby('name')["tensor"].count()
tensor=df.groupby('name')["tensor"].sum()
embedding_list=tensor/number_people
name_list=dict.fromkeys(name_list)
name_list=list(name_list) 
# Normalize L2
for i in range (len(embedding_list)):
    embedding_list[i]=torch.nn.functional.normalize(embedding_list[i], p = 2, dim = 1)


data = [embedding_list, name_list]
torch.save(data, 'data.pt') # saving data.pt file

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import glob
import shutil
import numpy as np
import tensorflow as tf

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    emb=torch.nn.functional.normalize(emb, p = 2, dim = 1)  # Normolize l2
    saved_data = torch.load('data.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db,p=2).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list)) 

# Matching
train_paths = glob.glob("./test/*")
k=0
label=[]
predict=[]
data_result=pd.DataFrame(columns=["label","predict"])

for i,train_path in tqdm(enumerate(train_paths)):
    name = train_path.split("\\")[-1]    
    dir=train_path+"/*"
    dir=glob.glob(dir)
    dir_name=train_path.replace("\\","/")  

    #Name of Picture
    name_process=dir_name.split("/")[-1]
    name_process=name_process.split(".")[0]
    name_process=name_process.split("_0")[0]

    #process
    result = face_match(dir_name, 'data.pt')
    #add Data to Dataframe
    label.append(name_process)
    predict.append(result[0])
    if(result[0]==name_process):
        k+=1
#Save Value
data_result["label"]=label
data_result["predict"]=predict
data_result.to_csv("result.csv",index=False)
        
accuracy=((k/(i+1)))*100   
print("Accuracy of Model: {}".format(accuracy))  