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
print("Please waiting for a moment!")
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

dataset=datasets.ImageFolder('images') # photos folder path 
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
print("It's Done!")