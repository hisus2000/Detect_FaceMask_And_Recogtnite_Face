# Split dataset
from tqdm import tqdm
import glob
import pandas as pd
import shutil
import numpy as np

train_paths = glob.glob("./data_lfw/*")

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

for i,train_path in tqdm(enumerate(train_paths)):
    name = train_path.split("\\")[-1]    
    dir=train_path+"/*"
    dir=glob.glob(dir)
    dir_name=train_path.replace("\\","/")  

    direc.append(dir_name)

    lenght=len(dir)
    test_sample=int(np.round(lenght*0.2))
    # test_sample=1
    if test_sample==0:
        test_sample=1    
    number_in_array.append(test_sample)

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