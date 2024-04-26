import pandas as pd
import json
import glob
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import os
import warnings 
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay, recall_score
from PIL import Image
import os
from pylab import *
from utils.utils import *
from sklearn.model_selection import train_test_split
import json
import lzma
import json
from google.cloud import vision

def get_df_comments(path):

    files = []
    comments = []
    data = pd.DataFrame()
    for i in glob.glob(path):

        with lzma.open(i, "rt") as file:
            content = file.read()
        a = json.loads(content)
        comment = a['node']['edge_media_to_comment']['count']
        files.append(i)
        comments.append(comment)
    data['files'] = files
    data['comments'] = comments   
    return data


def resize_images(source_dir, target_dir, new_size=(128, 128)):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Recorrer todos los archivos en el directorio fuente
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_dir, filename)
            img = Image.open(img_path)
            img = img.resize(new_size, Image.Resampling.LANCZOS)  
            img.save(os.path.join(target_dir, filename))

def get_resized_images(path, size):
    for i in glob.glob(path):
        source_directory = f'{i}/'  
        target_directory = f'{i}/images'  

        resize_images(source_directory, target_directory, size)

def get_path(path):
    parts = path.split("/")

    
    before_images = parts[0]+'/'+parts[1]
    return before_images


def merged_data(data, path):
    name_keys = []
    file_keys = []
    df_file_name = pd.DataFrame()
    for i in glob.glob(path):
        parts = i.split("images")
        before_images = parts[0].strip("/")
        name_keys.append(before_images)
        for j in glob.glob(f'{i}*'):
            file_keys.append(j)
            

    df_file_name['name_keys'] = name_keys
    df_file_name['file_keys'] = file_keys
    data_merged = pd.merge(data, df_file_name, left_on='path', right_on='name_keys', how='left')
    return data_merged


def detect_objects(path):
    df = pd.DataFrame()
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    objects = []
    scores = []
    for object_ in response.localized_object_annotations:
        objects.append(object_.name)
        scores.append(object_.score)
      
    df['objects'] = objects
    df['scores'] = scores  
    df['path'] = path
    
    return df

def get_features_images_path():
    df = pd.DataFrame()

    for i in glob.glob('data/*/images/*.jpg'):
        a = detect_objects(i)
        a['path'] = i
        df = pd.concat([df, a])
   
    return df

def get_avg_features(features):
    b = features.pivot_table(
                            index=['path'],
                            columns='objects',
                            values='scores',
                            aggfunc='mean'
                        )
    b.columns = [i.lower().replace(' ','_') for i in b.columns]
    b = b.reset_index()
    b = b.fillna(0)
    return b


import csv


def get_list(file_path):
    #file_path = 'list_output.csv'
    data_list = []

# Open the text file in read mode
    with open(file_path, 'r') as file:
        # Read each line in the file
        for line in file:
            # Strip the newline character from the end of each line
            clean_line = line.strip()
            # Append the cleaned line to the list
            data_list.append(clean_line)
    return data_list