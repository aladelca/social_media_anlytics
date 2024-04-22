import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings 
import glob
from PIL import Image
import os

def convert_timestamp(input_timestamp):
    # Replace 'T' with '_' and ':' with '-'
    formatted_timestamp = input_timestamp.replace('T', '_').replace(':', '-')
    # Append '_UTC' to the formatted timestamp
    formatted_timestamp += '_UTC'
    return formatted_timestamp

def get_dataframe(influencer):
    data = pd.DataFrame()
    for i in glob.glob(f'{influencer}/descargas_instagram/info/*'):
        df = pd.read_json(i).T
        data = pd.concat([data, df] , axis=0)
    data = data.reset_index(drop=True)
    data = data.drop_duplicates(subset = ['likes','fecha_utc'])
    data['key_date'] = data['fecha_utc'].apply(convert_timestamp)
    data['influencer'] = influencer 
    return data

def get_descriptions(influencer):
    data = pd.DataFrame()
    captions = []
    filenames = []
    for i in glob.glob(f'{influencer}/*txt'):
        with open(i, 'r') as file:
            text = file.read() 
        captions.append(text)
        filenames.append(i.replace(f'{influencer}/','').replace('.txt',''))
        
    data['filename'] = filenames
    data['caption'] = captions
    data['influencer'] = influencer 
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

def find_image_filename(key_date, image_list):
    for filename in image_list:
        if filename.startswith(key_date):
            return filename
    return None 

def load_and_prepare_images(df, image_size=(128, 128)):
    images = []
    for img_path in df['img_source']:
        img = Image.open(img_path)
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        img = np.array(img)
        img = img.astype('float32') / 255.0  # Normalizar a [0, 1]
        images.append(img)
    return np.array(images)

def preprocessing(influencer):
    df = get_dataframe(influencer)
    descriptions = get_descriptions(influencer)
    df = df.merge(descriptions, left_on=['influencer','key_date'], right_on=['influencer','filename'], how='left')

    source_directory = f'{influencer}/'  
    target_directory = f'{influencer}/images'  

    resize_images(source_directory, target_directory)

    path = f'{influencer}/images/'
    image_files = glob.glob(f'{path}*')
    image_files = [i.replace(path,'') for i in image_files]

    df['img_source'] = df['key_date'].apply(lambda x: path + find_image_filename(x, image_files))
    return df

from google.cloud import vision

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
def get_features_images(influencer):
    df = pd.DataFrame()
    for i in glob.glob(f'{influencer}/images/*jpg'):
        a = detect_objects(i)
        df = pd.concat([df, a])
    df['influencer'] = influencer   
    return df

def clean_filename(row):
    path = row['path']
    influencer = row['influencer']

    path = path.replace(f'{influencer}/images/', '')
    
    path = path.replace('_1.jpg', '')
    path = path.replace('_2.jpg', '')
    path = path.replace('_3.jpg', '')
    path = path.replace('_4.jpg', '')
    path = path.replace('_5.jpg', '')
    path = path.replace('.jpg', '')
    return path

def preprocess_features(df_features, data):
    b = df_features.pivot_table(
                            index=['path','influencer'],
                            columns='objects',
                            values='scores',
                            aggfunc='mean'
                        )
    b.columns = [i.lower().replace(' ','_') for i in b.columns]
    b = b.reset_index()
    b = b.fillna(0)
    b['clean_path'] = b.apply(clean_filename, axis=1)
    b = b.drop_duplicates(subset=['clean_path','influencer'])
    c = b.drop(columns=['path'])
    data = data.merge(c, left_on=['key_date','influencer'], right_on=['clean_path','influencer'], how='left')
    return data
