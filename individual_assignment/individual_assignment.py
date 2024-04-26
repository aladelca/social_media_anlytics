# Load libraries

import pandas as pd
import numpy as np
import lda
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
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
import os

# Limit NumPy to use one thread
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

### Get comments and images from json files
data = get_df_comments('data/*/*json.xz')
### Resize images to avoid large time of preprocessing
#get_resized_images('data/*',(512,512))

### Get cleaned path
data['path'] = data['files'].apply(get_path)
print('cleaned data')
### Merge data with image path and comments
df_merged = merged_data(data, 'data/*/images/*')

print('merged data')

### Download features using google vision api (already donwloaded and saved in data folder as features.csv)

Application_Credentials = '/Users/aladelca/Downloads/massive-acrobat-421018-1d8b6ce1a11a.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Application_Credentials
try:
    df_features = pd.read_csv('features.csv')
    print('File found')
except:
    df_features = get_features_images_path()

### Merge data with features
features = get_avg_features(df_features)
data_final = df_merged.merge(features, left_on = ['file_keys'], right_on = ['path'], how = 'left')
data_final = data_final.fillna(0)

### Plot comments

sns.boxplot(data_final['comments'])
plt.title('Comments distribution')
plt.show()

############ Task A


import gensim
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary

# Tokenize documents

texts = [list(data_final.columns)]
# Dictionary and corpus for LDA
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


coherence_values = []
topic_numbers = range(2, 15)  # from 2 to 15 topics

for num_topics in topic_numbers:
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(topic_numbers, coherence_values)
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.title('Coherence Scores by Number of Topics')
plt.show()

#### Comments

#According to the coherence score, based on the proposed work from [Roder et al](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf), and applied with `gensim`, the ideal number of topics that maximize the coherence is 4, so that's the number that it is going to be used for this exercise

NOT_CONSIDERED_COLUMNS = ['files', 'likes', 'path_x', 'name_keys', 'file_keys', 'path_y','target', 'comments']
data_features = data_final.loc[:,~ data_final.columns.isin(NOT_CONSIDERED_COLUMNS)]
data_features = pd.DataFrame(np.where(data_features>0,1,0), columns = data_features.columns)
lda = LatentDirichletAllocation(n_components=4, random_state=42)

#Train the model.
lda.fit(data_features)



#print the top 25 words in each topic
n = 25



for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([data_features.columns[i] for i in topic.argsort()[-n:]])

#### Names for the topics

#According to the results, the most suitable names for the topics would be:

#* Topic 0: Clothing accesories 
#* Topic 1: Desk/business articles
#* Topic 2: Food and cook
#* Topic 3: clothes and people

df_topic = pd.DataFrame(lda.fit_transform(data_features), columns = ['clothing_accesories', 'desk_business_articles', 'food_cook', 'clothes_people'])
df_topic['comments'] = data_final['comments']
df_topic['quantiles'] = pd.qcut(df_topic['comments'], 4, labels = ['low', 'medium', 'high', 'very_high'])

df_analysis = df_topic.groupby(['quantiles']).agg({'clothing_accesories':'mean', 'desk_business_articles':'mean', 'food_cook':'mean', 'clothes_people':'mean'})
df_analysis = df_analysis.loc[['low','very_high'],:]
df_analysis.plot(kind = 'bar')
plt.title('Topics distribution by quantiles')
plt.xlabel('Quantiles')
plt.legend(title = 'Topics')
plt.show()

print(df_analysis)


data_features['comments'] = data_final['comments']
data_features['quantiles'] = pd.qcut(data_features['comments'], 4, labels = ['low', 'medium', 'high', 'very_high'])
print(data_features)


df_elements = data_features.groupby(['quantiles']).agg('sum').T
print(df_elements.sort_values('low', ascending=False).iloc[1:25]['low'])
print(df_elements.sort_values('very_high', ascending=False).iloc[1:25]['very_high'])

#### Comments and insights

#We can see that the quantile with higher levels of engagements has more weight of clothes and people elements of that topic. On the other hand, the quantile with the lowest engagement has, on average, more presence of elements of desk and business articles topic. Another interesting fact is that the quantile with the highest engagement has a little presence of elements of food and cook, and the quantile with the lowest level of engagement has more presence of those elements. On average, both quantiles has the same average of weight for clothing and accessories elements.

#Analyzing the specific elements present on the images with the lowest engagement, we can see that the most present elements are person, clothing, top, food and tableware; and even though the components person, clothing and top are shared for the quantiles with the higher and the lowest engagement, the difference comes with the other terms. For example, the quantile with the highest engagement has more photos with elements such as pants, outerwear and dress; which correlates with the findings at topic level.

### Task B

#According to the findings, to get more engagement, the post in the instagram page should have photos with more presence of people wearing fancy clothes and accesories, and avoid food or cooking content. To validate this findings, I will use the CNN trained for the group assignment (it has been train with instagram data as well but retrained using comments instead of likes) and analyze the predictions

from keras.models import load_model
import pickle
columns = get_list('list_output.txt')
model = load_model('model.h5')
esc = pickle.load(open('scaler.pkl', 'rb'))
def load_and_prepare_images(df, column_name, image_size=(128, 128)):
    images = []
    for img_path in df[column_name]:
        img = Image.open(img_path)
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        img = np.array(img)
        img = img.astype('float32') / 255.0  # Normalizar a [0, 1]
        images.append(img)
    return np.array(images)

def predict(path):
    source = path
    target = path+'/images'
    resize_images(source, target)
    features = pd.DataFrame()
    data_images = pd.DataFrame()
    data_images['path'] = glob.glob('demo/images/*.jpg')

    for i in glob.glob(f'{path}/images/*.jpg'):
        
        a = detect_objects(i)
        
        a['path'] = i
        features = pd.concat([features, a])
    
    b = features.pivot_table(
                                index=['path'],
                                columns='objects',
                                values='scores',
                                aggfunc='mean'
                            )
    b.columns = [i.lower().replace(' ','_') for i in b.columns]
    b = b.reset_index()
    b = b.fillna(0)
    NOT_CONSIDERED_COLUMNS = ['files', 'likes', 'path_x', 'name_keys', 'file_keys', 'path_y','target','path']
    c = pd.DataFrame(columns=columns)
    images = load_and_prepare_images(data_images, 'path')
    c[[i for i in b.columns if i not in NOT_CONSIDERED_COLUMNS]] = b[[i for i in b.columns if i not in NOT_CONSIDERED_COLUMNS]]
    c = c.fillna(0)
    c = esc.transform(c)
    data_images['preds'] =  model.predict([images, c])

    return data_images

glob.glob('demo/images/*.jpg')
print(predict('demo'))