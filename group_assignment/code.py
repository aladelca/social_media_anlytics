import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier 
from utils.functions import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import optuna
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

##### Part 1

# Read data
data = pd.read_csv('data/train.csv')
data_dl = data.copy()
data.head()

# Exploratory data analysis

sns.barplot(x = data.isna().mean(), y = data.isna().mean().index)
plt.title('Missing values in the data')
plt.show()

data['Choice'].value_counts().plot(kind = 'bar')
plt.title('Distribution of the target variable')
plt.show()

plt.figure(figsize = (10, 10))
sns.heatmap(data.corr(), annot = True, annot_kws={'size': 6})
plt.show()

## Creation of new features
data = create_new_variables(data)
data = replace_inf(data)

VARS = [
    'A_B_follow_ratio',
    'A_B_mention_ratio',
    'A_B_retweet_ratio',
    'A_B_followers_ratio',
    'A_B_following_ratio',
    'A_B_posts_ratio',
    'A_B_listed_ratio',
    'A_B_mentions_received_ratio',
    'A_B_mentions_sent_ratio',
    'A_B_retweets_received_ratio',
    'A_B_retweets_sent_ratio',
    'A_B_network_feature_1_ratio',
    'A_B_network_feature_2_ratio',
    'A_B_network_feature_3_ratio'
    ]

TARGET = ['Choice']

## Machine learning approach

x = data[VARS]
y = data[TARGET]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

model_list = {
    'xgb':XGBClassifier(random_state = 123),
    'catboost':CatBoostClassifier(random_state = 123, verbose = False),
    'lgbm':LGBMClassifier(random_state = 123, verbose = -1),
    'randomForest':RandomForestClassifier(random_state = 123)
    }

list_metrics = []
for i in model_list.values():
    model = i
    model, metrics, cm = main_process(model,x_train, y_train, x_test, y_test, False, 0.5)
    list_metrics.append(metrics['auc'])

a = sns.barplot(x = list(model_list.keys()), y = list_metrics)

for i in range(len(list_metrics)):
    a.text(i, round(list_metrics[i],3) + 0.01, round(list_metrics[i],3), color='black', ha="center")
plt.title('AUC score for different models')
plt.show()

## Deep learning approach

x = data_dl.drop(columns = ['Choice'])
y = data_dl[TARGET]
y = y.astype(np.float32)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
x_training, x_validation, y_training, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=123)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_training)
x_val_scaled = scaler.transform(x_validation)
x_test_scaled = scaler.transform(x_test)



input_shape = (11, 1)  # Number of features for each user
siamese_model = create_siamese_network(input_shape)

# Compile the model with Binary Crossentropy loss function
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape the input data for Conv1D layer
x_train_scaled = scale_data(x_train_scaled)
x_val_scaled = scale_data(x_val_scaled)
x_test_scaled = scale_data(x_test_scaled)
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = siamese_model.fit(
    [x_train_scaled[:, :11], x_train_scaled[:, 11:]], y_training,
    validation_data=([x_val_scaled[:, :11], x_val_scaled[:, 11:]], y_validation),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping]
)

plot_training_validation(history)

predictions_proba = siamese_model.predict([x_test_scaled[:, :11], x_test_scaled[:, 11:]])
predictions_proba = predictions_proba.flatten()
predictions = np.where(predictions_proba > 0.5, 1, 0)
predictions_proba = predictions_proba.flatten()

deep_learning_metrics = get_metrics(predictions, predictions_proba, y_test)
list_metrics.append(deep_learning_metrics[0]['auc'])

a = sns.barplot(x = list(model_list.keys()) + ['deep learning'], y = list_metrics)

for i in range(len(list_metrics)):
    a.text(i, round(list_metrics[i],3) + 0.01, round(list_metrics[i],3), color='black', ha="center")
plt.title('AUC score for different models')
plt.xticks(rotation = 45)

### Hyperparameter tuning only for xgboost

### Create validation set

x_training, x_val, y_training, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 123)
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, x_training, y_training, x_val, y_val), n_trials=100)

best_params = {
    'objective': 'CrossEntropy',
    'depth': 10,
    'learning_rate': 0.054142680909405445,
    'n_estimators': 7625,
    'boosting_type': 'Plain',
    'bootstrap_type': 'Bernoulli',
    'colsample_bylevel': 0.08320979403660492,
    'subsample': 0.6592979938870895
    }

## Optimizing threshold

model = CatBoostClassifier(**best_params, random_state = 123, verbose = False)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)
model = training(model, x_train, y_train)
preds,probas = predict(model, x_test, 0.5)
precision, recall, threshold = plot_precision_recall(y_test, probas)

final_threshold = threshold[np.argmax(recall == precision)]
final_model, metrics, cm = main_process(model, x_train, y_train, x_test, y_test, True, final_threshold, True)
print(metrics)
final_preds = np.where(final_model.predict_proba(x_test)[:,1]>final_threshold,1,0)

## Interpretation

df_importance = pd.DataFrame(final_model.feature_importances_,final_model.feature_names_, columns = ['importance']).sort_values(by = 'importance', ascending = False)   
sns.barplot(x = df_importance['importance'], y = df_importance.index)
plt.title('Feature importance')

explainer = shap.TreeExplainer(final_model)
shap_values = explainer(x_test)
shap.plots.beeswarm(shap_values)

test = data.iloc[x_test.index]
test['preds'] = final_preds
test['revenue'] = np.where(
    (test['Choice'] == test['preds']) & (test['Choice']==1), (test['A_follower_count']*0.0003)-10, 
    np.where((test['Choice'] == test['preds']) & (test['Choice']==0), (test['B_follower_count']*0.0003)-10, 0))

print(test[['revenue']].sum())

test['revenue_wo_analytics'] = np.where(test['Choice']==0, (test['B_follower_count']*0.0002)-5, 
                                        np.where(test['Choice']==1, (test['A_follower_count']*0.0002)-5, 0))
print(test[['revenue_wo_analytics']].sum())

#### Part II

df_submission = pd.read_json('data/0sanitymemes_submissions.zst' ,compression='infer',lines=True, encoding_errors = 'ignore')
df_comments = pd.read_json('data/0sanitymemes_comments.zst', compression='infer', lines=True)

### Preprocessing

df_result = network_preprocessing_analysis(df_comments, df_submission)

G = nx.DiGraph()

# Add edges from 'id_comment' to 'id_submission'
edges = df_result[['author_kid','author_parent']].values.tolist()
G.add_edges_from(edges)

plot_network(G, "Reddit Comments and Submissions Network")

submission_summary, comments_summary = get_metrics_per_author(df_comments, df_submission)
df_centralities = get_centrality_metrics(G)

df_centralities_joined = pd.concat([df_centralities, submission_summary, comments_summary],axis = 1)
df_centralities_joined.columns = ['Degree','Betweenness','Closeness','#Posts',"#Comments"]

df_centralities_processed = preprocessing_metrics(df_centralities_joined)
weights = (0.2,0.2,0.2,0.2,0.2)
top_influencers = get_top_influencers(df_centralities_joined, weights,20)
print(top_influencers)

top_influencers_100 = get_top_influencers(df_centralities_joined, weights,100)

### Create subgraph with top 100 influencers
subgraph = G.subgraph(top_influencers_100.index)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(subgraph, k=0.15)
nx.draw(subgraph, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=5, font_weight='bold', arrows=True, edge_color='gray', width=0.5)
plt.title("Top 100 Influencers Network")
plt.show()