from sklearn import metrics
import matplotlib.pyplot as plt
import catboost as cb
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model

def training(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

def predict(model, x_test, threshold = 0.5):
    
    proba = model.predict_proba(x_test)
    preds = np.where(proba[:,1] > threshold, 1, 0)
    return preds, proba

def get_metrics(y_pred, y_proba, y_test, plot=False):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
    specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
    cm = metrics.confusion_matrix(y_test, y_pred)
    try:
        auc = metrics.roc_auc_score(y_test, y_proba[:,1])
    except:
        auc = metrics.roc_auc_score(y_test, y_proba)
    if plot:
        disp = metrics.ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()
    else:
        pass
    final_metrics = {
        'accuracy': accuracy, 
        'sensitivity': sensitivity, 
        'specificity': specificity, 
        'auc': auc}
    return final_metrics, cm

def create_new_variables(data):
    data['A_follow_ratio'] = data['A_follower_count'] / data['A_following_count']
    data['B_follow_ratio'] = data['B_follower_count'] / data['B_following_count']
    data['A_B_follow_ratio'] = data['A_follow_ratio'] / data['B_follow_ratio']

    data['A_mention_ratio'] = data['A_mentions_received'] / data['A_mentions_sent']
    data['B_mention_ratio'] = data['B_mentions_received'] / data['B_mentions_sent']
    data['A_B_mention_ratio'] = data['A_mention_ratio'] / data['B_mention_ratio']

    data['A_retweet_ratio'] = data['A_retweets_received'] / data['A_retweets_sent']
    data['B_retweet_ratio'] = data['B_retweets_received'] / data['B_retweets_sent']
    data['A_B_retweet_ratio'] = data['A_retweet_ratio'] / data['B_retweet_ratio']


    data['A_B_followers_ratio'] = data['A_follower_count'] / data['B_follower_count']
    data['A_B_following_ratio'] = data['A_following_count'] / data['B_following_count']
    data['A_B_posts_ratio'] = data['A_posts'] / data['B_posts']
    data['A_B_listed_ratio'] = data['A_listed_count'] / data['B_listed_count']
    data['A_B_mentions_received_ratio'] = data['A_mentions_received'] / data['B_mentions_received']
    data['A_B_mentions_sent_ratio'] = data['A_mentions_sent'] / data['B_mentions_sent']
    data['A_B_retweets_received_ratio'] = data['A_retweets_received'] / data['B_retweets_received']
    data['A_B_retweets_sent_ratio'] = data['A_retweets_sent'] / data['B_retweets_sent']
    data['A_B_network_feature_1_ratio'] = data['A_network_feature_1'] / data['B_network_feature_1']
    data['A_B_network_feature_2_ratio'] = data['A_network_feature_2'] / data['B_network_feature_2']
    data['A_B_network_feature_3_ratio'] = data['A_network_feature_3'] / data['B_network_feature_3']
    return data

def replace_inf(data):
    data = data.replace([np.inf, -np.inf], np.nan)
    return data

def scale_data(x_train):
    x_train_scaled = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    return x_train_scaled

def create_siamese_network(input_shape):
    input_A = Input(shape=input_shape)
    input_B = Input(shape=input_shape)

    # Convolutional layers
    conv_layer1 = Conv1D(32, 3, activation='relu')
    conv_layer2 = Conv1D(64, 3, activation='relu')

    # Shared dense layers
    dense_layer1 = Dense(128, activation='relu')
    dense_layer2 = Dense(64, activation='relu')

    # Process each input through convolutional and shared layers
    processed_A = GlobalMaxPooling1D()(conv_layer2(conv_layer1(input_A)))
    processed_B = GlobalMaxPooling1D()(conv_layer2(conv_layer1(input_B)))

    # Concatenate processed features
    merged_features = Concatenate()([processed_A, processed_B])

    # Dense layers
    merged_features = dense_layer2(dense_layer1(merged_features))

    # Output layer
    output = Dense(1, activation='sigmoid')(merged_features)

    model = Model(inputs=[input_A, input_B], outputs=output)
    return model


def plot_training_validation(history):
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def objective(trial, x_train, y_train, x_val, y_val):

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "depth": trial.suggest_int("depth", 1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 1000, 10000),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    model = cb.CatBoostClassifier(**param, random_state=123)

    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0, early_stopping_rounds=100)

    preds, proba = predict(model, x_val)
    metrics, cm = get_metrics(preds, proba, y_val, False)
    auc = metrics['auc']
    return auc

def get_metrics_per_author(df_comments, df_submission):
    ## Get number of comments and submissions per author    
    submission_summary = df_submission.groupby('author')['id'].nunique().sort_values(ascending = False)
    comments_summary = df_comments.groupby('author')['id'].nunique().sort_values(ascending = False)
    return submission_summary, comments_summary

def break_parent_id(df_comments):
    # Break the parent_id into level and the reald parent id
    df_comments['responded_to']=df_comments['parent_id'].apply(lambda x: x.split('_')[1])
    df_comments['level']=df_comments['parent_id'].apply(lambda x: x.split('_')[0])
    return df_comments

def get_responses_to_submissions(df_comments, df_submission):
    df_merged = pd.merge(
        df_comments[['author','responded_to','level','id']], 
        df_submission[['author','id']], 
        left_on='responded_to', 
        right_on='id', 
        suffixes=('_kid', '_parent'))
    return df_merged
def generate_response_submissions_label(df_merged):
    df_merged['Link Type'] = 'respond to a submission'
    return df_merged

def get_responses_to_comments(df_comments):
    df_merged2 = df_comments[['author','responded_to','level','id']].merge(
        df_comments.loc[df_comments.level=='t1',
                        ['author','responded_to','level','id']],
                        left_on = 'id', 
                        right_on = 'responded_to',
                        suffixes=('_parent', '_kid'))
    return df_merged2
def generate_response_comment_label(df_merged):
    df_merged['Link Type'] = 'respond to a comment'
    return df_merged

def merge_links(df_responses_to_submissions, df_responses_to_comments):
    df_result = pd.concat((df_responses_to_submissions[['author_kid','author_parent','Link Type']],df_responses_to_comments[['author_kid','author_parent','Link Type']]),axis=0)
    return df_result

def drop_deleted_interactions(df_result):
    df_result = df_result.loc[(df_result.author_kid != '[deleted]') & (df_result.author_parent != '[deleted]') ]
    return df_result

def network_preprocessing_analysis(df_comments, df_submissions):
    submission_summary, comments_summary = get_metrics_per_author(df_comments, df_submissions)
    df_comments = break_parent_id(df_comments)
    df_responses_submission = get_responses_to_submissions(df_comments, df_submissions)
    df_responses_submission = generate_response_submissions_label(df_responses_submission)
    df_responses_comments = get_responses_to_comments(df_comments)
    df_responses_comments = generate_response_comment_label(df_responses_comments)
    df_final = merge_links(df_responses_submission, df_responses_comments)
    df_final = drop_deleted_interactions(df_final)
    return df_final

def plot_network(G, title):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels= True, node_size=8, node_color='skyblue', font_size=5, arrows=True)
    plt.title(title)
    plt.show()

def get_centrality_metrics(G):
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    metrics_df = pd.DataFrame({
    'Degree': degree,
    'Betweenness': betweenness,
    'Closeness': closeness
        })
    metrics_df.index.rename('author', inplace = True)
    return metrics_df

def preprocessing_metrics(df_metrics):
    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(df_metrics)
    normalized_metrics_df = pd.DataFrame(normalized_metrics, columns=df_metrics.columns, index=df_metrics.index)
    return normalized_metrics_df

def get_top_influencers(normalized_metrics_df, weights, n):

    # Define weights for each metric
    w1, w2, w3, w4, w5 = weights

    # Calculate the scores
    scores = w1 * normalized_metrics_df['Degree'] + \
            w2 * normalized_metrics_df['Betweenness'] + \
            w3 * normalized_metrics_df['Closeness'] + w4*normalized_metrics_df['#Posts'] + w5 * normalized_metrics_df['#Comments']

    # Add the scores to the DataFrame
    normalized_metrics_df['Score'] = scores

    # Get the top n influencers
    top_influencers = normalized_metrics_df['Score'].nlargest(n)
    return top_influencers

def plot_roc_auc(y_test, probas):
    preds = probas[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_precision_recall(y_test, probas):
    y_proba = probas[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    auc_precision_recall = metrics.auc(recall, precision)
    plt.figure()
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precisión-Recall')
    plt.legend(loc='best')
    plt.show()
    return precision, recall, thresholds


def main_process(model, x_train, y_train, x_test, y_test, plot=False, threshold=0.5, roc_curve = False):
    model = training(model, x_train, y_train)
    y_pred, y_proba = predict(model, x_test, threshold)
    if roc_curve:
        plot_roc_auc(y_test, y_proba)
    else:
        pass
    metrics, cm = get_metrics(y_pred, y_proba, y_test, plot)
    return model, metrics, cm
