import numpy as np
import torch
import pandas as pd                         # 'pandas' to manipulate the dataset.
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split          # 'train_test_split' for splitting the data into train and test data.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences       # 'pad_sequences' for having same dimmension for each sequence.
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
import gensim.models.keyedvectors as word2vec #need to use due to depreceated model
# import gensim.models.word2vec as Word2Vec #need to use due to depreceated model
import gc
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from scipy import linalg
import pandas as pd
from keras.utils import to_categorical

import pandas as pd
from keras.utils import to_categorical

label_mapping = {'pos': 1, 'neu': 2, 'neg': 3}

X_train = pd.read_excel("data/Data_train.xlsx")

if X_train.isnull().values.any():
    X_train = X_train.dropna()

print(X_train.shape)

X_train = X_train[['processed_title', 'processed_review', 'user_rate']]
X_train_title = X_train['processed_title'].apply(str)
X_train_text = X_train['processed_review'].apply(str)

# Map string labels to integer values
y_train = X_train['user_rate'].map(label_mapping)

# Convert to one-hot encoding
train_labels = to_categorical(y_train - 1, num_classes=3)



# Assuming 'pos', 'neu', 'neg' are your unique classes
label_mapping = {'pos': 1, 'neu': 2, 'neg': 3}

X_test = pd.read_excel("data/Data_test.xlsx")

if X_test.isnull().values.any():
    X_test = X_test.dropna()

print(X_test.shape)
print(X_test.columns)

X_test = X_test[['processed_title', 'processed_review', 'user_rate']]
X_test_title = X_test['processed_title'].apply(str)
X_test_text = X_test['processed_review'].apply(str)

# Map string labels to integer values
y_test = X_test['user_rate'].map(label_mapping)

# Convert to one-hot encoding
test_labels = to_categorical(y_test - 1, num_classes=3)

from gensim.models import KeyedVectors

w2vModel = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=50000)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train_title)
tokenizer.fit_on_texts(X_train_text)
tokenizer.fit_on_texts(X_test_title)
tokenizer.fit_on_texts(X_test_text)

train_title = tokenizer.texts_to_sequences(X_train_title)
train_text = tokenizer.texts_to_sequences(X_train_text)
test_title = tokenizer.texts_to_sequences(X_test_title)
test_text = tokenizer.texts_to_sequences(X_test_text)

vocab_size = len(tokenizer.word_index) + 1

max_len_title = 30
max_len_text = 150

train_title = pad_sequences(train_title, padding = 'post', maxlen = max_len_title)
train_text = pad_sequences(train_text, padding = 'post', maxlen = max_len_text)
test_title = pad_sequences(test_title , padding = 'post', maxlen = max_len_title)
test_text = pad_sequences(test_text , padding = 'post', maxlen = max_len_text)


from sklearn.model_selection import train_test_split

train_sent_titles, val_sent_titles, train_sent_texts, val_sent_texts, train_ratings, val_ratings = train_test_split(train_title, train_text, train_labels, test_size=0.1)
print(len(train_sent_texts), len(val_sent_texts))
print(len(train_sent_titles), len(val_sent_titles))
print(len(train_ratings), len(val_ratings))

MAX_LEN = 256

def convert_sents_ids(sents):
    ids = []
    for sent in sents:
        sent = str(sent)
        # Split the sentence into words and encode them individually
        encoded_sent = [w2vModel[word] if word in w2vModel else np.zeros(w2vModel.vector_size) for word in sent.split()]
        ids.append(encoded_sent)

    # Pad sequences to a fixed length using Keras
    ids = pad_sequences(ids, maxlen=MAX_LEN, dtype="float32", value=0, truncating="post", padding="post")
    return ids

train_title_ids = convert_sents_ids(train_sent_titles)
train_text_ids = convert_sents_ids(train_sent_texts)
val_title_ids = convert_sents_ids(val_sent_titles)
val_text_ids = convert_sents_ids(val_sent_texts)
test_title_ids = convert_sents_ids(test_title)
test_text_ids = convert_sents_ids(test_text)



train_labels = torch.tensor(train_ratings)
test_labels = torch.tensor(test_labels)


from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(train_labels.shape)
# Assuming train_labels, val_labels, test_labels are NumPy arrays
y_train = to_categorical(train_labels)
y_train_1d = np.argmax(y_train, axis=1)
y_train_1d = np.argmax(y_train_1d, axis=1)
print(y_train_1d.shape)

train_title_ids_2d = np.array(train_title_ids).reshape(len(train_title_ids), -1)
train_text_ids_2d = np.array(train_text_ids).reshape(len(train_text_ids), -1)

train_data = np.concatenate((train_title_ids_2d, train_text_ids_2d), axis=1)


test_title_ids_2d = np.array(test_title_ids).reshape(len(test_title_ids), -1)
test_text_ids_2d = np.array(test_text_ids).reshape(len(test_text_ids), -1)

test_data = np.concatenate((test_title_ids_2d, test_text_ids_2d), axis=1)




lr = LogisticRegression()

lr.fit(train_data, y_train_1d)

y_pred_lr = lr.predict(test_data)

acc_lr = accuracy_score(np.argmax(test_labels, axis=1), y_pred_lr)
conf = confusion_matrix(np.argmax(test_labels, axis=1), y_pred_lr)
clf_report = classification_report(np.argmax(test_labels, axis=1), y_pred_lr)

print(f"Accuracy Score of Logistic Regression is: {acc_lr}")
print(f"Confusion Matrix:\n{conf}")
print(f"Classification Report:\n{clf_report}")