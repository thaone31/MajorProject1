from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import numpy as np

def train_model():
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    train_title_ids = data['train_title_ids']
    train_text_ids = data['train_text_ids']
    test_title_ids = data['test_title_ids']
    test_text_ids = data['test_text_ids']
    y_train_1d = data['y_train_1d']
    test_labels = data['test_labels']

    lr = LogisticRegression()

    # Train the Logistic Regression model
    train_title_ids_2d = np.array(train_title_ids).reshape(len(train_title_ids), -1)
    train_text_ids_2d = np.array(train_text_ids).reshape(len(train_text_ids), -1)
    train_data = np.concatenate((train_title_ids_2d, train_text_ids_2d), axis=1)
    lr.fit(train_data, y_train_1d)

    test_title_ids_2d = np.array(test_title_ids).reshape(len(test_title_ids), -1)
    test_text_ids_2d = np.array(test_text_ids).reshape(len(test_text_ids), -1)
    test_data = np.concatenate((test_title_ids_2d, test_text_ids_2d), axis=1)
    y_pred_lr = lr.predict(test_data)

    acc_lr = accuracy_score(test_labels, y_pred_lr)
    conf = confusion_matrix(test_labels, y_pred_lr)
    clf_report = classification_report(test_labels, y_pred_lr)

    print(f"Accuracy Score of Logistic Regression is: {acc_lr}")
    print(f"Confusion Matrix:\n{conf}")
    print(f"Classification Report:\n{clf_report}")

    return lr, acc_lr, conf, clf_report