from sklearn.naive_bayes import MultinomialNB
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

    nb = MultinomialNB()
    # Train the Logistic Regression model
    train_title_ids_2d = np.array(train_title_ids).reshape(len(train_title_ids), -1)
    train_text_ids_2d = np.array(train_text_ids).reshape(len(train_text_ids), -1)
    # Chuẩn bị dữ liệu
    train_data = np.concatenate((train_title_ids_2d, train_text_ids_2d), axis=1)
    # Huấn luyện mô hình Naive Bayes
    nb.fit(train_data, y_train_1d)

    test_title_ids_2d = np.array(test_title_ids).reshape(len(test_title_ids), -1)
    test_text_ids_2d = np.array(test_text_ids).reshape(len(test_text_ids), -1)
    test_data = np.concatenate((test_title_ids_2d, test_text_ids_2d), axis=1)
    # Dự đoán trên tập kiểm tra
    y_pred_nb = nb.predict(test_data)

    # Đánh giá hiệu suất
    acc_nb = accuracy_score(test_labels, y_pred_nb)
    conf_nb = confusion_matrix(test_labels, y_pred_nb)
    clf_report_nb = classification_report(test_labels, y_pred_nb)

    print(f"Accuracy Score of Naive Bayes is: {acc_nb}")
    print(f"Confusion Matrix:\n{conf_nb}")
    print(f"Classification Report:\n{clf_report_nb}")

    return nb, acc_nb, conf_nb, clf_report_nb