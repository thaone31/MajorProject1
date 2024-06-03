from sklearn.neighbors import KNeighborsClassifier
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

    knn = KNeighborsClassifier(n_neighbors=5)
    # Train the Logistic Regression model
    train_title_ids_2d = np.array(train_title_ids).reshape(len(train_title_ids), -1)
    train_text_ids_2d = np.array(train_text_ids).reshape(len(train_text_ids), -1)
    # Chuẩn bị dữ liệu
    train_data = np.concatenate((train_title_ids_2d, train_text_ids_2d), axis=1)
    # Huấn luyện mô hình Naive Bayes
    knn.fit(train_data, y_train_1d)

    test_title_ids_2d = np.array(test_title_ids).reshape(len(test_title_ids), -1)
    test_text_ids_2d = np.array(test_text_ids).reshape(len(test_text_ids), -1)
    test_data = np.concatenate((test_title_ids_2d, test_text_ids_2d), axis=1)
    # Dự đoán trên tập kiểm tra
    y_pred_knn = knn.predict(test_data)

    # Đánh giá hiệu suất
    acc_knn = accuracy_score(test_labels, y_pred_knn)
    conf_knn = confusion_matrix(test_labels, y_pred_knn)
    clf_report_knn = classification_report(test_labels, y_pred_knn)

    print(f"Accuracy Score of KNN is: {acc_knn}")
    print(f"Confusion Matrix:\n{conf_knn}")
    print(f"Classification Report:\n{clf_report_knn}")

    return knn, acc_knn, conf_knn, clf_report_knn

