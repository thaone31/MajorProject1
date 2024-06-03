from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import numpy as np

def train_model():
    with open('MLmodel/W2VWE/data.pkl', 'rb') as f:
        data = pickle.load(f)
    train_title_ids = data['train_title_ids']
    train_text_ids = data['train_text_ids']
    test_title_ids = data['test_title_ids']
    test_text_ids = data['test_text_ids']
    y_train_1d = data['y_train_1d']
    test_labels = data['test_labels']
    
    dt = DecisionTreeClassifier()
    train_title_ids_2d = np.array(train_title_ids).reshape(len(train_title_ids), -1)
    train_text_ids_2d = np.array(train_text_ids).reshape(len(train_text_ids), -1)
    # Chuẩn bị dữ liệu
    train_data = np.concatenate((train_title_ids_2d, train_text_ids_2d), axis=1)
    dt.fit(train_data, y_train_1d)

    test_title_ids_2d = np.array(test_title_ids).reshape(len(test_title_ids), -1)
    test_text_ids_2d = np.array(test_text_ids).reshape(len(test_text_ids), -1)
    test_data = np.concatenate((test_title_ids_2d, test_text_ids_2d), axis=1)
    # Dự đoán trên tập kiểm tra
    y_pred_dt = dt.predict(test_data)

    # Đánh giá hiệu suất
    acc_dt = accuracy_score(test_labels, y_pred_dt)
    conf_dt = confusion_matrix(test_labels, y_pred_dt)
    clf_report_dt = classification_report(test_labels, y_pred_dt)

    print(f"Accuracy Score of Decision Tree is: {acc_dt}")
    print(f"Confusion Matrix:\n{conf_dt}")
    print(f"Classification Report:\n{clf_report_dt}")
    return dt, acc_dt, conf_dt, clf_report_dt


train_model()