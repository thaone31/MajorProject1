import pandas as pd

# Đọc dữ liệu từ các file
data_train = pd.read_excel('data/Data_train.xlsx')
data_test = pd.read_excel('data/Data_test.xlsx')

# Gộp 2 file dữ liệu thành 1 file mới
data = pd.concat([data_train, data_test], ignore_index=True)

# Lưu file dữ liệu mới
data.to_excel('data.xlsx', index=False)