
import xml.etree.ElementTree as ET
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, concatenate, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# Đọc biến vocab từ file
with open('DLmodel/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
embedding_dim = 128
num_filters = 128
filter_sizes = [3, 4, 5]
max_seq_length = 256  # Define your max sequence length here
title_input = Input(shape=(max_seq_length,))
text_input = Input(shape=(max_seq_length,))

# Input for title
title_embedding = Embedding(vocab_size, embedding_dim)(title_input)
title_conv_blocks = []
for filter_size in filter_sizes:
    title_conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(title_embedding)
    title_pool = GlobalMaxPooling1D()(title_conv)  # Sử dụng GlobalMaxPooling1D thay vì MaxPooling1D
    title_conv_blocks.append(title_pool)
title_concat = concatenate(title_conv_blocks, axis=-1)
title_flat = Flatten()(title_concat)

# Input for text
text_embedding = Embedding(vocab_size, embedding_dim)(text_input)
text_conv_blocks = []
for filter_size in filter_sizes:
    text_conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(text_embedding)
    text_pool = GlobalMaxPooling1D()(text_conv)  # Sử dụng GlobalMaxPooling1D thay vì MaxPooling1D
    text_conv_blocks.append(text_pool)
text_concat = concatenate(text_conv_blocks, axis=-1)
text_flat = Flatten()(text_concat)


# Combine the two inputs
combined = concatenate([title_flat, text_flat])

# Additional layers of the model
dense1 = Dense(128, activation='relu')(combined)
output = Dense(3, activation='softmax')(dense1)

# Build the model
model_CNN = Model(inputs=[title_input, text_input], outputs=output)

model_CNN.load_weights("DLmodel/resources/CNNClassification.h5")
# Compile the model after transferring weights
model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model_CNN.summary()

def predict(comment):
    output = model_CNN.predict(comment)
    
    # Hiển thị kết quả dự đoán
    print("Kết quả dự đoán:")
    for i, prob in enumerate(output[0]):
        print(f"Lớp {i}: {prob * 100:.2f}%")
    return output

