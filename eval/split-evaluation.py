import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Function to load data
def load_data(filepaths):
    data = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        data.append(df.values)  # Assuming the data is already correctly formatted
    combined = np.vstack(data)
    return combined

# File paths
thinking_files = [
    'C:\\Users\\morot\\tabular\\mkeds\\sample_data\\thinking1.csv',
    'C:\\Users\\morot\\tabular\\mkeds\\sample_data\\thinking2.csv',
    'C:\\Users\\morot\\tabular\\mkeds\\sample_data\\thinking3.csv'
]
not_thinking_files = [
    'C:\\Users\\morot\\tabular\\mkeds\\sample_data\\not_thinking1.csv',
    'C:\\Users\\morot\\tabular\\mkeds\\sample_data\\not_thinking2.csv',
    'C:\\Users\\morot\\tabular\\mkeds\\sample_data\\not_thinking3.csv'
]

# Load and label data
thinking_data = load_data(thinking_files)
not_thinking_data = load_data(not_thinking_files)

# Labels: 1 for thinking, 0 for not thinking
X = np.vstack([thinking_data, not_thinking_data])
y = np.array([1] * len(thinking_data) + [0] * len(not_thinking_data))

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluation with train/test split
print("Evaluation with Train/Test Split:")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# CNN Model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# SVM Model
svm_model = SVC(kernel='rbf')

# Reshape for CNN and LSTM
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Training
cnn_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
lstm_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
svm_model.fit(X_train, y_train)

# Evaluation
cnn_predictions = cnn_model.predict(X_test_reshaped).round()
lstm_predictions = lstm_model.predict(X_test_reshaped).round()
svm_predictions = svm_model.predict(X_test)

print("CNN Accuracy:", accuracy_score(y_test, cnn_predictions))
print("LSTM Accuracy:", accuracy_score(y_test, lstm_predictions))
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))

# Evaluation with train/validation/test split
print("\nEvaluation with Train/Validation/Test Split:")
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# CNN Model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# SVM Model
svm_model = SVC(kernel='rbf')

# Reshape for CNN and LSTM
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Training with early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
cnn_model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=10, batch_size=32, verbose=0, callbacks=[early_stopping])
lstm_model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=10, batch_size=32, verbose=0, callbacks=[early_stopping])
svm_model.fit(X_train, y_train)

# Evaluation
cnn_predictions = cnn_model.predict(X_test_reshaped).round()
lstm_predictions = lstm_model.predict(X_test_reshaped).round()
svm_predictions = svm_model.predict(X_test)

print("CNN Accuracy:", accuracy_score(y_test, cnn_predictions))
print("LSTM Accuracy:", accuracy_score(y_test, lstm_predictions))
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))