# mkeds.py
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from neurosity import NeurositySDK
from dotenv import load_dotenv

class MKEDS:
    def __init__(self, cycles, not_thinking_duration, thinking_duration):
        load_dotenv()
        self.neurosity = NeurositySDK({
            "device_id": os.getenv("NEUROSITY_DEVICE_ID"),
        })
        self.neurosity.login({
            "email": os.getenv("NEUROSITY_EMAIL"),
            "password": os.getenv("NEUROSITY_PASSWORD")
        })
        self.cycles = cycles
        self.not_thinking_duration = not_thinking_duration
        self.thinking_duration = thinking_duration
        self.model = None

    def collect_data(self, label, duration):
        print(f"Please {label} for {duration} seconds.")
        data = []
        
        def callback(epoch):
            data.append(epoch)
        
        self.neurosity.brainwaves_raw(callback)
        time.sleep(duration)
        self.neurosity.remove_all_subscriptions()
        
        return data

    def preprocess_data(self, data):
        preprocessed_data = []
        labels = []

        for epoch in data:
            flattened_data = []
            for channel_data in epoch["data"]:
                flattened_data.extend(channel_data)
            preprocessed_data.append(flattened_data)
            label = 0 if "not thinking" in epoch["label"] else 1
            labels.append(label)

        preprocessed_data = np.array(preprocessed_data)
        labels = np.array(labels)

        return preprocessed_data, labels

    def create_cnn_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train_model(self, preprocessed_data, labels):
        X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, labels, test_size=0.2, random_state=42)

        input_shape = (X_train.shape[1], 1)
        model = self.create_cnn_model(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

        _, accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {accuracy}")

        self.model = model

    def run_inference(self, data):
        preprocessed_data = np.array(data).reshape(1, -1)
        prediction = self.model.predict(preprocessed_data)
        predicted_label = "Thinking" if prediction > 0.5 else "Not Thinking"
        print(f"Predicted: {predicted_label}")

    def run(self):
        data = []
        for i in range(self.cycles):
            not_thinking_data = self.collect_data("not thinking", self.not_thinking_duration)
            not_thinking_data = [{"data": epoch, "label": "not thinking"} for epoch in not_thinking_data]
            data.extend(not_thinking_data)

            thinking_data = self.collect_data("thinking", self.thinking_duration)
            thinking_data = [{"data": epoch, "label": "thinking"} for epoch in thinking_data]
            data.extend(thinking_data)

        preprocessed_data, labels = self.preprocess_data(data)
        self.train_model(preprocessed_data, labels)

        print("Starting real-time inference...")
        while True:
            inference_data = self.collect_data("inference", duration=1)
            self.run_inference(inference_data)

# main.py
from mkeds import MKEDS

def main():
    mkeds = MKEDS(cycles=3, not_thinking_duration=5, thinking_duration=5)
    mkeds.run()

if __name__ == "__main__":
    main()