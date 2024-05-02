# mkeds.py
import os
import time
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from neurosity import NeurositySDK

class MKEDS:
    def __init__(self, device_id, email, password, model_path="models"):
        self.neurosity = NeurositySDK({
            "device_id": device_id,
        })
        self.neurosity.login({
            "email": email,
            "password": password
        })
        self.model_path = model_path
        if os.path.isdir(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)

    def collect_data(self, output_file=None, duration=None, verbose=False):
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"data_{timestamp}.csv"

        collected_data = []

        with open(output_file, "w") as file:
            header = ",".join([f"{channel}_{sample}" for channel in ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4'] for sample in range(1, 17)])
            file.write(f"{header}\n")

            def callback(epoch):
                collected_data.append(epoch)
                row = ",".join([str(sample) for channel_data in epoch["data"] for sample in channel_data])
                file.write(f"{row}\n")
                if verbose:
                    print(epoch)  # Print the epoch data if verbose is True

            self.neurosity.brainwaves_raw(callback)
            start_time = time.time()

            while True:
                if duration is not None and time.time() - start_time >= duration:
                    break
                time.sleep(0.1)  # Sleep for a short interval to avoid busy-waiting

            self.neurosity.remove_all_subscriptions()
            print(f"Data saved to {output_file}")

        return collected_data

    def run_non_segmented(self, output_file=None, duration=None, verbose=False):
        if duration is None:
            print("Starting non-segmented data collection. Press Ctrl+C to stop...")
        else:
            print(f"Starting non-segmented data collection for {duration} seconds...")
        self.collect_data(output_file, duration, verbose)

    def run_segmented(self, num_cycles, cycle_duration, output_prefix="cycle", verbose=False):
        if not isinstance(cycle_duration, (int, float)) or cycle_duration <= 0:
            raise ValueError("Invalid cycle duration. Please provide a positive number of seconds.")
        print(f"Starting segmented data collection for {num_cycles} cycles of {cycle_duration} seconds each...")
        for i in range(num_cycles):
            cycle_label = f"{output_prefix}_{i + 1}"
            timestamp = int(time.time())
            output_file = f"{cycle_label}_{timestamp}.csv"
            print(f"Cycle {i + 1}: {cycle_label}")
            self.collect_data(output_file, cycle_duration, verbose)

    def train_thought(self, thought_name, num_cycles=3, cycle_duration=5, verbose=True):
        data = []
        labels = []

        for i in range(num_cycles):
            print(f"Cycle {i + 1}/{num_cycles}")

            # Rest phase
            print("Rest phase")
            rest_data = self.collect_data(duration=cycle_duration, verbose=verbose)
            data.extend(rest_data)
            labels.extend(["rest"] * len(rest_data))

            # Thought phase
            print(f"Thought phase: {thought_name}")
            thought_data = self.collect_data(duration=cycle_duration, verbose=verbose)
            data.extend(thought_data)
            labels.extend([thought_name] * len(thought_data))

        preprocessed_data, labels = self.preprocess_data(data, labels)

        # Encode labels to integers
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        model = self.train_model(preprocessed_data, labels, verbose=verbose)
        self.save_model(model, thought_name)
        print(f"Thought '{thought_name}' trained and saved.")

    def preprocess_data(self, data, labels):
        preprocessed_data = []

        for epoch in data:
            flattened_data = []
            for channel_data in epoch["data"]:
                flattened_data.extend(channel_data)
            preprocessed_data.append(flattened_data)

        preprocessed_data = np.array(preprocessed_data)
        labels = np.array(labels)

        return preprocessed_data, labels

    def create_cnn_model(self, input_shape, num_filters=(64, 64, 128, 128), kernel_size=3, pool_size=2, dense_units=64):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(num_filters[0], kernel_size, activation='relu', input_shape=input_shape),
            tf.keras.layers.Conv1D(num_filters[1], kernel_size, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size),
            tf.keras.layers.Conv1D(num_filters[2], kernel_size, activation='relu'),
            tf.keras.layers.Conv1D(num_filters[3], kernel_size, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train_model(self, preprocessed_data, labels, validation_split=0.2, epochs=10, batch_size=32, verbose=True):
        X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, labels, test_size=validation_split, random_state=42)

        input_shape = (X_train.shape[1], 1)
        model = self.create_cnn_model(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        if verbose:
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
        else:
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)

        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {accuracy}")

        return model

    def save_model(self, model, thought_name):
        model_path = os.path.join(self.model_path, f"{thought_name}.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, thought_name):
        if os.path.isfile(self.model_path):
            model_path = self.model_path
        else:
            model_path = os.path.join(self.model_path, f"{thought_name}.h5")

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            raise ValueError(f"Model for thought '{thought_name}' not found.")

    def run_inference(self, thought_names=None, duration=None, threshold=0.5):
        if thought_names is None:
            thought_names = list(self.models.keys())

        models = {}
        for thought_name in thought_names:
            model = self.load_model(thought_name)
            models[thought_name] = model

        output_file = "inference_results.csv"
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Predicted Thought", "Confidence"])

            print("Starting real-time inference...")
            start_time = time.time()

            def callback(epoch):
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                preprocessed_data = np.array([epoch["data"]]).reshape(1, -1)

                predictions = {}
                for thought_name, model in models.items():
                    prediction = model.predict(preprocessed_data)[0][0]
                    if prediction > threshold:
                        predictions[thought_name] = prediction

                if predictions:
                    predicted_thought = max(predictions, key=predictions.get)
                    confidence = predictions[predicted_thought]
                    print(f"Timestamp: {timestamp} | Predicted Thought: {predicted_thought} | Confidence: {confidence:.2f}")
                    writer.writerow([timestamp, predicted_thought, confidence])
                else:
                    print(f"Timestamp: {timestamp} | No thought detected above the threshold.")
                    writer.writerow([timestamp, "No thought detected", ""])

            self.neurosity.brainwaves_raw(callback)

            while True:
                if duration is not None and time.time() - start_time >= duration:
                    break
                time.sleep(0.1)

            self.neurosity.remove_all_subscriptions()
            print(f"Real-time inference completed. Results saved to {output_file}.")