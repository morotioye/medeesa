# kstream.py
import os
import time
from neurosity import NeurositySDK
from dotenv import load_dotenv

class KStream:
    def __init__(self, device_id, email, password):
        self.neurosity = NeurositySDK({
            "device_id": device_id,
        })
        self.neurosity.login({
            "email": email,
            "password": password
        })

    def collect_data(self, output_file=None, duration=None, verbose=False):
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"data_{timestamp}.csv"

        with open(output_file, "w") as file:
            header = ",".join([f"{channel}_{sample}" for channel in ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4'] for sample in range(1, 17)])
            file.write(f"{header}\n")

            def callback(epoch):
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

# mkeds.py
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from neurosity import NeurositySDK
from dotenv import load_dotenv
from kstream import KStream

class MKEDS:
    def __init__(self, device_id, email, password):
        self.kstream = KStream(device_id, email, password)
        self.models = {}

    def train_thought(self, thought_name, data_collection_func, data_collection_args, epochs=10, batch_size=32):
        data = []
        labels = []

        def collect_and_label_data(label):
            data_collection_args["output_file"] = None
            data_collection_args["verbose"] = False
            collected_data = data_collection_func(**data_collection_args)
            collected_data = [{"data": epoch, "label": label} for epoch in collected_data]
            return collected_data

        # Collect data for the thought
        print(f"Collecting data for thought: {thought_name}")
        thought_data = collect_and_label_data(thought_name)
        data.extend(thought_data)
        labels.extend([thought_name] * len(thought_data))

        # Collect data for not thinking
        print(f"Collecting data for not thinking")
        not_thinking_data = collect_and_label_data("not_thinking")
        data.extend(not_thinking_data)
        labels.extend(["not_thinking"] * len(not_thinking_data))

        preprocessed_data, labels = self.preprocess_data(data, labels)
        model = self.train_model(preprocessed_data, labels, epochs, batch_size)
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

    def train_model(self, preprocessed_data, labels, epochs=10, batch_size=32, validation_split=0.2, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, labels, test_size=validation_split, random_state=42)

        input_shape = (X_train.shape[1], 1)
        model = self.create_cnn_model(input_shape)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

        _, accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {accuracy}")

        return model

    def save_model(self, model, thought_name):
        model_path = f"models/{thought_name}.h5"
        model.save(model_path)
        self.models[thought_name] = model_path

    def load_model(self, thought_name):
        model_path = self.models.get(thought_name)
        if model_path:
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            raise ValueError(f"Model for thought '{thought_name}' not found.")

    def run_inference(self, thought_names, data_collection_func, data_collection_args, verbose=False):
        models = {}
        for thought_name in thought_names:
            model = self.load_model(thought_name)
            models[thought_name] = model

        print("Starting real-time inference...")
        while True:
            data_collection_args["output_file"] = None
            data_collection_args["verbose"] = verbose
            inference_data = data_collection_func(**data_collection_args)
            preprocessed_data = np.array(inference_data).reshape(1, -1)
            
            predictions = {}
            for thought_name, model in models.items():
                prediction = model.predict(preprocessed_data)
                predictions[thought_name] = prediction[0][0]

            predicted_thought = max(predictions, key=predictions.get)
            print(f"Predicted thought: {predicted_thought}")