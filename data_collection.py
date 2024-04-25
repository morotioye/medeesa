# data_collection.py

import os
import time
from neurosity import NeurositySDK
from dotenv import load_dotenv

class DataCollectionModule:
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

    def collect_data(self, label, duration):
        print(f"Please {label} for {duration} seconds.")
        data = []

        def callback(epoch):
            data.append(epoch)

        self.neurosity.brainwaves_raw(callback)
        time.sleep(duration)
        self.neurosity.remove_all_subscriptions()

        timestamp = int(time.time())
        filename = f"{label.replace(' ', '_')}_{timestamp}.csv"

        with open(filename, "w") as file:
            file.write("channel,sample_1,sample_2,sample_3,sample_4,sample_5,sample_6,sample_7,sample_8,sample_9,sample_10,sample_11,sample_12,sample_13,sample_14,sample_15,sample_16\n")
            for epoch in data:
                for i, channel_data in enumerate(epoch["data"]):
                    file.write(f"{epoch['info']['channelNames'][i]},{','.join(map(str, channel_data))}\n")

        print(f"Data saved to {filename}\n")
        return filename

    def run(self):
        data_files = []
        for i in range(self.cycles):
            not_thinking_file = self.collect_data("not thinking", self.not_thinking_duration)
            data_files.append(not_thinking_file)
            thinking_file = self.collect_data("thinking", self.thinking_duration)
            data_files.append(thinking_file)
        return data_files