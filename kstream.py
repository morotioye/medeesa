import os
import time
from neurosity import NeurositySDK
from dotenv import load_dotenv

class KStream:
    def __init__(self):
        load_dotenv()
        self.neurosity = NeurositySDK({
            "device_id": os.getenv("NEUROSITY_DEVICE_ID"),
        })
        self.neurosity.login({
            "email": os.getenv("NEUROSITY_EMAIL"),
            "password": os.getenv("NEUROSITY_PASSWORD")
        })

    def collect_data(self, duration, output_file=None):
        data = []
        
        def callback(epoch):
            data.append(epoch)
            print(epoch)  # Print the epoch data
        
        self.neurosity.brainwaves_raw(callback)
        time.sleep(duration)
        self.neurosity.remove_all_subscriptions()
        
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"data_{timestamp}.csv"
        
        with open(output_file, "w") as file:
            header = ",".join([f"{channel}_{sample}" for channel in data[0]['info']['channelNames'] for sample in range(1, 17)])
            file.write(f"{header}\n")
            
            for epoch in data:
                row = ",".join([str(sample) for channel_data in epoch["data"] for sample in channel_data])
                file.write(f"{row}\n")
        
        print(f"Data saved to {output_file}")

    def run_non_segmented(self, duration, output_file=None):
        print(f"Starting non-segmented data collection for {duration} seconds...")
        self.collect_data(duration, output_file)

    def run_segmented(self, num_cycles, cycle_duration, thinking_label="thinking", not_thinking_label="not_thinking"):
        print(f"Starting segmented data collection for {num_cycles} cycles of {cycle_duration} seconds each...")
        
        for i in range(num_cycles):
            print(f"Cycle {i + 1}: Thinking")
            thinking_file = f"thinking_cycle_{i + 1}_{int(time.time())}.csv"
            self.collect_data(cycle_duration, thinking_file)
            
            print(f"Cycle {i + 1}: Not Thinking")
            not_thinking_file = f"not_thinking_cycle_{i + 1}_{int(time.time())}.csv"
            self.collect_data(cycle_duration, not_thinking_file)

# Usage examples
kstream = KStream()

# Non-segmented data collection
kstream.run_non_segmented(duration=10)

# Segmented data collection
kstream.run_segmented(num_cycles=3, cycle_duration=5)