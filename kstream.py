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
