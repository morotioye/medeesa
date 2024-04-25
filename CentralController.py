# central_controller.py

from data_collection import DataCollectionModule
from data_preprocessing import DataPreprocessingModule
from model_training import ModelTrainingModule
from real_time_inference import RealTimeInferenceModule
from ui_module import UIModule

class CentralController:
    def __init__(self):
        self.data_collection = DataCollectionModule(cycles=10, not_thinking_duration=20, thinking_duration=20)
        self.data_preprocessing = DataPreprocessingModule()
        self.model_training = ModelTrainingModule()
        self.real_time_inference = None
        self.ui = UIModule()

    def run(self):
        # Data Collection
        data_files = self.data_collection.run()

        # Data Preprocessing
        preprocessed_data = self.data_preprocessing.preprocess_data(data_files)

        # Model Training
        trained_model = self.model_training.train_model(preprocessed_data)

        # Real-time Inference
        self.real_time_inference = RealTimeInferenceModule(trained_model)
        self.real_time_inference.run_inference()

        # User Interaction
        while True:
            user_input = self.ui.get_user_input()
            if user_input == "quit":
                break
            # Process user input and perform corresponding actions
            # ...

if __name__ == "__main__":
    controller = CentralController()
    controller.run()