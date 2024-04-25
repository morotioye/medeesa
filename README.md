# MKEDS: Multi-Kinesis Event Detection System

MKEDS is a modular and efficient system for detecting multiple kinesis events simultaneously using EEG data. It leverages the Neurosity SDK to collect raw EEG data, preprocess it, train a machine learning model, and perform real-time inference. The system is designed to be modular, maintainable, and extensible.

## Modules

### 1. DataCollectionModule

The `DataCollectionModule` handles the data collection process. It allows for dynamic cycles (e.g., 10 cycles, 5 not thinking, 5 thinking, 20 seconds each). It collects raw EEG data using the Neurosity SDK and saves it to CSV files.

### 2. DataPreprocessingModule

The `DataPreprocessingModule` is responsible for preprocessing the collected data. It applies the pre-filtering steps mentioned in the reference, such as notch filtering and bandpass filtering. If additional filtering is required, it can be implemented using the `rawUnfiltered` brainwaves parameter and the Neurosity Pipes library.

### 3. ModelTrainingModule

The `ModelTrainingModule` handles the training of the machine learning model. It splits the preprocessed data into training and testing datasets, trains the model using the training data, and evaluates the trained model using the testing data.

### 4. RealTimeInferenceModule

The `RealTimeInferenceModule` performs real-time inference using the trained model. It continuously collects real-time EEG data from the Neurosity device, preprocesses it, and feeds it into the trained model to make predictions or inferences. It triggers appropriate actions or provides feedback based on the model's predictions.

### 5. UIModule

The `UIModule` handles user interactions and displays feedback based on the model's predictions. It can also get user input or commands.

## Central Controller

The `CentralController` integrates all the modules and coordinates the flow of data and control between them. It initializes the modules, runs the data collection, data preprocessing, model training, and real-time inference processes, and facilitates user interaction.

## Code Organization

The code is organized into separate files for each module, making it modular and maintainable. The `CentralController` acts as the entry point of the program and orchestrates the entire process.

## Usage

To use MKEDS, follow these steps:

1. Set up the required environment variables in the `.env` file, including your Neurosity device ID, email, and password.

2. Run the `main.py` script to start the system

3. Follow the prompts displayed by the system to perform data collection, training, and real-time inference.

4. Interact with the system through the provided user interface to view predictions, trigger actions, or provide feedback.

## Requirements

- Python 3.x
- Neurosity SDK
- Neurosity Pipes library (optional, for additional filtering)
- Machine learning libraries (e.g., scikit-learn, TensorFlow, PyTorch)

## Contributing

Contributions to MKEDS are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository.

## License

MKEDS is open-source software licensed under the [MIT License](LICENSE).