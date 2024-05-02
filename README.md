MKEDS: Multi-Kinesis Event Detection System
MKEDS (Multi-Kinesis Event Detection System) is a Python library that provides a framework for collecting EEG data, training machine learning models to detect specific thought patterns, and performing real-time inference on EEG data streams. It is built on top of the Neurosity SDK and utilizes TensorFlow and scikit-learn for model training and evaluation.
Features

Collect EEG data in non-segmented and segmented formats
Train machine learning models to detect specific thought patterns
Perform real-time inference on EEG data streams
Customizable model architecture and training parameters
Save and load trained models for later use
Write inference results to a CSV file for further analysis

Installation

Clone the MKEDS repository:
Copy codegit clone https://github.com/your-username/mkeds.git

Install the required dependencies:
Copy codepip install neurosity tensorflow scikit-learn python-dotenv

Set up the Neurosity SDK credentials by creating a .env file in the project directory with the following contents:
Copy codeNEUROSITY_DEVICE_ID=your_device_id
NEUROSITY_EMAIL=your_email
NEUROSITY_PASSWORD=your_password
Replace your_device_id, your_email, and your_password with your actual Neurosity SDK credentials.

Usage
Collecting EEG Data
To collect EEG data, you can use the run_non_segmented or run_segmented methods of the MKEDS class.
pythonCopy codefrom mkeds import MKEDS

# Instantiate the MKEDS class
mkeds = MKEDS(device_id, email, password)

# Collect non-segmented data
mkeds.run_non_segmented(output_file="non_segmented_data.csv", duration=3, verbose=True)

# Collect segmented data
mkeds.run_segmented(num_cycles=2, cycle_duration=2, output_prefix="segment", verbose=True)
Training a Thought Model
To train a thought model, use the train_thought method of the MKEDS class.
pythonCopy codefrom mkeds import MKEDS

# Instantiate the MKEDS class
mkeds = MKEDS(device_id, email, password)

# Train a thought using segmented data
mkeds.train_thought(thought_name="focus", num_cycles=3, cycle_duration=5, verbose=True)
Performing Real-Time Inference
To perform real-time inference on an EEG data stream, use the run_inference method of the MKEDS class.
pythonCopy codefrom mkeds import MKEDS

# Instantiate the MKEDS class for inference with a pre-trained model
mkeds_inference = MKEDS(device_id, email, password, model_path="models/focus.h5")

# Perform real-time inference
mkeds_inference.run_inference(thought_names=["focus"], duration=5, threshold=0.0)
How It Works
MKEDS consists of several components that work together to enable thought detection and real-time inference:

Data Collection: The collect_data method of the MKEDS class is responsible for collecting EEG data from the Neurosity device. It saves the collected data to a CSV file and returns the data as a list of dictionaries.
Data Preprocessing: The preprocess_data method flattens the collected EEG data and converts it into a suitable format for training the machine learning model. It also encodes the thought labels using scikit-learn's LabelEncoder.
Model Architecture: The create_cnn_model method defines the architecture of the convolutional neural network (CNN) used for thought detection. It consists of multiple convolutional layers, pooling layers, and dense layers.
Model Training: The train_model method trains the CNN model using the preprocessed EEG data and thought labels. It splits the data into training and validation sets, compiles the model, and fits it to the training data.
Model Saving and Loading: The save_model and load_model methods allow saving the trained model to a file and loading it later for inference.
Real-Time Inference: The run_inference method performs real-time inference on an EEG data stream. It loads the pre-trained model, continuously processes the incoming EEG data, and predicts the thought associated with each data point. The predicted thoughts and their confidences are printed to the console and written to a CSV file.

Conclusion
MKEDS provides a powerful and flexible framework for collecting EEG data, training thought detection models, and performing real-time inference on EEG data streams. With its intuitive API and customizable parameters, MKEDS enables developers and researchers to easily integrate thought detection capabilities into their applications and experiments.
For more detailed information on the available methods and their parameters, please refer to the source code and inline documentation.
Happy thought detection with MKEDS!