EEG Data Analysis Model Testing README

Overview:
This document outlines the testing strategy for evaluating three different machine learning models
on EEG data for real-time classification tasks. The models under test include Convolutional Neural Networks (CNNs),
Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) networks, and Support Vector Machines (SVM)
with real-time learning capabilities.

Models Under Test:
1. Convolutional Neural Networks (CNNs)
   - Pros: Effective at extracting spatial and temporal features from multidimensional data, automatic feature detection.
   - Cons: Computationally intensive, though mitigated by modern hardware and software optimizations.

2. Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks
   - Pros: Ideal for time-series data like EEG, capable of capturing temporal dependencies and changes over time.
   - Cons: Require more memory and computation time, suitable for complex temporal dynamics analysis.

3. Support Vector Machine (SVM) with Real-Time Learning
   - Pros: Suitable for real-time classification tasks, less computationally demanding than deep learning models.
   - Cons: Requires careful tuning of hyperparameters and kernel functions for efficient multiclass classification.

Testing Setup:
Each model will be tested using the following approach:
- Data Preprocessing: Apply standard preprocessing steps to EEG data, including normalization and noise filtering.
- Training: Train each model on a designated training dataset comprised of EEG recordings.
- Validation: Use a separate validation dataset to fine-tune model parameters and prevent overfitting.
- Testing: Evaluate the models on a withheld testing dataset to assess performance metrics such as accuracy, precision, recall, and F1-score.

Evaluation Metrics:
- Accuracy: Percentage of total correct predictions.
- Precision: Proportion of positive identifications that were actually correct.
- Recall: Proportion of actual positives that were identified correctly.
- F1-Score: Weighted average of precision and recall.

Results:
- (To be added after testing completion)
  - CNN Accuracy: 0.9852941176470589
  - LSTM Accuracy: 0.9705882352941176
  - SVM Accuracy: 0.9411764705882353

Additional Notes:
- Consider potential model optimizations based on initial test results.
- Explore the impact of different EEG channel configurations and sample rates on model performance.
