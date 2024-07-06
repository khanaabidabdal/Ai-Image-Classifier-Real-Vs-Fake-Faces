
# Ai Image Classifier : Fake Vs Real Faces

## Overview :

The ["Reality Check"](https://huggingface.co/spaces/khanaabidabdal/RealityCheck) application is designed to classify images of faces as either real or AI-generated. Utilizing the ResNet50 model, our classifier achieves up to 95% accuracy. This repository provides an overview of the application, detailed results from experiments, and future scope for enhancements.

## Introduction :

This research addresses the growing concern of the misuse of generative technology by developing models capable of detecting AI-generated images. Two primary techniques were employed: feature extraction and fine-tuning of the ResNet50 model. The experiments highlight the importance of model customization and iterative training to enhance the capabilities of deep learning models for specific tasks.

## User Interface :

The user interface (UI) of the "Reality Check" application is designed to be intuitive and user-friendly, allowing users to easily classify images of faces as either real or AI-generated.

## Important Guidelines :

Focus on Faces: Ensure the uploaded images clearly show faces.

Image Examples: Refer to examples to understand suitable images.

Avoid Other AI-Generated Content: Do not upload images of non-facial AI-generated content.

## Usage Instructions :
Prepare Your Image: Ensure the face is visible and prominent.

Upload the Image: Use the provided interface to upload your image for classification.

Receive Classification: The model will analyze the facial features and classify the image as either real or fake.

## Interface Components :

**Image Upload Section:**

1) Drop Image Here / Click to Upload: Drag and drop an image or click to open a file dialog.

2) Submit Button: Click to submit the image for classification.

**Classification Results:**

1) Predictions: Displays the classification result.

2) Prediction Time: Shows the time taken for the analysis.

3) Examples: Displays sample images suitable for classification.

## Real-World Data Integration
Incorporating real-world data improved model performance:

Accuracy: 96.85%

Precision (fake): 94.75%

Recall (fake): 94.16%

F1 Score: 94.45%

## About Files 

For developmentof this model, I wrote *'app.py'*, *'model.py'*, *'requirements.txt'*, and downlaoded the state_dict of the trained model and saved it as the RealityCheck. 

## Conclusion and Future Scope
**Conclusion**

Fine-tuning the ResNet50 model significantly enhances classification accuracy. Incorporating real-world data was crucial for maintaining model relevance. This research lays the groundwork for effective AI-generated image detection.

**Future Scope**

Data Augmentation: Implement advanced data augmentation techniques.

Larger Datasets: Train the model on larger, varied datasets.

Alternative Architectures: Explore other architectures like EfficientNet, DenseNet, or Vision Transformers.

Continual Learning: Develop mechanisms for continual learning.
Real-Time Application: Enhance the model for real-time classification.

Broader Application: Extend capabilities to detect other AI-generated content.

Ethical and Legal Considerations: Research ethical and legal implications of AI-generated content detection.

## Submission to Discover Artificial Intelligence
This research has been submitted to the journal Discover Artificial Intelligence through Springer Nature, highlighting its significance and contribution to the field of AI.

## References
Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).

Jovanović, Radiša. "Convolutional Neural Networks for Real and Fake Face Classification." Sinteza 2022.

He, Kaiming, et al. "Deep residual learning for image recognition." CVPR, 2016.

["140k Real and Fake Faces," Kaggle.](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

"70k Real Faces," Kaggle Deepfake Detection Challenge.

"1 Million Fake Faces on Kaggle," Kaggle Deepfake Detection Challenge.



