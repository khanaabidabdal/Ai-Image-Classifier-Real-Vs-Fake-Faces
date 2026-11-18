
# Ai Image Classifier : Fake Vs Real Faces

## Published Research

This project is related to the peer-reviewed research article:

**Khan Aabid Abdal Mashkoor (2025).** *Classifying Real and AI-Generated Images Using Fine-Tuned ResNet50.* **Journal of Advanced Artificial Intelligence, Engineering and Technology**, Volume 1, Issue 2 (Mar–Apr 2025), Article 07.

- **DOI:** https://doi.org/10.56147/aaiet.1.2.7
- **Official publication:** https://artificial-intelligence.engineering-technology.wren-research-journals.com/1/article/view/7
- **Published article PDF:** https://artificial-intelligence.engineering-technology.wren-research-journals.com/1/article/download/7/9/65

The published study evaluates feature extraction and fine-tuning of ResNet50 using the [140K Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). The paper reports 89.97% accuracy for feature extraction and 97.76%, 95.42%, and 96.85% for fine-tuned experiments at 5, 10, and 15 epochs respectively. The 5-epoch result varied on repetition, so the highest reported accuracy should not be interpreted as guaranteed performance on new images.

## Additional Synthetic-Face Images (Post-Research Extension)

Following the original research, approximately **150–200 additional AI-generated face images** were collected from [This Person Does Not Exist](https://thispersondoesnotexist.com/) and used them for additional model training to adapt the classifier to more synthetic-face examples as generative image technology evolves.

**Research distinction:** These images are a later project extension, **not part of the published 140K-image dataset or the published experimental results**. The additional images were used for retraining, not solely as a held-out evaluation set. Their exact count, the mix with the original training data, retraining configuration, sampling procedure, and independent evaluation metrics have not yet been documented here. Accordingly, no new accuracy or generalization claim is made for this sample.

This Person Does Not Exist is associated with StyleGAN-generated faces, so this additional collection alone does not establish performance on other generator families, such as diffusion models. Future work should evaluate independently sourced real and synthetic images from multiple generators, with a documented held-out test protocol.

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
The repository reports the following metrics for an experiment; the independent evaluation protocol after retraining on the additional 150–200 images has not yet been documented:

Accuracy: 96.85%

Precision (fake): 94.75%

Recall (fake): 94.16%

F1 Score: 94.45%

## About Files 

For developmentof this model, I wrote *'app.py'*, *'model.py'*, *'requirements.txt'*, and downlaoded the state_dict of the trained model and saved it as the RealityCheck. 

## Conclusion and Future Scope
**Conclusion**

Fine-tuning the ResNet50 model significantly enhances classification accuracy. The application model was subsequently retrained with approximately 150–200 additional synthetic-face images. Whether this improved generalization remains unverified without independent held-out evaluation. This research lays the groundwork for effective AI-generated image detection.

**Future Scope**

Data Augmentation: Implement advanced data augmentation techniques.

Larger Datasets: Train the model on larger, varied datasets.

Alternative Architectures: Explore other architectures like EfficientNet, DenseNet, or Vision Transformers.

Continual Learning: Develop mechanisms for continual learning.
Real-Time Application: Enhance the model for real-time classification.

Broader Application: Extend capabilities to detect other AI-generated content.

Ethical and Legal Considerations: Research ethical and legal implications of AI-generated content detection.

## References
Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).

Jovanović, Radiša. "Convolutional Neural Networks for Real and Fake Face Classification." Sinteza 2022.

He, Kaiming, et al. "Deep residual learning for image recognition." CVPR, 2016.

["140k Real and Fake Faces," Kaggle.](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

"70k Real Faces," Kaggle Deepfake Detection Challenge.

"1 Million Fake Faces on Kaggle," Kaggle Deepfake Detection Challenge.



