# Reality Check — Real vs. AI-Generated Face Classifier

A ResNet50-based face-image classifier with a Gradio interface. **Reality Check** predicts whether an uploaded face image is real or AI-generated and displays class probabilities and inference time.

**[Try the application on Hugging Face](https://huggingface.co/spaces/khanaabidabdal/RealityCheck)** · **[Read the published research](https://doi.org/10.56147/aaiet.1.2.7)**

> **Scope:** This project focuses on face images. A prediction is not proof that an image is real or synthetic, and performance can vary across image sources and generators.

## Relationship to the published research

This application builds on the research paper:

**Khan Aabid Abdal Mashkoor (2025).** *Classifying Real and AI-Generated Images Using Fine-Tuned ResNet50.* **Journal of Advanced Artificial Intelligence, Engineering and Technology**, Volume 1, Issue 2 (Mar–Apr 2025).

- **DOI:** https://doi.org/10.56147/aaiet.1.2.7
- **Journal article:** https://artificial-intelligence.engineering-technology.wren-research-journals.com/1/article/view/7
- **Published PDF:** https://artificial-intelligence.engineering-technology.wren-research-journals.com/1/article/download/7/9/65
- **Original dataset:** [140K Real and Fake Faces (Kaggle)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

The **published research experiments** and the **later application retraining** are distinct. The additional images described below were not part of the original paper's reported experiments.

## Application update: retraining with additional synthetic faces

After the original model was developed, I noticed that it struggled with some AI-generated face images encountered later. One possible explanation is that the training data did not adequately represent the images being encountered; advances in generative image technology can also contribute to such differences, although the specific cause was not isolated.

To improve the application:

1. I collected **approximately 150–200 additional AI-generated face images** from [This Person Does Not Exist](https://thispersondoesnotexist.com/).
2. I **combined those images with the original dataset**, using data accessed through Google Drive.
3. I **retrained the ResNet50-based classifier using transfer learning** on the combined dataset.
4. After retraining, I tested the application on **different, previously unseen images** obtained from the same website. **These test images were not the images added to the training dataset.**
5. I observed that the updated application classified those new examples more successfully than before.

This is a **post-publication application improvement**, not a revision of the published paper's experimental results.

### Interpretation and limitations

The tests on different images provide useful evidence of improved behavior on previously unseen samples from that source. However, the exact number of test images, before/after scores, and a formal evaluation protocol have not yet been documented, so **no numerical improvement or new accuracy figure is claimed for this retraining step**.

Images from the same website may share generator-specific characteristics. Success on unseen images from that source does not establish generalization to every modern image generator, including diffusion-based systems. A future benchmark should use documented, independent test sets from multiple sources and generator families.

## Published experimental results

The following metrics are **from the original paper**, not from the later application retraining:

| ResNet50 approach | Accuracy |
| --- | ---: |
| Feature extraction | 89.97% |
| Fine-tuning — 5 epochs | 97.76% |
| Fine-tuning — 10 epochs | 95.42% |
| Fine-tuning — 15 epochs | 96.85% |

The paper notes that the highest 5-epoch result was not consistently reproduced. These values should not be interpreted as guaranteed accuracy on arbitrary new images.

## Application features

- Upload a face image through a Gradio interface.
- View predictions for **Real** and **Fake** classes.
- See prediction probabilities and inference time.
- Try the example images provided with the application.

The classifier is intended for face images, not general AI-generated artwork or other image categories.

## Repository files

| Path | Description |
| --- | --- |
| `Real_Or_Fake.ipynb` | Development and experimentation notebook |
| `RealityCheck.pth` | Saved PyTorch model weights |
| `model.py` | ResNet50 model definition |
| `app.py` | Gradio prediction interface |
| `requirements.txt` | Python dependencies |
| `eaxmples/` | Example images (existing directory name) |

**Implementation note:** The repository currently contains issues that may prevent a fresh local run: `model.py` refers to an undefined `output_shape`, while `app.py` expects an `examples/` directory instead of the existing `eaxmples/`. These are existing code issues; this documentation update does not modify or verify the running application. The saved model file is also large and may require special handling when cloning or redistributing.

## Future work

- Reproduce and document the retraining procedure, data splits, and exact additional-image count.
- Benchmark the original and updated models on the same held-out test sets.
- Evaluate images from different generators and sources, including diffusion models.
- Improve the application setup and make model loading reproducible.
- Investigate augmentation, alternative architectures, and robustness to image processing.

## References

- [Published research article and DOI](https://doi.org/10.56147/aaiet.1.2.7)
- [140K Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- [This Person Does Not Exist](https://thispersondoesnotexist.com/)
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. NeurIPS.
