# Lung-Tumor-detection-Computer_Vision

![Segmentation Demo](https://github.com/KhushaliP/Lung-Tumor-detection-Computer_Vision/blob/main/images/Segnet-collage-cropped.png?raw=true)

## Overview

In this project, we focus on the segmentation of tumor regions within lung CT images, aiming to enhance early detection and improve survival rates. Leveraging deep learning algorithms, our objective is to automate this critical aspect of medical image analysis.

## Objectives

### Early Detection for Improved Survival

Early tumor detection is paramount for increasing survival chances. The inherent complexities in precisely diagnosing tumors pose challenges for medical professionals. Image Segmentation emerges as a key solution to overcome these challenges, providing a more accurate and timely diagnosis.

### Automation for Treatment Precision

While segmented images aid in diagnoses and reduce time, manual analysis remains a bottleneck. Our goal is to automate the segmentation process, enabling efficient and precise treatment planning. Information about tumor area and volume guides treatment plans, ensuring the precise dosage of medication.

## Technology Stack

- **matplotlib, numpy:** Visualization and numerical operations.
- **tqdm:** Progress bar for task tracking.
- **pytorch-lightning, torch, torchvision, torchmetrics:** Core components for deep learning model development and training.
- **opencv-python:** Computer vision library for image processing.
- **nibabel:** Handling medical imaging data in NIfTI format.
- **imgaug:** Image augmentation for dataset diversification.
- **celluloid:** Creating animated plots for dynamic visualizations.
- **wandb:** Weights & Biases integration for experiment tracking and collaboration.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/KhushaliP/Lung-Tumor-detection-Computer_Vision.git
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Segmentation:**
   ```bash
   python segmentation_script.py
   ```

## Contributing

We welcome contributions and bug reporting. Please feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

---

