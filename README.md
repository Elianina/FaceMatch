# COSC595 Implementation: A Comparison of Four Deep Learning Architectures for Gendered Facial Recognition   

---
 
**Course:** COSC595 Information Technology Project: Implementation 

**Team:** Carl Fokstuen, YuTing Lee, Mark Malady, Nayani Samaranayake, Vishal Cheroor Ravi  

**Supervisors:** Prof. Raymond Chiong, Dr. Farshid Hajati   

**University of New England (UNE)** 

---

## Project Overview

This document constitutes the project implementation's technical handbook.   

This project implements and evaluates four deep learning architectures for the binary gender classification problem from two-dimensional facial images.   

The models are compared for accuracy, efficiency, and are as follows:

1. **VGG-16**
2. **EfficientNet-B4**
3. **ResNet-50**
4. **DenseNet-121**

All models are trained on the **CelebFaces Attributes (CelebA) Dataset**, which contains 202,599 celebrity face images from 10,177 unique 
identities with 40 binary attributes including gender labels.

### CelebA Dataset

If you do not have the CelebA dataset locally:
- Download the dataset from Kaggle: 
  - https://www.kaggle.com/datasets/jessicali9530/celeba-dataset, or
- Prepare Kaggle's API credentials: 
  - https://www.kaggle.com/docs/api

---

## Prerequisites

- **Anaconda or Miniconda** installed
- **CUDA-capable GPU** NVIDIA RTX 3080 Ti used in training for this project
- **16GB+ RAM** for training
- **~10GB/+ storage** for dataset and model checkpoints


### CelebA Dataset

If you don't have the CelebA dataset locally:
- Download from Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
- Prepare Kaggle API credentials: https://www.kaggle.com/docs/api

---

## Dependencies

### Core Requirements
- Python 3.1x
- PyTorch 2.x
- CUDA 1x

### Required Packages
```txt
- pytorch
- conda-forge
- defaults
- python=3.11
- pytorch
- torchvision
- pytorch-cuda=12.1
- numpy
- pandas
- scikit-learn
- pillow
- matplotlib
- seaborn
- tqdm
- jupyter
- ipykernel
- kaggle
- pip
- pip:
  - torchinfo
````

---

## Installation

### 1.Clone the repository and change directory to the project's root.
```bash
git clone https://github.com/Elianina/FaceMatch.git
cd COSC595_Implementation
```

### 2.Create the conda environment from the environment.yml file
```bash
conda env create -f environment.yml  
```

### 3.Activate the environment
```bash
conda activate cosc595
```

---

## Running the Models
### VGG-16
```bash
cd FaceMatch/Models/VGG-16             # Change directory to the model's root
python main.py                         # Run the VGG-16 model
`````

### EfficientNet-B4
```bash
cd FaceMatch/Models/EfficientNet-B4    # Change directory to the model's root
python main.py                         # Run the DenseNet-121 model
```

### ResNet-50
```bash
cd FaceMatch/Models/ResNet-50          # Change directory to the model's root
python main.py                         # Run the DenseNet-121 model
```
#### DenseNet-121
```bash
cd FaceMatch/Models/DenseNet-121       # Change directory to the model's root
python main.py                         # Run the DenseNet-121 model
````

---

## Training Configuration

Each model's configuration con be found in the model's `main.py` file.

#### Example:

```python
CONFIG = {
    # Paths to CelebA dataset files and directories if using local dataset
    'img_dir': r"",                                                # The path to the CelebA image directory
    'attr_file': r"",                                              # The path to the CelebA attribute file as .csv

    # Model configurations 
    'batch_size': 0,                                               # Model batch size
    'num_epochs': 0,                                               # Number of training epochs
    'num_workers': 0,                                              # Parallel data loading processes
    'train_ratio': 0,                                              # % of data used for training
    'val_ratio': 0,                                                # % for validation, remaining % aside for testing

    'classifier_dropout': 0,                                       # Dropout rate in the final classification layer
    'drop_connect_rate': 0,                                        # Drop connect rate for regularisation
    'num_classes': 0,                                              # Binary classification: Male (1) vs Female (0)

    # Output paths
    'save_path': '../../Results/models/..pth',                     # Best model epoch
    'log_dir': '../../Results/logs/',                              # Training logs
    'plots_dir': '../../Results/plots/',                           # Plots
}
```

---

## Project Structure
Below is the COSC595 Implementation project structure. 

```
COSC595_Implementation/
│
├── FaceMatch/                                   # The main project directory
│   │
│   ├── Models/                                  # Neural network model implementations
│   │   │
│   │   ├── DenseNet-121/                        # DenseNet-121 model directory
│   │   │   ├── main.py                              # Entry point for training/testing DenseNet-121
│   │   │   ├── model.py                             # DenseNet-121 architecture definition
│   │   │   └── trainer.py                           # Training and evaluation logic for DenseNet-121
│   │   │
│   │   ├── EfficientNet-B4/                     # EfficientNet-B4 model directory
│   │   │   ├── main.py                              # Entry point for training/testing EfficientNet-B4
│   │   │   ├── model.py                             # EfficientNet-B4 architecture definition
│   │   │   └── trainer.py                           # Training and evaluation logic for EfficientNet-B4
│   │   │
│   │   ├── ResNet-50/                           # ResNet-50 model directory
│   │   │   ├── main.py                              # Entry point for training/testing ResNet-50
│   │   │   ├── model.py                             # ResNet-50 architecture definition
│   │   │   └── trainer.py                           # Training and evaluation logic for ResNet-50
│   │   │
│   │   └── VGG-16/                              # VGG-16 model directory
│   │       ├── main.py                              # Entry point for training/testing VGG-16
│   │       ├── model.py                             # VGG-16 architecture definition
│   │       └── trainer.py                           # Training and evaluation logic for VGG-16
│   │
│   ├── Results/                                 # All training outputs and evaluation results
│   │   │
│   │   ├── comparison/                          # Cross-model comparison results
│   │   │   ├── model_comparison_summary.csv         # Tabular comparison of all model metrics
│   │   │   ├── model_comparison_report.txt          # Detailed text report comparing all models
│   │   │   ├── comparison_metrics.png               # Bar chart comparing key metrics across models
│   │   │   ├── comparison_roc_curves.png            # ROC curves for all models on single plot
│   │   │   └── comparison_training_curves.png       # Training/validation curves for all models
│   │   │
│   │   ├── logs/                                # Training logs for each model
│   │   │   ├── DenseNet-121_training_log.txt        # Complete training log for DenseNet-121
│   │   │   ├── EfficientNet-B4_training_log.txt     # Complete training log for EfficientNet-B4
│   │   │   ├── ResNet-50_training_log.txt           # Complete training log for ResNet-50
│   │   │   └── VGG-16_training_log.txt              # Complete training log for VGG-16
│   │   │
│   │   ├── misclassified/                       # Misclassified image sample analysis
│   │   │   └── model_YYYYMMDD_HHMMSS.csv            # model errors - table of misclassified images
│   │   │
│   │   ├── models/                              # Saved model checkpoints
│   │   │   ├── DenseNet-121_best.pth                # Best DenseNet-121 model weights
│   │   │   ├── EfficientNet-B4_best.pth             # Best EfficientNet-B4 model weights
│   │   │   ├── ResNet-50_best.pth                   # Best ResNet-50 model weights
│   │   │   └── VGG-16_best.pth                      # Best VGG-16 model weights
│   │   │
│   │   └── plots/                               # Visualisation outputs per model
│   │       ├── model_confusion_matrix.png           # model confusion matrix
│   │       ├── model_metrics_summary.png            # model metrics visualization
│   │       ├── model_roc_curve.png                  # model ROC curve
│   │       └── model_training_curves.png            # model training history
│   │
│   └── utils/                                   # Shared utility modules
│       ├── __init__.py                              # Package initialisation
│       ├── celeba_api.py                            # CelebA dataset API and downloading utilities
│       ├── data_loader.py                           # Dataset loading and preprocessing
│       ├── image_eval_utils.py                      # Image evaluation and visualisation utilities
│       ├── log_utils.py                             # Logging and console output utilities
│       ├── metrics_utils.py                         # Metrics calculation and reporting
│       └── model_comparison_utils.py                # Cross-model comparison and benchmarking
│
├── .gitignore                                   # Git ignore rules
├── __init__.py                                  # Package initialisation for the COSC595_Implementation
└── environment.yml                              # Conda environment specification
```

### Directory Descriptions

- **`Models/`**: Contains the implementation of four CNN architectures (VGG-16, EfficientNet-B4, ResNet-50, DenseNet-121)
  - Each model has its own subdirectory with `main.py`, `model.py`, and `trainer.py`
- **`Results/`**: Stores the model's training outputs and their evaluation metrics
  - `comparison/`: The model comparison results formatted as a CSV file and a text report 
  - `logs/`: Training logs for each model across training epochs
  - `misclassified/`: CSV files of any misclassified samples
  - `models/`: Saved models (.pth files)
  - `plots/`: Plotting outputs (confusion matrices, ROC curves, training curves, metrics summaries)

- **`utils/`**: Shared utility modules
  - Data loading, preprocessing, metrics calculation, logging, and model comparison tools

- **Root files**: Configuration files (.gitignore, environment.yml) and documentation (README.md)

---

## References

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.   
    In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*   
    (pp. 770-778). IEEE. https://doi.org/10.1109/CVPR.2016.90  

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected   
    convolutional networks. In *Proceedings of the IEEE Conference on Computer Vision   
    and Pattern Recognition* (pp. 4700-4708). IEEE. https://doi.org/10.1109/CVPR.2017.243  

Li, J. (2018). *CelebFaces Attributes (CelebA) Dataset* [Data set]. Kaggle.   
    https://www.kaggle.com/datasets/jessicali9530/celeba-dataset   

Liu, Z. (n.d.). *DenseNet*. GitHub. https://github.com/liuzhuang13/DenseNet   

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z.,   
    Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M.,   
    Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019).   
    PyTorch: An imperative style, high-performance deep learning library. In *Advances in   
    Neural Information Processing Systems* 32, 8024-8035.   
    https://papers.nips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html  

PyTorch. (n.d.). *Source code for torchvision.models.densenet*. PyTorch Documentation.   
    https://pytorch.org/vision/0.8/_modules/torchvision/models/densenet.html   

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A.,   
    Khosla, A., Bernstein, M., Berg, A. C., & Fei-Fei, L. (2015). ImageNet large scale visual   
    recognition challenge. *International Journal of Computer Vision*, *115*(3), 211-252.   
    https://doi.org/10.1007/s11263-015-0816-y   

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale   
    image recognition. In *Proceedings of the International Conference on Learning   
    Representations* (ICLR). https://arxiv.org/abs/1409.1556  

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural   
    networks. In *Proceedings of the 36th International Conference on Machine Learning*   
    (pp. 6105-6114). PMLR. http://proceedings.mlr.press/v97/tan19a.html   

---

## Datasets

**CelebA (CelebFaces Attributes Dataset)**   
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
- **Description:** 202,599 celebrity face images from 10,177 individuals with 40 binary    
  attribute annotations including gender, age, and facial features
- **Citation:** Li, J. (2018)

---

## Ethical Considerations and Potential Harms

**Important Notice:** This project implements gender classification technology that has   
the potential for societal harm if misused. Users should be aware of the following   
considerations:

### Potential Risks:
- **Privacy Concerns**:    
    The four models can possibly infer gendered demographic information from two-dimensional facial images.  
- **Misgendering**:
    The binary classification (male/female) used in this work does not account for non-binary or gender-diverse individuals  
- **Surveillance Misuse**:    
    It has been identified and acknowledged by the authors that this technology has the potential to be deployed in contexts that can violate privacy rights or enable discriminatory practices  
- **Reinforcement of Binary Gender Norms**:   
    The model architecture inherently assumes gender is binary, which may not reflect individual lived experiences  

### Responsible Use Guidelines:
It is recommended that this system should only be used:   
- Within the boundaries of ethical research practices and institutional review   
- With explicit informed consent from individuals whose images are to be analysed   
- In contexts where demographic classification serves a legitimate, and beneficial purpose   
- With awareness of limitations and potential for gendered misclassification   

### Purpose:
This implementation is designed for educational and research purposes.   

**NOTE: The authors strongly and explicitly discourage the deployment of these models in real-world    
applications without comprehensive ethical review, privacy safeguards, and the consideration    
of any potential negative impacts on the targeted individual/s, prior to deployment.**   

---

## Licence
MIT License

Copyright © 2025 Carl Fokstuen, Mark Malady, & YuTing Lee.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished 
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**Note:** This licence was taken verbatim from the Open Source Initiative article titled:   
The MIT Licence (https://opensource.org/license/mit).

---

## Contact
For questions or issues, contact the project team through the GitHub repository, or the team's supervisory staff, care of
The University of New England (UNE).
