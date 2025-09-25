# COSC595 Implementation: A Comparison of Four Deep Learning Architectures for Gendered Facial Recognition   

---

**University of New England (UNE)**   
**Course:** COSC595 Information Technology Project: Implementation  
**Team:** Carl Fokstuen, YuTing Lee, Mark Malady, Nayani Samaranayake, Vishal Cheroor Ravi  
**Supervisors:** Prof. Raymond Chiong, Dr. Farshid Hajati
---
## Project Overview
This project implements and evaluates multiple deep learning architectures for gendered facial recognition for comparative purposes. 
These models include 1. VGG-16, 2. EfficientNet-B4, 3. ResNet-50, and 4. DenseNet-121. The models are trained on the CelebA dataset, 
which contains 202,599 celebrity face images from 10,177 individuals. The dataset contains 40 attributes, including 5 facial features, 
age, gender, and race. The attributes are used to train the models to predict the gender of a celebrity from a given image.

--- 

## Project Installation

### Prerequisites
- Anaconda or Miniconda installed
- CUDA-capable GPU (recommended)
- 16GB+ RAM for training
- if you do not have celebA dataset available locally, you can download it from Kaggle:
  - Download the CelebA dataset from Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
  - Prepare Kaggle API credentials, see: https://www.kaggle.com/docs/api



### Dependencies
  - python
  - pytorch-cuda
  - pytorch
  - torchvision
  - torchaudio
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - tqdm
  - pillow
  - jupyter
  - ipykernel
  - Kaggle API
  - matplotlib

### Environment Setup
1.Clone the repository and change directory into the project's root.
```bash
git clone https://github.com/Elianina/FaceMatch.git
cd COSC595_Implementation
```

2.Create the conda environment from environment.yml file
```bash
conda env create -f environment.yml  
```

3.Activate the environment
```bash
conda activate cosc595
```

### Alternatively, follow the manual environment setup
```bash
conda create -n cosc595 python=3.9
conda activate cosc595
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas numpy scikit-learn matplotlib tqdm pillow jupyter -c conda-forge
````

### Running Models
#### VGG-16
```bash
cd Models/VGG16   # Change directory to the model's root
python main.py       # Run the VGG-16 model
`````
#### EfficientNet-B4
```bash
cd Models/EfficientNet-B4   # Change directory to the model's root
python main.py           # Run the DenseNet-121 model
```
#### ResNet-50
```bash
cd Models/ResNet-50   # Change directory to the model's root
python main.py           # Run the DenseNet-121 model
```
#### DenseNet-121
```bash
cd Models/DenseNet-121   # Change directory to the model's root
python main.py           # Run the DenseNet-121 model
````

---

## Project structure
Below is the COSC595 Implementation project structure draft as it relates to Assessment 2. 
```
COSC595_Implementation/
├── Datasets/  # Use CelebA dataset api script from utils/celeba_api.py               
└── Docs/
│   └── technical_handbook.md   # Technical handbook (Assessment 2 copy for completeness)
├── Models/
│   ├── vgg16/
│   │   ├──README.md           # VGG-16 specific documentation
│   │   ├── main.py            # Model entry point
│   │   ├── model.py           # VGG-16 architecture
│   │   ├── data_loader.py     # Dataset handling
│   │    ── trainer.py         # Training/evaluation logic
│   ├── EfficientNet-B4/
│   │   ├── README.md          # EfficientNet-B4 specific documentation
│   │   ├── main.py            # Model entry point
│   │   ├── model.py           # EfficientNet-B4 architecture
│   │   ├── data_loader.py     # Dataset handling
│   │   └── trainer.py         # Training/evaluation logic
│   ├── ResNet-50/
│   │   ├── README.md          # ResNet-50 specific documentation
│   │   ├── main.py            # Model entry point
│   │   ├── model.py           # ResNet-50 architecture
│   │   ├── data_loader.py     # Dataset handling
│   │   └── trainer.py         # Training/evaluation logic
│   └── DenseNet-121/
│       ├── README.md           # DenseNet-121 specific documentation
│       ├── main.py             # Model entry point
│       ├── model.py            # DenseNet-121 architecture
│       ├── data_loader.py      # Dataset handling
│       └── trainer.py          # Training/evaluation logic
├── Results/
│   ├── models/                 # Best trained model file per architecture
│   │   ├── VGG16_best.pth
│   │   ├── EfficientNet-B4_best.pth
│   │   ├── ResNet-50_best.pth
│   │   └── DenseNet-121_best.pth
│   ├── logs/                   # Training logs
│   └── plots/                  # Performance graphs
│   └── comparison/             # Models comparison result and graphs
├── README.md                   # Project overview & setup instructions if required
└── environment.yml             # Python dependencies
```

---

## References

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2018). Densely Connected Convolutional Networks. arXiv:1608.06993v5.   
Liu, Z. (N.D.) DenseNet. https://github.com/liuzhuang13/DenseNet   
Li, J. (2018). CelebFaces Attributes (CelebA) Dataset. Kaggle. https://www.kaggle.com/datasets/jessicali9530/celeba-dataset   

### Datasets

#### CelebA
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
