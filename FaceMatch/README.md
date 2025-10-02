# COSC595 Implementation: A Comparison of Four Deep Learning Architectures for Gendered Facial Recognition   

---
 
**Course:** COSC595 Information Technology Project: Implementation 

**Team:** Carl Fokstuen, YuTing Lee, Mark Malady, Nayani Samaranayake, Vishal Cheroor Ravi  

**Supervisors:** Prof. Raymond Chiong, Dr. Farshid Hajati   

**University of New England (UNE)** 

---

## Project Overview
This document constitutes as the project implementation's technical handbook.   

Additionally, this project implements and evaluates multiple deep learning architectures for gendered facial recognition for comparative 
purposes. These models include: 

1. VGG-16, 
2. EfficientNet-B4,
3. ResNet-50, and
4. DenseNet-121.

The models are trained on the CelebA dataset,
which contains 202,599 celebrity face images from 10,177 individuals.  
The dataset contains 40 attributes, including 5 facial features,
age, gender, and race. The attributes are used to train the models to predict the gender of a celebrity from a given image.  

--- 

## Project Installation

### Prerequisites
- Anaconda or Miniconda installed
- CUDA-capable GPU (recommended)
- 16GB+ RAM for training
- if you do not have celebA dataset available locally, you can download it from Kaggle:
  - Download the CelebA dataset from Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset, or
  - Prepare Kaggle API credentials, see: https://www.kaggle.com/docs/api



## Dependencies

### Core Requirements
- Python 3.9+
- PyTorch 2.0.1+
- CUDA 11.8+ (for GPU training)

### Required Packages
```bash
torch>=2.0.1
torchvision>=0.15.2
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
pillow>=10.0.0
matplotlib>=3.7.2
seaborn>=0.12.2
tqdm>=4.65.0
kaggle>=1.5.16
````

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
- **Privacy Concerns**: The four models can infer gendered demographic information from facial images   
  without explicit individual consent.  
- **Misgendering**: Binary classification (male/female) does not account for non-binary   
  or gender-diverse individuals  
- **Surveillance Misuse**: Technology could be deployed in contexts that violate privacy   
  rights or enable discriminatory practices  
- **Reinforcement of Binary Gender Norms**: The model architecture inherently assumes   
  gender is binary, which may not reflect lived experiences  

### Responsible Use Guidelines:
It is recommended that this system should only be used:   
- Within the boundaries of ethical research practices and institutional review   
- With explicit informed consent from individuals whose images are analyzed   
- In contexts where demographic classification serves a legitimate, beneficial purpose   
- With awareness of limitations and potential for misclassification   

### Academic Purpose:
This implementation is designed for educational and research purposes within the    
COSC595 course at the University of New England. It demonstrates technical capabilities    
in deep learning while emphasizing the importance of responsible AI development.   

**The authors strongly and explicitly discourage the deployment of these models in real-world    
applications without a prior comprehensive ethical review, privacy safeguards, and the consideration    
of any potential negative impacts on the targeted individual, or member of a protected class or community.**   


