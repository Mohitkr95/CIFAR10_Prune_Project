# CIFAR-10 Model Training and Pruning

This project provides an implementation for training a convolutional neural network (CNN) on the CIFAR-10 dataset and subsequently applying model pruning to reduce the model size.

## Table of Contents

- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Results](#results)
- [Contribute](#contribute)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Mohitkr95/CIFAR10_Prune_Project.git
   ```

2. Change your current directory to the project folder:
   ```
   cd CIFAR10_Prune_Project
   ```

3. Install required packages (Optional: It's recommended to create a virtual environment before installing the packages):
   ```
   pip install torch torchvision
   ```

## Directory Structure

```
CIFAR10_Prune_Project/
│
├── data/                   
│
├── models/                 
│   ├── model.pth
│   └── pruned_model.pth
│
├── src/                    
│   ├── __init__.py
│   ├── dataset.py          
│   ├── model.py            
│   ├── train.py            
│   └── prune.py            
│
├── utils/                  
│   └── __init__.py
│
├── main.py                 
│
└── README.md               
```

## Usage

1. Run the main script to train and prune the model:
   ```
   python main.py
   ```

2. The original and pruned model weights will be saved in the `models/` directory.

## Results

After running the `main.py` script, you'll get the following results:

![image](attachments\training.png)

- Training and validation accuracy of the original model.
- Validation accuracy after model pruning with different pruning ratios.
- Saved model weights for both the original and pruned models in the `models/` directory.

## License

[MIT](LICENSE) © Mohitkr95