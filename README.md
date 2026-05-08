# EEG Movement Classification Comparison 
This repository will carry out a comparison between two approaches for classifying EEG movement patterns.

 The dataset is detailed at docs/DatabaseDescription.pdf.
 
## Setup

### 1. Create Virtual Environment
To set up the environment we will use python 3.12, and we will create a .venv:

```bash
py -3.12 -m venv .venv
```

### 2. Activate Virtual Environment

**Linux**
```bash
. .venv/bin/activate
```

 **Windows**
```bash
. .venv/Scripts/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

## Repository structure
This repository is intended to do the full pipeline of a AI/ML learning application. It will go from dataset exploration and processing, to model training and validation.

```bash
├── data
│   ├── original
│   │   ├── S1
│   │   │   ├── ME_S01_r01.mat
│   │   │   ├── ...
│   │   ├── S12
│   │   │   ├── ME_S12_r01.mat
│   │   │   ├── ...
│   │   ├── S3
│   │   │   ├── ME_S03_r01.mat
│   │   │   ├── ...
│   │   ├── ...
│   └── processed # Used to train model
│       ├── dataset_test.npz
│       └── dataset_train.npz
├── docs
│   └── DatabaseDescription.pdf
├── notebooks
│   └── visualize_processed_eeg.ipynb
├── scripts
│   ├── build_dataset.py
│   └── visualizer.py
├── src
│   ├── config.py
│   ├── load_data.py
│   ├── pipeline.py
│   └── utils.py
├── README.md
└── requirements.txt
```

`scripts/` folder contains files that are intended to be run alone. Those consume code from `src/`. The scripts should be run from the root directory:

```bash
# For building the dataset
python -m scripts.build_dataset

# For launching visualizer app
python -m scripts.visualizer

# For training model
python -m scripts.train_csp
python -m scripts.train_dln

# For evaluating a model
python -m scripts.test "models/some_model.pth"
```
