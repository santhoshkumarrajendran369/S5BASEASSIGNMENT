# MNIST Classification ML Pipeline

A robust machine learning pipeline implementing a CNN model for MNIST digit classification with automated testing and CI/CD integration.

## 🎯 Project Overview

This project implements an efficient Convolutional Neural Network (CNN) that achieves high accuracy on the MNIST dataset while maintaining a small parameter footprint (<25,000 parameters). The pipeline includes automated testing, model validation, and CI/CD integration through GitHub Actions.

## 🌟 Key Features

- Efficient CNN architecture with squeeze-excitation attention mechanism
- Automated testing and validation pipeline
- Parameter count optimization (<25,000)
- High accuracy achievement (>95%) in single epoch training
- Integrated CI/CD with GitHub Actions
- Automated model artifact storage

## 🏗️ Architecture

The CNN architecture includes:
- Three convolutional blocks with batch normalization
- Squeeze-excitation attention mechanism
- Efficient parameter utilization
- Leaky ReLU activation
- Minimal dropout for regularization

## 📊 Model Performance

- Parameters: <25,000
- Training Accuracy: >95% (single epoch)
- Input Shape: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## 🚀 Quick Start

1. Clone the repository:
bash
git clone <your-repo-url>
cd <repo-name>

2. Create and activate virtual environment:
bash
python -m venv .venv
source .venv/bin/activate

3. Install dependencies:
bash
pip install -r requirements.txt

4. Run training:
bash
python train.py

5. Run tests:
bash
pytest tests/test_model.py -v

## 📁 Project Structure
.
├── model/
│ ├── init.py
│ └── network.py # CNN model architecture
├── tests/
│ └── test_model.py # Model tests
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # CI/CD configuration
├── train.py # Training script
├── requirements.txt # Project dependencies
├── setup.py # Package setup
└── .gitignore # Git ignore rules

## 🔍 Testing

The automated testing suite verifies:
- Model parameter count (<25,000)
- Input/output shape compatibility (28x28 → 10)
- Model accuracy threshold (>80% on test set)
- Model loading and saving functionality

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python 3.8 environment
2. Installs required dependencies
3. Runs model training
4. Executes test suite
5. Stores trained model as artifact (90-day retention)

## 📦 Dependencies

Core requirements:
- PyTorch
- torchvision
- pytest

## 🛠️ Model Training Details

### Architecture Features
- Squeeze-excitation attention blocks
- Batch normalization layers
- Leaky ReLU activation
- Efficient parameter usage
- Dropout regularization (0.1)

### Training Configuration
- Optimizer: SGD with Nesterov momentum (0.9)
- Learning Rate: 0.2 → 0.4 with OneCycleLR
- Batch Size: 512 (training), 128 (fine-tuning)
- Data Split: 98% training, 2% fine-tuning
- Mixed Precision Training enabled

### Data Augmentation
- Random rotation (±5 degrees)
- Random affine transforms (scale: 0.98-1.02)
- Minimal distortion for stability

## 📈 Training Process

The training follows a two-phase approach:
1. Initial Training Phase
   - Aggressive learning rate cycling
   - Large batch size for stability
   - Mixed precision training
   - Gradient clipping (0.5)

2. Fine-tuning Phase
   - Adam optimizer
   - Lower learning rate (0.0001)
   - Smaller batch size
   - Weight decay regularization

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Acknowledgments

- MNIST Dataset creators
- PyTorch development team
- GitHub Actions team

## 📧 Contact

Your Name - [santhoshkumar3ram@gmail.com](mailto:santhoshkumar3ram@gmail.com)

Project Link: [https://github.com/santhoshkumarrajendran369/S5BASEASSIGNMENT](https://github.com/santhoshkumarrajendran369/S5BASEASSIGNMENT)

