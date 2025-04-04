# 🔢 Dimensionality Estimator NN

A compact PyTorch-based neural network designed to estimate dimensionality from abstract input features and evaluate performance on both synthetic data and the MNIST dataset. Built with simplicity and modularity in mind.

---

## 🚀 Features

- Small and efficient model architecture (16 hidden units)
- Estimation from symbolic physical/mathematical inputs
- Accuracy metrics and ASCII-based visual feedback
- Optional evaluation on the MNIST dataset
- MIT Licensed – Free for personal and commercial use

---

## 🧠 Model Overview

```python
Input -> Linear -> ReLU -> Linear -> Output
Input size: dynamically determined by the feature vector

Hidden layer: 16 neurons (tunable)

Output: Single scalar representing total estimated dimensionality

📦 Requirements
Python 3.x

PyTorch

torchvision

Install with:

bash
Copy
Edit
pip install torch torchvision
🧪 Training: Dimensionality Estimation
This model uses a custom input composed of:

Base Dimensions: Physically symbolic values (e.g., Metric Tensor, Gravitational Waves)

Augmented Features: Fibonacci mod variations & digital root contributions

The network is trained using MSELoss to match the sum of all feature dimensions.

🔧 To Run
bash
Copy
Edit
python estimator.py
Expected output includes:

Estimated dimensionality

Absolute and relative error

ASCII-based accuracy bar

🔤 MNIST Evaluation
The model is reused to test its generalization capabilities on digit recognition with minimal changes.

Highlights:
Flattened 28×28 pixel images → 784 input size

Same model structure and training loop

Accuracy and loss displayed per epoch

📁 Project Structure
text
Copy
Edit
estimator.py           # Main script (custom + MNIST training)
📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute it as needed.

text
Copy
Edit
MIT License

Copyright (c) 2025 Ben Adams

Permission is hereby granted, free of charge, to any person obtaining a copy
...
(Include the full MIT license text in your repo's LICENSE file.)  
