📌 Project Overview

This project demonstrates the implementation of core deep learning architectures completely from scratch using Python and NumPy, without using high-level deep learning frameworks such as TensorFlow or PyTorch.

The objective is to understand the mathematical foundations behind neural networks by manually implementing:

Forward Propagation

Backpropagation

Gradient Descent Optimization

Energy-based Learning (Contrastive Divergence)

The models are trained and evaluated on the MNIST handwritten digit dataset.

🎯 Objectives

The project implements and evaluates three different neural architectures:

Multi-Layer Perceptron (MLP) – Supervised classification

Sparse Autoencoder – Unsupervised representation learning & anomaly detection

Restricted Boltzmann Machine (RBM) – Generative modeling

📂 Dataset

Dataset Used: MNIST Handwritten Digits

60,000 training images

10,000 testing images

Image size: 28 × 28 (grayscale)

Flattened input size: 784

Pixel normalization: [0, 255] → [0, 1]

🛠 Tools & Technologies

Programming Language: Python 3.x

Numerical Computation: NumPy

Visualization: Matplotlib

Dataset Fetching: Scikit-learn

Deep Learning Frameworks Used: ❌ None

🧠 Model Architectures
1️⃣ Multi-Layer Perceptron (MLP)
Architecture

Input Layer: 784 neurons

Hidden Layer: 128 neurons

Output Layer: 10 neurons

Activations

Hidden: ReLU

Output: Softmax

Loss Function

Categorical Cross-Entropy

Optimization

Stochastic Gradient Descent (SGD)

He Initialization (Hidden Layer)

Xavier Initialization (Output Layer)

Best Hyperparameters

Learning Rate: 0.05

Batch Size: 64

Hidden Units: 128

2️⃣ Sparse Autoencoder
Architecture

Input Layer: 784

Bottleneck Layer: 64

Output Layer: 784

Activations

Encoder: ReLU

Decoder: Sigmoid

Loss Function

Mean Squared Error (MSE)

Regularization

L1 Sparsity Penalty (λ = 10⁻⁴)

Application

Image reconstruction

Outlier detection using 95th percentile reconstruction error threshold

3️⃣ Restricted Boltzmann Machine (RBM)
Architecture

Visible Units: 784

Hidden Units: 64

Bipartite Graph Structure

Training Algorithm

Contrastive Divergence (CD-1)

Gibbs Sampling

Purpose

Learn joint probability distribution

Extract generative feature filters

🔬 Experiments Conducted
✔ Hyperparameter Tuning (MLP)

Grid search over:

Learning rates (0.1, 0.05)

Hidden units (64, 128)

Batch sizes (64, 128)

✔ Autoencoder Outlier Detection

Reconstruction error threshold set at 95th percentile

Random noise images successfully flagged as anomalies

📊 Results Summary
✅ MLP

Fast convergence

High classification accuracy

Smooth cross-entropy loss decay

✅ Autoencoder

Successful reconstruction of digits

Effective compression

Reliable anomaly detection

✅ RBM

Learned stroke-based digit features

Extracted meaningful generative filters

📈 Key Learning Outcomes

Manual implementation of backpropagation

Deep understanding of gradient flow

Understanding generative vs discriminative models

Energy-based learning concepts

Importance of weight initialization

⚠ Limitations

CPU-only training (no GPU acceleration)

Slower execution compared to deep learning frameworks

Basic SGD instead of adaptive optimizers

🚀 Future Improvements

Implement Adam / RMSprop

Extend MLP to CNN architecture

Implement deeper autoencoders

Add momentum-based optimization

📚 References

Goodfellow, Bengio & Courville — Deep Learning, MIT Press

LeCun et al. (1998) — Gradient-Based Learning Applied to Document Recognition

Hinton (2002) — Training Products of Experts using Contrastive Divergence