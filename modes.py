import numpy as np

# ==========================================
# Helper Activation & Loss Functions
# ==========================================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    # Clipped to prevent overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    # Shift x for numerical stability
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    # Add epsilon to prevent log(0)
    log_preds = -np.log(y_pred[np.arange(m), y_true] + 1e-9)
    return np.sum(log_preds) / m

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# ==========================================
# Task A: 2-Layer MLP from Scratch
# ==========================================

class TwoLayerMLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr
        # He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        # Xavier initialization for Softmax
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true_labels):
        m = X.shape[0]
        
        # One-hot encode true labels for gradient calculation
        Y_one_hot = np.zeros((m, self.W2.shape[1]))
        Y_one_hot[np.arange(m), y_true_labels] = 1

        # Output layer gradients (Softmax + Cross Entropy simplification)
        dZ2 = self.A2 - Y_one_hot
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_deriv(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # SGD Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, X_batch, y_batch):
        preds = self.forward(X_batch)
        loss = cross_entropy_loss(preds, y_batch)
        self.backward(X_batch, y_batch)
        
        predictions = np.argmax(preds, axis=1)
        acc = np.mean(predictions == y_batch)
        return loss, acc

# ==========================================
# Task B: Autoencoder (Undercomplete + Sparse)
# ==========================================

class SparseAutoencoder:
    def __init__(self, input_size=784, latent_size=64, lr=0.01, sparsity_weight=1e-4):
        self.lr = lr
        self.sparsity_weight = sparsity_weight # L1 penalty weight
        
        # Encoder weights
        self.W_enc = np.random.randn(input_size, latent_size) * 0.01
        self.b_enc = np.zeros((1, latent_size))
        
        # Decoder weights
        self.W_dec = np.random.randn(latent_size, input_size) * 0.01
        self.b_dec = np.zeros((1, input_size))

    def forward(self, X):
        # Encoder (ReLU activation)
        self.Z_enc = np.dot(X, self.W_enc) + self.b_enc
        self.A_enc = relu(self.Z_enc)
        
        # Decoder (Sigmoid activation to output values 0-1 like normalized images)
        self.Z_dec = np.dot(self.A_enc, self.W_dec) + self.b_dec
        self.A_dec = sigmoid(self.Z_dec)
        return self.A_dec

    def backward(self, X):
        m = X.shape[0]
        
        # MSE Loss derivative
        dA_dec = (self.A_dec - X) / m 
        dZ_dec = dA_dec * sigmoid_deriv(self.Z_dec)
        
        dW_dec = np.dot(self.A_enc.T, dZ_dec)
        db_dec = np.sum(dZ_dec, axis=0, keepdims=True)
        
        # Backprop to encoder + L1 Sparsity penalty derivative (sign of activation)
        dA_enc = np.dot(dZ_dec, self.W_dec.T) + (self.sparsity_weight * np.sign(self.A_enc))
        dZ_enc = dA_enc * relu_deriv(self.Z_enc)
        
        dW_enc = np.dot(X.T, dZ_enc)
        db_enc = np.sum(dZ_enc, axis=0, keepdims=True)

        # SGD Update
        self.W_dec -= self.lr * dW_dec
        self.b_dec -= self.lr * db_dec
        self.W_enc -= self.lr * dW_enc
        self.b_enc -= self.lr * db_enc

    def reconstruction_error(self, X):
        reconstructions = self.forward(X)
        return np.mean((X - reconstructions)**2, axis=1) # Error per sample for outlier detection

# ==========================================
# Task C: Restricted Boltzmann Machine (RBM)
# ==========================================

class RBM:
    def __init__(self, visible_size=784, hidden_size=64, lr=0.01):
        self.lr = lr
        self.W = np.random.normal(0, 0.01, (visible_size, hidden_size))
        self.v_bias = np.zeros(visible_size)
        self.h_bias = np.zeros(hidden_size)

    def sample_hidden(self, v):
        h_prob = sigmoid(np.dot(v, self.W) + self.h_bias)
        h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(float)
        return h_prob, h_sample

    def sample_visible(self, h):
        v_prob = sigmoid(np.dot(h, self.W.T) + self.v_bias)
        v_sample = (np.random.rand(*v_prob.shape) < v_prob).astype(float)
        return v_prob, v_sample

    def contrastive_divergence(self, v0):
        # Positive phase
        h0_prob, h0_sample = self.sample_hidden(v0)
        pos_associations = np.dot(v0.T, h0_prob)

        # Negative phase (1 step of Gibbs sampling)
        v1_prob, v1_sample = self.sample_visible(h0_sample)
        h1_prob, _ = self.sample_hidden(v1_sample)
        neg_associations = np.dot(v1_prob.T, h1_prob)

        # Update weights (SGD)
        m = v0.shape[0]
        self.W += self.lr * ((pos_associations - neg_associations) / m)
        self.v_bias += self.lr * np.mean(v0 - v1_prob, axis=0)
        self.h_bias += self.lr * np.mean(h0_prob - h1_prob, axis=0)
        
        # Return reconstruction error for tracking
        return mse_loss(v1_prob, v0)