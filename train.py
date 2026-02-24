import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time

# Import the custom classes we built in models.py
from models import TwoLayerMLP, SparseAutoencoder, RBM

# ==========================================
# 1. Dataset Loading & Preprocessing
# ==========================================
def load_mnist():
    print("Downloading MNIST dataset... (This might take a minute)")
    # Fetch MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    
    # Normalize pixel values to be between 0 and 1
    X = mnist.data.to_numpy() / 255.0
    y = mnist.target.to_numpy().astype(int)
    
    # Split into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")
    return X_train, X_test, y_train, y_test

def create_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for i in range(0, X.shape[0], batch_size):
        batch_idx = indices[i:i + batch_size]
        if y is not None:
            yield X[batch_idx], y[batch_idx]
        else:
            yield X[batch_idx]

# ==========================================
# 2. Training Loops
# ==========================================
def train_mlp(X_train, y_train, epochs=20, batch_size=64, lr=0.1):
    print("\n--- Training 2-Layer MLP ---")
    mlp = TwoLayerMLP(input_size=784, hidden_size=128, output_size=10, lr=lr)
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        epoch_loss, epoch_acc = [], []
        
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            loss, acc = mlp.train_step(X_batch, y_batch)
            epoch_loss.append(loss)
            epoch_acc.append(acc)
            
        avg_loss = np.mean(epoch_loss)
        avg_acc = np.mean(epoch_acc)
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")
        
    return mlp, history
def tune_mlp_hyperparameters(X_train, y_train, X_test, y_test):
    print("\n--- Starting Hyperparameter Tuning for MLP ---")
    
    # Define the grid
    learning_rates = [0.1, 0.05]
    hidden_sizes = [64, 128]
    batch_sizes = [64, 128]
    
    best_acc = 0
    best_params = {}
    
    # Grid Search Loop
    for lr in learning_rates:
        for hs in hidden_sizes:
            for bs in batch_sizes:
                print(f"Testing config -> LR: {lr}, Hidden: {hs}, Batch: {bs}")
                
                # Initialize model with current hyperparams
                mlp = TwoLayerMLP(input_size=784, hidden_size=hs, output_size=10, lr=lr)
                
                # Train for a quick 3 epochs just to gauge performance
                for epoch in range(3):
                    for X_batch, y_batch in create_batches(X_train, y_train, bs):
                        mlp.train_step(X_batch, y_batch)
                
                # Evaluate on test set
                test_preds = mlp.forward(X_test)
                test_predictions = np.argmax(test_preds, axis=1)
                test_acc = np.mean(test_predictions == y_test)
                
                print(f"Resulting Test Accuracy: {test_acc:.4f}\n")
                
                # Save the best model parameters
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_params = {'lr': lr, 'hidden_size': hs, 'batch_size': bs}
                    
    print(f"=== Best Hyperparameters Found ===")
    print(f"LR: {best_params['lr']}, Hidden Size: {best_params['hidden_size']}, Batch Size: {best_params['batch_size']}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    
    return best_params


def train_autoencoder(X_train, epochs=15, batch_size=64, lr=0.1):
    print("\n--- Training Sparse Autoencoder ---")
    ae = SparseAutoencoder(input_size=784, latent_size=64, lr=lr, sparsity_weight=1e-4)
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = []
        for X_batch in create_batches(X_train, None, batch_size):
            # Forward pass
            reconstructions = ae.forward(X_batch)
            # Backward pass
            ae.backward(X_batch)
            # Track MSE loss
            epoch_loss.append(np.mean((reconstructions - X_batch)**2))
            
        avg_loss = np.mean(epoch_loss)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - MSE Loss: {avg_loss:.4f}")
        
    return ae, loss_history

def train_rbm(X_train, epochs=10, batch_size=64, lr=0.05):
    print("\n--- Training RBM ---")
    rbm = RBM(visible_size=784, hidden_size=64, lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = []
        # RBM is traditionally trained on binary/thresholded data, but works on [0,1] normalized
        for X_batch in create_batches(X_train, None, batch_size):
            loss = rbm.contrastive_divergence(X_batch)
            epoch_loss.append(loss)
            
        avg_loss = np.mean(epoch_loss)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Reconstruction Error: {avg_loss:.4f}")
        
    return rbm, loss_history

# ==========================================
# 3. Visualization Methods (For your Report)
# ==========================================
def plot_mlp_curves(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Cross-Entropy Loss', color='red')
    plt.title('MLP Training Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Accuracy', color='blue')
    plt.title('MLP Training Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_reconstructions(ae, X_test, num_images=5):
    
    # Grab a few test images
    sample_images = X_test[:num_images]
    reconstructed_images = ae.forward(sample_images)
    
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_rbm_filters(rbm, num_filters=16):
    
    # RBM weights are shape (784, hidden_size). Transpose to get (hidden_size, 784)
    filters = rbm.W.T
    
    # Plot a 4x4 grid of filters
    grid_size = int(np.sqrt(num_filters))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    fig.suptitle("RBM Learned Filters (Weights)")
    
    for i, ax in enumerate(axes.flat):
        if i < len(filters):
            # Reshape the 784 weight vector back into a 28x28 image
            ax.imshow(filters[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def test_outlier_detection(ae, X_test):
    print("\n--- Running Outlier Detection ---")
    
    # 1. Calculate reconstruction errors for normal test data
    normal_errors = ae.reconstruction_error(X_test)
    
    # 2. Set threshold at the 95th percentile of normal data
    threshold = np.percentile(normal_errors, 95)
    print(f"Calculated Anomaly Threshold (95th percentile): {threshold:.4f}")
    
    # 3. Create "Outliers" (e.g., random noise images)
    num_outliers = 50
    np.random.seed(42)
    outliers = np.random.rand(num_outliers, 784) # Pure random uniform noise
    
    # 4. Calculate errors for outliers
    outlier_errors = ae.reconstruction_error(outliers)
    
    # 5. Evaluate
    normal_flagged = np.sum(normal_errors > threshold)
    outliers_flagged = np.sum(outlier_errors > threshold)
    
    print(f"Normal images flagged as anomalies: {normal_flagged} out of {len(X_test)} ({(normal_flagged/len(X_test))*100:.2f}%)")
    print(f"Noise images flagged as anomalies: {outliers_flagged} out of {num_outliers} ({(outliers_flagged/num_outliers)*100:.2f}%)")
    
    # Plotting the histogram for the report
    plt.figure(figsize=(8, 4))
    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal Digits', color='blue')
    plt.hist(outlier_errors, bins=10, alpha=0.6, label='Random Noise (Outliers)', color='red')
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.2f})')
    plt.title('Outlier Detection via Reconstruction Error')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_mnist()
    
    # 2. Train MLP
    mlp, mlp_history = train_mlp(X_train, y_train, epochs=10, batch_size=64, lr=0.05)
    plot_mlp_curves(mlp_history)
    # Run Hyperparameter tuning
    best_params = tune_mlp_hyperparameters(X_train, y_train, X_test, y_test)
    
    # 3. Train Autoencoder
    ae, ae_history = train_autoencoder(X_train, epochs=10, batch_size=64, lr=0.1)
    plot_reconstructions(ae, X_test, num_images=5)
    
    #test_outlier_Detection
    test_outlier_detection(ae, X_test)

    # 4. Train RBM
    rbm, rbm_history = train_rbm(X_train, epochs=10, batch_size=64, lr=0.05)
    plot_rbm_filters(rbm, num_filters=16)
    
    print("\nAll models trained and visualizations generated successfully!")