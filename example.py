from network import *
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Example dataset from sklearn
X, y = fetch_california_housing(return_X_y=True)
y = y.reshape(-1, 1).astype(np.float32)
X = X.astype(np.float32)


# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split data
n_total = len(X)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

train_dataset = Dataset(X_train, y_train)
val_dataset = Dataset(X_val, y_val)
test_dataset = Dataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


model = Model(
    LinearLayer(8, 8),
    ReLU(),
    LinearLayer(8, 1)
)
loss_fn = MSELoss()
optimizer = AdamWOptimizer(model, learning_rate=0.001, weight_decay=0.01)


# Tracking: Trainings-/Validierungsverlaufsdaten
train_loss_history = []
val_loss_history = []
val_r2_history = []

num_epochs = 100

for epoch in range(num_epochs):
    # Train
    train_losses = []
    for x_batch, y_batch in train_loader:
        y_pred = model.forward(x_batch)
        loss = loss_fn.forward(y_pred, y_batch)
        grad_loss = loss_fn.backward()
        model.backward(grad_loss)
        optimizer.step()
        train_losses.append(loss)

    # Validate
    val_losses = []
    val_preds = []
    val_truths = []
    for x_batch, y_batch in val_loader:
        y_pred = model.forward(x_batch)
        loss = loss_fn.forward(y_pred, y_batch)
        val_losses.append(loss)
        val_preds.append(y_pred)
        val_truths.append(y_batch)

    mean_train = float(np.mean(train_losses))
    mean_val = float(np.mean(val_losses))
    val_pred_full = np.concatenate(val_preds, axis=0)
    val_truth_full = np.concatenate(val_truths, axis=0)
    val_r2 = r2_score(val_truth_full.ravel(), val_pred_full.ravel())

    train_loss_history.append(mean_train)
    val_loss_history.append(mean_val)
    val_r2_history.append(float(val_r2))

    print(
        f"Epoch {epoch+1:03d} | Train Loss: {mean_train:.4f} | Val Loss: {mean_val:.4f} | Val R2: {val_r2:.4f}"
    )


# Test
test_losses = []
for x_batch, y_batch in test_loader:
    y_pred = model.forward(x_batch)
    loss = loss_fn.forward(y_pred, y_batch)
    test_losses.append(loss)

print(f"Test Loss: {np.mean(test_losses):.4f}")


# Plot: Trainings- und Validierungsverlust sowie Val-R2-Verlauf
epochs = np.arange(1, num_epochs + 1)
plt.figure(figsize=(12, 4))

# Loss-Kurven
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_history, label="Train Loss")
plt.plot(epochs, val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Train/Val Loss über Epochen")
plt.legend()
plt.grid(alpha=0.3)

# R2-Kurve
plt.subplot(1, 2, 2)
plt.plot(epochs, val_r2_history, color="tab:green", label="Val R2")
plt.xlabel("Epoch")
plt.ylabel("R2 Score")
plt.title("Val R2 über Epochen")
plt.ylim(-1.0, 1.0)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
try:
    # In Notebook/GUI-Umgebungen anzeigen; in Headless-Umgebungen bleibt nur die PNG-Datei
    plt.show()
except Exception:
    pass

