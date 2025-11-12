from network import *
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


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
    LinearLayer(8, 64),
    ReLU(),
    LinearLayer(64, 1)
)
loss_fn = MSELoss()
optimizer = AdamOptimizer(model, learning_rate=0.001)


for epoch in range(100):
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
    for x_batch, y_batch in val_loader:
        y_pred = model.forward(x_batch)
        loss = loss_fn.forward(y_pred, y_batch)
        val_losses.append(loss)

    print(f"Epoch {epoch+1:03d} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")


# Test
test_losses = []
for x_batch, y_batch in val_loader:
    y_pred = model.forward(x_batch)
    loss = loss_fn.forward(y_pred, y_batch)
    test_losses.append(loss)

print(f"Test Loss: {np.mean(test_losses):.4f}")

