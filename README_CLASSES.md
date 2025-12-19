# Zweite README: Klassen- und API-Übersicht

Diese README beschreibt kompakt alle bereitgestellten Klassen, ihre wichtigsten Methoden und wie sie in einem einfachen Trainingsablauf zusammenspielen. Der Code ist minimalistisch und nutzt ausschließlich NumPy – ideal zum Lernen und für kleine Experimente.

## Architektur und Datenfluss

Dataset → DataLoader (Batches) → Model (Layers: Linear, ReLU, …) → Loss (MSELoss) → Optimizer (AdamWOptimizer)

- Dataset liefert Beispiele (x, y).
- DataLoader erzeugt Mini-Batches aus dem Dataset.
- Model verkettet Layers und führt Vorwärts-/Rückwärtsdurchlauf aus.
- Loss berechnet den Fehler und liefert den Gradienten zur Vorhersage.
- Optimizer aktualisiert die Parameter der Layers anhand ihrer Gradienten.

---

## Modul: `network/dataset.py`

### Klasse: `Dataset`
Einfacher Container für Features und Targets.

Wichtigste Methoden:
- `__len__(self) -> int`
  - Anzahl der Beispiele (Batch-Dimension von `x`).
- `__getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]`
  - Gibt das Paar `(x[idx], y[idx])` zurück.

Inputs/Outputs und Annahmen:
- `x`: NumPy-Array der Form `(N, D)` oder allgemein mit erster Achse als Beispiele.
- `y`: NumPy-Array der Form `(N, …)`; die erste Achse korrespondiert zu `x`.

Edge Cases/Hinweise:
- Es erfolgt keine Formvalidierung – sicherstellen, dass `x.shape[0] == y.shape[0]`.

---

## Modul: `network/dataloader.py`

### Klasse: `DataLoader`
Erzeugt iterierbare Mini-Batches aus einem `Dataset`.

Konstruktor:
- `DataLoader(dataset, batch_size, shuffle=True)`

Wichtigste Methoden:
- `__iter__(self)`
  - Initialisiert/reshuffled Indexreihenfolge, setzt Cursor zurück; gibt `self` zurück.
- `__next__(self) -> tuple[np.ndarray, np.ndarray]`
  - Liefert das nächste Batch `(x_batch, y_batch)` als gestapelte Arrays; wirft `StopIteration`, wenn alle Beispiele konsumiert sind.

Details/Verhalten:
- Shuffling: Bei jedem neuen Iterator-Durchlauf (z. B. jeder Epoche) werden die Indizes neu gemischt, wenn `shuffle=True`.
- Letztes Batch kann kleiner sein als `batch_size` oder exakt passen – je nach `N % batch_size`.

---

## Modul: `network/linear.py`

### Klasse: `LinearLayer`
Affine Schicht: `y = x @ W^T + b`. Speichert Eingaben für Backpropagation.

Konstruktor:
- `LinearLayer(input_size, output_size)`
  - Initialisiert Gewichte/Bias mit sinnvoller Varianz (He-Init für ReLU-Default).

Wichtigste Methoden:
- `init_weights(self, activation="relu")`
  - Initialisiert `weights` (Form `(output_size, input_size)`) und `biases` (Form `(output_size,)`).
  - Standard: He-Varianzschätzung für ReLU; alternativ einfache Varianz für andere Aktivierungen.
- `forward(self, input) -> np.ndarray`
  - Cacht `input`; gibt `input @ W^T + b` zurück.
  - Unterstützt Batch-Inputs `(B, input_size)` und Einzel-Inputs `(input_size,)` (wird intern für Gradienten vereinheitlicht).
- `backward(self, upstream_gradient) -> np.ndarray`
  - Erwartet Gradienten der Form wie `forward`-Output.
  - Berechnet und speichert `dw` (Form `(output_size, input_size)`) und `db` (Form `(output_size,)`).
  - Gibt Gradienten w. r. t. Input zurück: `dx` mit Form `(B, input_size)` bzw. `(input_size,)` für Einzeleingaben.
- `parameters(self) -> list[np.ndarray]`
  - Liefert `[weights, biases]` in fester Reihenfolge.
- `gradients(self) -> list[np.ndarray]`
  - Liefert `[dw, db]` passend zu `parameters()`.

Edge Cases/Hinweise:
- `backward` setzt voraus, dass zuvor `forward` aufgerufen wurde (wegen gecachtem Input).
- Bei Einzeleingaben wird die Ausgabe/der Gradient wieder auf 1D zurückgesqueezed.

---

## Modul: `network/relu.py`

### Klasse: `ReLU`
Elementweise ReLU-Aktivierung: `max(0, x)`.

Wichtigste Methoden:
- `forward(self, input) -> np.ndarray`
  - Cacht `input`; gibt `np.maximum(0, input)` zurück.
- `backward(self, upstream_gradient) -> np.ndarray`
  - Leitet nur dort Gradienten weiter, wo der gecachte Input > 0 war.
- `parameters(self) -> list`
  - Leere Liste (keine Parameter).
- `gradients(self) -> list`
  - Leere Liste (keine Gradienten).

Hinweis:
- Zustandslos bzgl. Trainierbarkeit – nur Cache für Backprop.

---

## Modul: `network/mse.py`

### Klasse: `MSELoss`
Mittlerer quadratischer Fehler (voll reduziert über alle Elemente).

Wichtigste Methoden:
- `forward(self, predicted, truth) -> float`
  - Speichert `predicted` und `truth`; gibt `mean((predicted - truth)^2)` zurück.
- `backward(self) -> np.ndarray`
  - Gibt Gradienten der Loss w. r. t. `predicted` zurück: `2 / predicted.size * (predicted - truth)`.

Hinweise:
- Die Normierung erfolgt über die Gesamtanzahl Elemente (`predicted.size`), nicht nur die Batchgröße. Das ist konsistent, wenn `predicted` z. B. Form `(B, 1)` hat.

---

## Modul: `network/model.py`

### Klasse: `Model`
Ein sequentielles Container-Modell, das beliebige Layer mit `forward`/`backward` verkettet.

Konstruktor:
- `Model(*layers)`
  - Speichert die Layer in der Reihenfolge des Vorwärtsdurchlaufs.

Wichtigste Methoden:
- `forward(self, input) -> np.ndarray`
  - Leitet `input` nacheinander durch alle Layer (`layer.forward`).
- `backward(self, input) -> np.ndarray`
  - Führt Backpropagation in umgekehrter Reihenfolge aus (`layer.backward`).

Annahmen an Layer:
- Jedes Layer muss `forward(x)` und `backward(grad)` implementieren.
- Für Optimizer-Unterstützung sollten Layer `parameters()` und `gradients()` anbieten (leere Listen sind erlaubt, z. B. für ReLU).

---

## Modul: `network/adam.py`

### Klasse: `AdamWOptimizer`
Optimizer nach AdamW (Adam mit entkoppeltem Weight Decay) für alle Parameter in `model.layers`.

Konstruktor:
- `AdamWOptimizer(model, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)`
  - Legt pro Layer und Parameter separate Momente `m` und `v` an und hält den Timestep `t`.

Wichtigste Methoden:
- `step(self) -> None`
  - Erwartet, dass vorher `model.backward(...)` die Gradienten in den Layern befüllt hat.
  - Aktualisiert jedes Parameter-Array `param` gemäß Adam-Formeln mit Bias-Korrektur und Weight Decay:
    - `m = beta1 * m + (1 - beta1) * grad`
    - `v = beta2 * v + (1 - beta2) * grad**2`
    - `m_hat = m / (1 - beta1**t)`, `v_hat = v / (1 - beta2**t)`
    - `param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)`
  - Inkrementiert `t` am Ende jedes Schritts.

Anforderungen an `model.layers`:
- Jedes Layer muss `parameters()` (Liste von Parametern) und `gradients()` (gleiche Struktur, Gradienten dazu) bereitstellen.

---

## Modul: `network/__init__.py`

Re-exportiert die öffentliche API für bequemen Import:

```python
from network import AdamWOptimizer, DataLoader, Dataset, LinearLayer, Model, MSELoss, ReLU
```

---

## Zusammenspiel im Training (Kurzbeispiel)

Auszug aus `example.py` mit Kommentaren:

```python
model = Model(
    LinearLayer(8, 8),
    ReLU(),
    LinearLayer(8, 1)
)
loss_fn = MSELoss()
optimizer = AdamWOptimizer(model, learning_rate=0.005, weight_decay=0.01)

for x_batch, y_batch in train_loader:
    # Vorwärtsdurchlauf
    y_pred = model.forward(x_batch)
    loss = loss_fn.forward(y_pred, y_batch)

    # Loss-Gradient (dL/dy_pred)
    grad_loss = loss_fn.backward()

    # Backprop durch das Modell (füllt Layer-Gradienten)
    model.backward(grad_loss)

    # Parameter-Update
    optimizer.step()
```

---

## Hinweise und Best Practices

- Shapes: Achten Sie darauf, dass die Shapes zwischen Layern konsistent sind (z. B. `LinearLayer(input_size, output_size)` passend zur vorigen Ausgabe).
- Loss-Reduktion: `MSELoss` reduziert über alle Elemente; das beeinflusst die Skalierung der Gradienten.
- Zustand in Layern: `forward` cacht Eingaben; rufen Sie nicht `backward` ohne vorherigen `forward` auf.
- Reproduzierbarkeit: Optional zu Beginn `np.random.seed(<int>)` setzen.

## Verweis auf Tests

Unter `tests/` finden Sie unit tests, die das Verhalten der Komponenten prüfen:
- `test_linear.py`, `test_relu.py`, `test_model.py` – Vorwärts-/Rückwärtslogik
- `test_mse.py` – Loss und Gradient
- `test_dataset.py`, `test_dataloader.py` – Datenpfad
- `test_adam.py` – Optimizer-Update

Diese Tests dienen als lebende Spezifikation der API und sind eine gute Ergänzung zu dieser README.
