# A (simple) Neural Network & Gradient Descent.

A neural network with one hidden layer implemented from scratch in Julia, trained on the classic Iris dataset using gradient descent and backpropagation.

---

## Project structure

```
.
├── main.jl              # Entry point: data loading, training, evaluation
├── neural_network.jl    # NeuralNetwork struct, forward_pass, grad (backprop), loss
├── gradient_descent.jl  # Optimizer abstract type, GradientDescent, train!
├── utils_tools.jl       # Data preprocessing: split, normalize, onehot, prepare_data
├── evaluation.jl        # Inference: predict, accuracy (test), results plot
├── Project.toml         # Direct dependencies
├── Manifest.toml        # Full dependency tree (pinned versions)
├── iris.png             # Scatter plots: petal and sepal dimensions by species
└── loss.png             # Training and validation loss curves with early stop marker
```

### File responsibilities

- **`neural_network.jl`** — defines the `NeuralNetwork{T}` struct (parametric on float type) and all core math: forward pass, backpropagation via `grad`, and `cross_entropy` / `loss_total`.
- **`gradient_descent.jl`** — defines an `Optimizer` abstract type (extensible to other methods) and implements `GradientDescent` with a `train!` loop that records both train and validation loss at each iteration.
- **`utils_tools.jl`** — data pipeline helpers: reproducible train/test split (seeded), z-score normalization (fitted on train only, applied to test), and one-hot encoding.
- **`evaluation.jl`** — evaluation utilities: `predict` (argmax over softmax output), `test` (prints accuracy), and `results` (loss curve plot with early stop marker).
- **`main.jl`** — wires everything together: loads Iris, prepares data, initializes the model with random weights, runs training, and saves plots.

---

## Planned extensions

The `Optimizer` abstract type in `gradient_descent.jl` is designed for easy extension.

- [ ] Gradient descent with Armijo line search
- [ ] Nonlinear Conjugate Gradient (Fletcher–Reeves)

---

## Dataset

[Iris](https://archive.ics.uci.edu/ml/datasets/iris) — 150 samples, 3 species (_setosa_, _versicolor_, _virginica_), 4 features (petal/sepal length and width). Loaded via `RDatasets.jl`.
