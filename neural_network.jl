# Red neuronal con una capa oculta.
mutable struct NeuralNetwork{T <: AbstractFloat}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
end

# Softmax por columnas. Restamos el máximo para evitar overflow.
function softmax(Z)
    E = exp.(Z .- maximum(Z; dims=1))
    return E ./ sum(E; dims=1)
end

# Utilizamos ReLU y Softmax como funciones de activación.
# Sirve para un dato (vector) o para varios a la vez (matriz, un dato por columna).
function forward_pass(model::NeuralNetwork, X)
    A2 = max.(0, model.W1 * X .+ model.b1)   # ReLU
    return softmax(model.W2 * A2 .+ model.b2) # Softmax
end

# Obs: la derivada de ReLU es una matriz diagonal, la pensamos como vector.
relu_derivative(x) = Float64.(x .> 0)

# Backpropagation. Devuelve el gradiente promedio sobre todos los datos
# (columnas de X), que usamos como dirección de descenso.
function grad(model::NeuralNetwork, X, Y)
    n = size(X, 2)

    Z1 = model.W1 * X .+ model.b1
    A2 = max.(0, Z1)
    Y_hat = softmax(model.W2 * A2 .+ model.b2)

    delta_2 = Y_hat .- Y
    delta_1 = relu_derivative(Z1) .* (model.W2' * delta_2)

    # Derivadas parciales respecto de W_1, b_1, W_2 y b_2.
    W1_part = delta_1 * X' / n
    b1_part = vec(sum(delta_1; dims=2)) / n
    W2_part = delta_2 * A2' / n
    b2_part = vec(sum(delta_2; dims=2)) / n

    return (W1_part, b1_part, W2_part, b2_part)
end

# Función de pérdida (suma sobre todos los datos).
function cross_entropy(Y_hat, Y; tol = 1e-10)
    return -sum(Y .* log.(Y_hat .+ tol))
end

# Calculamos la pérdida total (promedio por dato).
function loss_total(model::NeuralNetwork, X, Y)
    return cross_entropy(forward_pass(model, X), Y) / size(X, 2)
end

# Inicializamos la red con pesos aleatorios (He, pensada para ReLU) y sesgos nulos.
# Fijamos una seed para poder replicar.
function init_network(n_in, n_hidden, n_out; seed = 2012)
    rng = Xoshiro(seed)
    W1 = randn(rng, n_hidden, n_in) * sqrt(2 / n_in)
    b1 = zeros(n_hidden)
    W2 = randn(rng, n_out, n_hidden) * sqrt(2 / n_hidden)
    b2 = zeros(n_out)
    return NeuralNetwork(W1, b1, W2, b2)
end
