mutable struct NeuralNetwork
    W1::Matrix{Float64}  # Weights for the first layer
    b1::Vector{Float64}  # Biases for the first layer
    W2::Matrix{Float64}  # Weights for the second layer
    b2::Vector{Float64}  # Biases for the second layer
end

### x es un dato osea [sepallength, sepalwidth, petallength, petalwidth]
### la función esta implementada para cada dato. despues simplemente
### usamos X.forward_pass para obtener las predicciones de la red neuronal.

function foward_pass(model::NeuralNetwork, x) 
    # predicción de la red neuronal
    a1 = x 
    z1 = model.W1 * a1 .+ model.b1
    a2 = max.(0, z1)  # ReLU 
    z2 = model.W2 * a2 .+ model.b2
    a3 = exp.(z2) ./ sum(exp.(z2))  # Softmax
    return a3
end

# LO MISMO, esto es para cada dato.
function grad(model::NeuralNetwork, x, y) 
    W1_grad = zeros(size(model.W1))
    b1_grad = zeros(size(model.b1))
    W2_grad = zeros(size(model.W2))
    b2_grad = zeros(size(model.b2))

    # Implement gradient computation
    #
    # HACER LA CUENTA. CON CUIDADO.
    # pensar clase mauri.

    return (W1_grad, b1_grad, W2_grad, b2_grad)
end
