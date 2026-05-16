using LinearAlgebra

# Arquitectura de la red.
mutable struct NeuralNetwork{T <:AbstractFloat}
    W1::Matrix{T}  
    b1::Vector{T}  
    W2::Matrix{T}  
    b2::Vector{T}  
end

function forward_pass(model::NeuralNetwork, x) 
    a1 = x 
    z1 = model.W1 * a1 .+ model.b1
    a2 = max.(0, z1)                # ReLU
    z2 = model.W2 * a2 .+ model.b2
    a3 = exp.(z2) ./ sum(exp.(z2))  # Softmax
    return a3
end

function softmax(x)
    return exp.(x) ./ sum(exp.(x))
end 

function softmax_derivative(x)
    s = exp.(x) ./ sum(exp.(x))     
    return Diagonal(s) .- s * s'     
end

# relu_derivative es una matriz cuadrada diagonal, nos quedamos con la diagonal.
function relu_derivative(x)
    return Float64.(x .> 0)
end


# Backpropagation
function grad(model::NeuralNetwork, x, y) 
    # Evitamos que las soluciones exploten.
    tol = 10e-8
    
    a1      = x
    y_hat   = forward_pass(model, x) 
    a2      = max.(0, model.W1*a1 + model.b1)

    # delta_2 = softmax_derivative(model.W2 * a2 .+ model.b2)' * (y ./ (y_hat .+ tol))
    delta_2 = y_hat .- y 
    delta_1 = relu_derivative(model.W1 * a1 .+ model.b1) .* (model.W2' * delta_2)
    
    # Derivada parcial respecto de W_1.
    W1_part = delta_1 * a1'
    
    # Derivada parcial respecto de b_1    
    b1_part = delta_1

    # Derivada parcial respecto de W_2
    W2_part = delta_2 * a2'

    # Derivada parcial respecto de b_2
    b2_part = delta_2

    return (W1_part, b1_part, W2_part, b2_part)

end

function mean_tuple(d::AbstractArray{<:Tuple})
    return Tuple([mean([d[k][i] for k in eachindex(d)]) for i in 1:length(d[1])])
end

function grad_total(model, grad, X, y) 
    return mean_tuple([grad(model, X[:,k], y[:,k]) for k in 1:size(X,2)])
end
