abstract type Optimizer end 

mutable struct GradientDescent <: Optimizer
    alpha::Float64
end

function step!(model::NeuralNetwork, grad, alpha) 
    model.W1 .-= alpha .* grad[1]
    model.b1 .-= alpha .* grad[2]
    model.W2 .-= alpha .* grad[3]
    model.b2 .-= alpha .* grad[4]
end

function train!(opt::GradientDescent, model::NeuralNetwork, X, y, iterations::Int)
    for _ in 1:iterations
        g = grad_total(model, grad, X, y)
        step!(model, g, opt.alpha)
    end
end
