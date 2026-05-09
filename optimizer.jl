abstract type Optimizer end

# Descent methods step.
function step!(model::NeuralNetwork, grad, alpha) 
    model.W1 .-= alpha .* grad.W1
    model.b1 .-= alpha .* grad.b1
    model.W2 .-= alpha .* grad.W2
    model.b2 .-= alpha .* grad.b2
end

# Gradient Descent
mutable struct GradientDescent <: Optimizer
    alpha::Float64
end

function train!(opt::GradientDescent, model::NeuralNetwork, X, y, iterations::Int)
    for _ in 1:iterations
        g = grad(model, X, y)
        step!(model, g, opt.alpha)
    end
end

# Armijo line search
mutable struct Armijo <: Optimizer
    alpha::Float64    
    eps_1::Float64    
    eps_2::Float64    
end

function armijo_step(opt::Armijo, model::NeuralNetwork, X, y, g) 
    ### Implementar !!!!

end

function train!(opt::Armijo, model::NeuralNetwork, X, y, iterations::Int)
    for _ in 1:iterations
        g = grad(model, X, y)
        alpha = armijo_step(opt, model, X, y, g)
        step!(model, g, alpha)
    end
end

# Newton
mutable struct Newton <: Optimizer
    alpha::Float64
end

function train!(opt::Newton, model::NeuralNetwork, X, y, iterations::Int)
    for _ in 1:iterations
        g = grad(model, X, y)
        H = hess(model, X, y)
        d = H \ g          
        step!(model, d, opt.alpha)
    end
end

# BFGS 
mutable struct BFGS <: Optimizer
    alpha::Float64
end

function bfgs_update(H_inv, s, y) end

function train!(opt::BFGS, model::NeuralNetwork, X, y, iterations::Int) end
