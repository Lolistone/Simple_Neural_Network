# Obs: La idea de implementarlo así es para a futuro considerar otros optimizadores.
abstract type Optimizer end 

# Definimos el descenso por gradiente.
mutable struct GradientDescent <: Optimizer
    alpha::Float64
end

# Dar un paso del método.
function step!(model::NeuralNetwork, grad, alpha) 
    model.W1 .-= alpha .* grad[1]
    model.b1 .-= alpha .* grad[2]
    model.W2 .-= alpha .* grad[3]
    model.b2 .-= alpha .* grad[4]
end

# Entrenamos el modelo, guardando a cada paso la función de pérdida
# tanto en los datos de entrenamiento como de testeo. Devolvemos la mejor iteración.
function train!(opt::GradientDescent, model::NeuralNetwork,
                X_train, y_train,
                X_val,   y_val,
                max_iter::Int;
                patience = 20)

    train_losses = Float64[]
    val_losses   = Float64[]
    best_val     = Inf
    best_iter    = 1
    no_improve   = 0

    for iter in 1:max_iter

        l_train = loss_total(model, X_train, y_train)
        l_val   = loss_total(model, X_val,   y_val)
        push!(train_losses, l_train)
        push!(val_losses,   l_val)

        # Chequeamos si el modelo mejora o no.
        if l_val < best_val
            best_val  = l_val
            best_iter = iter
            no_improve = 0
        else
            no_improve += 1
        end

        # Si no mejora cortamos. 
        if no_improve >= patience
            println("Terminamos en la iteración: $iter (Mejor valor en la iteración: $best_iter)")
            break
        end

        # Paso del gradiente
        g = grad_total(model, grad, X_train, y_train)
        step!(model, g, opt.alpha)
    end

    return model, train_losses, val_losses, best_iter
end
