# Obs: La idea de implementarlo así es para a futuro considerar otros optimizadores.
abstract type Optimizer end

# Definimos el descenso por gradiente.
mutable struct GradientDescent <: Optimizer
    alpha::Float64
end

# Dar un paso del método.
function step!(model::NeuralNetwork, g, alpha)
    model.W1 .-= alpha .* g[1]
    model.b1 .-= alpha .* g[2]
    model.W2 .-= alpha .* g[3]
    model.b2 .-= alpha .* g[4]
end

# Copiamos los parámetros de un modelo en otro.
function restore!(model::NeuralNetwork, best::NeuralNetwork)
    model.W1 .= best.W1
    model.b1 .= best.b1
    model.W2 .= best.W2
    model.b2 .= best.b2
end

# Entrenamos el modelo, guardando a cada paso la función de pérdida
# tanto en los datos de entrenamiento como de validación (early stopping).
# Al terminar, el modelo queda con los pesos de la mejor iteración.
function train!(opt::GradientDescent, model::NeuralNetwork,
                X_train, y_train,
                X_val,   y_val,
                max_iter::Int;
                patience = 20)

    train_losses = Float64[]
    val_losses   = Float64[]
    best_val     = Inf
    best_iter    = 1
    best_model   = deepcopy(model)
    no_improve   = 0

    for iter in 1:max_iter
        l_train = loss_total(model, X_train, y_train)
        l_val   = loss_total(model, X_val,   y_val)
        push!(train_losses, l_train)
        push!(val_losses,   l_val)

        # Chequeamos si el modelo mejora o no.
        if l_val < best_val
            best_val   = l_val
            best_iter  = iter
            best_model = deepcopy(model)
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
        step!(model, grad(model, X_train, y_train), opt.alpha)
    end

    restore!(model, best_model)
    return model, train_losses, val_losses, best_iter
end
