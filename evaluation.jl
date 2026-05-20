# Dada la probabilidad que devuelve softmax, vemos a que clase corresponde.
function predict(model::NeuralNetwork, x)
    probs = forward_pass(model, x)
    return argmax(probs)
end

# Accuracy
function test(model::NeuralNetwork, X_test, y_test)
    n = size(X_test, 2)
    correct = 0

    for k in 1:n
        y_pred_idx  = predict(model, X_test[:, k])
        y_true_idx  = argmax(y_test[:, k])   

        if y_pred_idx == y_true_idx
            correct += 1
        end
    end

    accuracy = correct / n

    println("─────────────────────────────────")
    println("Test set:  $n muestras")
    println("Correctas: $correct")
    println("Accuracy:  $(round(accuracy * 100, digits=2))%")
    println("─────────────────────────────────")

    return accuracy
end

# Gráfico del error para entrenamiento y testeo.
function results(loss_train::Vector{Float64}, loss_val::Vector{Float64}, best_iter::Int)
    iters = 1:length(loss_train)
 
    p = plot(iters, loss_train,
        label     = "Train loss",
        xlabel    = "Iteración",
        ylabel    = "Cross-Entropy",
        title     = "Pérdida por iteración",
        linewidth = 2,
        color     = :steelblue)
 
    plot!(p, iters, loss_val,
        label     = "Validation loss",
        linewidth = 2,
        color     = :tomato,
        legend    = :topright)
 
    vline!(p, [best_iter],
        label     = "Early stop (iter $best_iter)",
        linestyle = :dash,
        color     = :gray,
        linewidth = 1.5)
    
    return p
end
