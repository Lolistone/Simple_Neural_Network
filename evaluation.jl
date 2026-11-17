# Dada la probabilidad que devuelve softmax, vemos a que clase corresponde.
predict(model::NeuralNetwork, x::AbstractVector) = argmax(forward_pass(model, x))
predict(model::NeuralNetwork, X::AbstractMatrix) = [argmax(p) for p in eachcol(forward_pass(model, X))]

# Accuracy
function accuracy(model::NeuralNetwork, X_test, y_test)
    n = size(X_test, 2)
    y_true  = [argmax(c) for c in eachcol(y_test)]
    correct = count(predict(model, X_test) .== y_true)
    acc = correct / n

    println("─────────────────────────────────")
    println("Test set:  $n muestras")
    println("Correctas: $correct")
    println("Accuracy:  $(round(acc * 100, digits=2))%")
    println("─────────────────────────────────")

    return acc
end

# Gráfico del error para entrenamiento y validación.
function results(loss_train, loss_val, best_iter::Int)
    iters = 1:length(loss_train)

    p = plot(iters, loss_train,
        label     = "Train loss",
        xlabel    = "Iteración",
        ylabel    = "Cross-Entropy",
        title     = "Pérdida por iteración",
        linewidth = 2,
        color     = :steelblue,
        size      = (700, 400))

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
