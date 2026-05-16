function predict(model::NeuralNetwork, x)
    probs = forward_pass(model, x)
    return argmax(probs)
end

function test(model::NeuralNetwork, X_test, y_test)
    n = size(X_test, 2)
    correct = 0

    for k in 1:n
        y_pred_idx  = predict(model, X_test[:, k])
        y_true_idx  = argmax(y_test[:, k])   # índice de la clase real (one-hot)

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

function results()
end
