# Funciones auxiliares para preparar los datos.

# "Mezclamos" y dividimos el dataset en entrenamiento y testeo.
function train_test_split(X, y; ratio_train = 0.8, seed = 1212)
    n = length(y)
    n_train = round(Int, ratio_train * n)

    i_rand = randperm(Xoshiro(seed), n)
    i_train = i_rand[1:n_train]
    i_test  = i_rand[n_train+1:end]

    return X[i_train, :], y[i_train], X[i_test, :], y[i_test]
end

# Normalizamos los datos (media y desvío calculados solo con entrenamiento).
function standardize(X_train, X_test; dims = 1)
    col_mean = mean(X_train; dims)
    col_std  = std(X_train; dims)

    return (X_train .- col_mean) ./ col_std, (X_test .- col_mean) ./ col_std
end

# Representamos cada clase con un vector.
function onehot(y, classes)
    y_onehot = zeros(length(classes), length(y))

    for i in eachindex(y)
        y_onehot[:, i] = y[i] .== classes
    end
    return y_onehot
end

function prepare_data(X, y; do_normal = true, do_onehot = true, seed = 2012, kwargs...)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed = seed)

    if do_normal
        X_train, X_test = standardize(X_train, X_test; kwargs...)
    end

    classes = unique(y)

    if do_onehot
        y_train = onehot(y_train, classes)
        y_test  = onehot(y_test, classes)
    end

    # Un dato por columna.
    return X_train', y_train, X_test', y_test, classes
end
