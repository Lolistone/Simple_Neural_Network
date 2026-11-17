# Funciones auxiliares para preparar los datos.

# "Mezclamos" y dividimos el dataset en entrenamiento, validación y testeo.
# Validación se usa para el early stopping y testeo solo para la evaluación final.
function split_data(X, y; ratio_train = 0.6, ratio_val = 0.2, seed = 2012)
    n = length(y)
    n_train = round(Int, ratio_train * n)
    n_val   = round(Int, ratio_val * n)

    i_rand  = randperm(Xoshiro(seed), n)
    i_train = i_rand[1:n_train]
    i_val   = i_rand[n_train+1:n_train+n_val]
    i_test  = i_rand[n_train+n_val+1:end]

    return X[i_train, :], y[i_train], X[i_val, :], y[i_val], X[i_test, :], y[i_test]
end

# Normalizamos los datos (media y desvío calculados solo con entrenamiento).
function standardize(X_train, Xs...; dims = 1)
    col_mean = mean(X_train; dims)
    col_std  = std(X_train; dims)
    scale(X) = (X .- col_mean) ./ col_std

    return scale(X_train), map(scale, Xs)...
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
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, seed = seed)

    if do_normal
        X_train, X_val, X_test = standardize(X_train, X_val, X_test; kwargs...)
    end

    classes = unique(y)

    if do_onehot
        y_train = onehot(y_train, classes)
        y_val   = onehot(y_val, classes)
        y_test  = onehot(y_test, classes)
    end

    # Un dato por columna.
    return X_train', y_train, X_val', y_val, X_test', y_test, classes
end
