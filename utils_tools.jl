function split(X, y; dims=1, ratio_train=0.8)
    n = length(y)
	
    n_train = round(Int, ratio_train*n) # Round the cutoff to an integer
    i_rand = randperm(n)				# Generate a random permutation of indices from 1 to n
    i_train = i_rand[1:n_train] 		# Use the first n_train indices for training
    i_test = i_rand[n_train+1:end] 		# Everything else is for testing

	X_train = X[i_train,:]
	y_train = y[i_train]
	X_test  = X[i_test, :]
	y_test = y[i_test]
    
    return X_train, y_train, X_test, y_test

end

function normalize(X_train, X_test; dims=1)
    col_mean = mean(X_train; dims)
    col_std = std(X_train; dims)

    return (X_train .- col_mean) ./ col_std, (X_test .- col_mean) ./ col_std
end

function onehot(y, classes)
	y_onehot = zeros(length(classes), length(y))
	
	for i in eachindex(y)
		y_onehot[:,i] = y[i].== classes
	end
	return y_onehot
end

function prepare_data(X, y; do_normal=true, do_onehot=true, kwargs...)
    X_train, y_train, X_test, y_test = split(X, y)

    if do_normal
        X_train, X_test = normalize(X_train, X_test; kwargs...)
    end

    classes = unique(y)

    if do_onehot
        y_train = onehot(y_train, classes)
        y_test = onehot(y_test, classes)
    end

    return X_train', y_train, X_test', y_test, classes
end

