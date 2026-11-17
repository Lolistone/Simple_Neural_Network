using LinearAlgebra, RDatasets, Random, Statistics, Plots

include("utils_tools.jl")
include("neural_network.jl")
include("gradient_descent.jl")
include("evaluation.jl")

# Una única seed para el split y la inicialización, así se puede replicar.
const SEED = 2012

# Importamos el dataset.
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = iris.Species

# Preparamos nuestros datos: entrenamiento, validación y testeo.
X_train, y_train, X_val, y_val, X_test, y_test, classes = prepare_data(X, y, seed = SEED)

# Chequeamos que todo esté en orden.
println("Predictors: ", size(X_train, 1))
println("Train / Val / Test: ", size(X_train, 2), " / ", size(X_val, 2), " / ", size(X_test, 2))

# Inicializamos nuestra red neuronal con pesos aleatorios.
model = init_network(size(X_train, 1), 5, length(classes), seed = SEED)

# Creamos un optimizador. En este caso, el método de descenso por gradiente.
opt = GradientDescent(0.1)

# Entrenamos la red neuronal (early stopping con el set de validación).
model, loss_train, loss_val, best_iter = train!(opt, model, X_train, y_train, X_val, y_val, 10000)

# Guardamos el gráfico de las pérdidas.
mkpath("img")
savefig(results(loss_train, loss_val, best_iter), "img/loss.png")

# Medimos la accuracy del modelo en datos que no usó para entrenar ni para frenar.
accuracy(model, X_test, y_test)
