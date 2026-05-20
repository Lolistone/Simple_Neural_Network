using LinearAlgebra, RDatasets, Random, Statistics, Plots

include("utils_tools.jl")       
include("neural_network.jl")    
include("gradient_descent.jl")  
include("test.jl")        

# Importamos el dataset. 
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = iris.Species

# Miramos las 10 primeras filas y los tipos que hay.
show(first(iris, 10))
println("Classes:", join( unique(y), ","))

# Visualización de los datos.
p1 = scatter(iris.PetalLength, iris.PetalWidth, group=iris.Species, 
             xlabel="Petal Length", ylabel="Petal Width", title="Petals")

p2 = scatter(iris.SepalLength, iris.SepalWidth, group=iris.Species, 
             xlabel="Sepal Length", ylabel="Sepal Width", title="Sepals")

savefig(plot(p1, p2, layout=(1,2), size=(900, 400), legend=:topleft), "iris.png")

# Preparamos nuestros datos de entrenamiento. Fijamos una seed para poder replicar.
X_train, y_train, X_test, y_test, classes = prepare_data(X, y, seed = 2012)

# Chequeamso que todo esté en orden.
println("Predictors: ", size(X_train, 1))  
println("Training set size: ", size(X_train, 2))

# Inicializamos nuestra red neuronal con pesos aleatorios.
model = NeuralNetwork(rand(5,4), rand(5), rand(3,5), rand(3))

# Creamos un optimizador. En este caso, el método de descenso por gradiente.
opt = GradientDescent(0.1)

# Entrenamos la red neuronal.
model, loss_train, loss_val, best_iter = train!(opt, model, X_train, y_train, X_test, y_test, 10000)

p = results(loss_train, loss_val, best_iter)

# Muestra los resultados
savefig(plot(p, layout=(1,2), size=(900, 400), legend=:topleft), "loss.png")

# Medimos la accuracy del modelo.
test(model, X_test, y_test)
