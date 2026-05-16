using LinearAlgebra, RDatasets, Random, Statistics, Plots

# Importamos las que implementamos en los otros archivos.
# main.jl
include("utils_tools.jl")       # sin dependencias
include("neural_network.jl")    # sin dependencias
include("gradient_descent.jl")  # usa NeuralNetwork
include("test.jl")        # usa NeuralNetwork + forward_pass

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

p3 = scatter(iris.PetalLength, iris.SepalWidth, iris.SepalLength, group=iris.Species, 
             xlabel="Petal Length", ylabel="Sepal Width", zlabel="Sepal Length", 
             title="3D Scatter of Iris Dataset")

display(plot(p1, p2, layout=(1,2), size=(900, 400), legend=:topleft))
display(p3)

# Preparamos nuestros datos de entrenamiento.
X_train, y_train, X_test, y_test, classes = prepare_data(X, y)

# Chequeamso que todo esté en orden.
println("Predictors: ", size(X_train, 1))  
println("Training set size: ", size(X_train, 2))

# Inicializamos nuestra red neuronal con pesos aleatorios.
model = NeuralNetwork(rand(5,4), rand(5), rand(3,5), rand(3))

# Creamos un optimizador. En este caso, el método de descenso por gradiente.
opt = GradientDescent(0.1)

# Entrenamos la red neuronal.
train!(opt, model, X_train, y_train, 100)

# Testeamos el modelo.
accuracy = test(model, X_test, y_test)
