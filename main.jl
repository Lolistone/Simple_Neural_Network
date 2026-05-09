using LinearAlgebra, RDatasets, Random, Statistics, Plots
include("utils_tools.jl")
include("optimizer.jl")
include("benchmarks.jl")
include("neural_network.jl")

# We will use the iris dataset to test our neural network implementation. 
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = iris.Species

# Let's check what this dataset looks like, showing the first 10 rows.
show(first(iris, 10))
println("Classes:", join( unique(y), ","))

p1 = scatter(iris.PetalLength, iris.PetalWidth, group=iris.Species, 
             xlabel="Petal Length", ylabel="Petal Width", title="Petals")

p2 = scatter(iris.SepalLength, iris.SepalWidth, group=iris.Species, 
             xlabel="Sepal Length", ylabel="Sepal Width", title="Sepals")

display(plot(p1, p2, layout=(1,2), size=(900, 400), legend=:topleft))

# 3D plot.
p3 = scatter(iris.PetalLength, iris.SepalWidth, iris.SepalLength, group=iris.Species, 
             xlabel="Petal Length", ylabel="Sepal Width", zlabel="Sepal Length", 
             title="3D Scatter of Iris Dataset")

display(p3)

model = NeuralNetwork(rand(5,4), rand(5), rand(3,5), rand(3))



