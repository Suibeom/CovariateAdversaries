using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold,mse, crossentropy, throttle
using Base.Iterators: repeated, partition
using Juno: @progress

using CuArrays

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()[1:1000]
X = hcat(float.(reshape.(imgs, :))...) |> gpu
labels = MNIST.labels()[1:1000]
Y = onehotbatch(mod.(labels,2), 0:1) |> gpu
N=64
encoder = Dense(28^2, N, leakyrelu) |> gpu
decoder = Dense(N, 28^2, leakyrelu) |> gpu

a = Chain(encoder, decoder)

aloss(x,y) = mse(a(x), x)

pretrainloss(x,y) = crossentropy(m(x),y)
function nansafemin(a,b) # First should be bounded!!
  if isnan(b) 
    println("NaN!")
    return a
  end
  return min(a,b)
end

m = Chain(
  Dense(28^2, N, relu),
  Dense(N, 2),
  softmax) |> gpu
#clippedloss(x, y) = nansafemin(mse(m(a(x)), y),crossentropy(m(a(x)),y))
#loss(x, y) = min(clippedloss(x, y),clippedloss(x, -1*y.+1))
<<<<<<< HEAD

=======
loss(x, y) = min(mse(m(a(x)), y),mse(m(a(x)), -1*y.+1)) + 0.01*min(mse(m(x), y),mse(m(x), -1*y.+1))
ac(x, y) = mean(onecold(m(a(x))) .== onecold(y))
accuracy(x,y) = max(ac(x,y), 1-ac(x,y))
dataset = repeated((X, Y), 2)
dataseta = repeated((X,Y),2)
>>>>>>> d5b0eaedce4cf50b4876d930ed994900fa1725ee

opt1 = ADAM(params(a))

loss(x, y) = min(mse(m(a(x)), y),mse(m(a(x)), -1*y.+1)) + 0.1*min(mse(m(x), y),mse(m(x), -1*y.+1))
ac(x, y) = mean(onecold(m(a(x))) .== onecold(y))
accuracy(x,y) = max(ac(x,y), 1-ac(x,y))
diffloss(x, y) = aloss(x,y) - 2*loss(x,y)
function evalcb()
   @show(accuracy(X, Y))
   @show(aloss(X,Y))
end
A = 0
function advcb()
  @show(diffloss(X,Y))
  global A = diffloss(X,Y)
end
using Images
img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))
function sample()
  # 20 random digits
  before = [imgs[i] for i in rand(1:length(imgs), 20)]
  # Before and after images
  after = img.(map(x -> cpu(a)(float(vec(x))).data, before))
  # Stack them all together
  hcat(vcat.(before, after)...)
end
imgss = MNIST.images()
labelss=MNIST.labels()
@progress for k in 1:20000

imgs = deepcopy([imgss[b + (1000*mod(k,60))] for b in 1:1000])
X1 = hcat(float.(reshape.(imgs, :))...) |> gpu 
labels = deepcopy([labelss[b + (1000*mod(k,60))] for b in 1:1000])
Y1 = onehotbatch(mod.(labels,2), 0:1) |> gpu 
dataset = repeated((X1,Y1),2)
dataseta = dataset



m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 2),
  softmax) |> gpu
println("Pretraining adversary")
opt2 = ADAM(params(m))
for i in 1:100
Flux.train!(pretrainloss, dataseta, opt2 )
end
diffloss(x, y) = aloss(x,y) - 1*(1+(k/20))*loss(x,y)
@info "Epoch $k"
for i = 1:10
println("Training autoencoder")
Flux.train!(diffloss, dataset, opt1, cb = throttle(evalcb,10))
println("Training adversary")
A0 = advcb()

#Flux.train!(loss, dataseta, opt2, cb=throttle(advcb,10))
#while A0 > A #Train until you get back to at least as good as you had!
#Flux.train!(loss, dataseta, opt2, cb=throttle(advcb,10))
#end
#println("Breakpoint reached.")
for i in 1:10
Flux.train!(loss, dataseta, opt2)
end
GC.gc(); CuArrays.reclaim()
end
end
