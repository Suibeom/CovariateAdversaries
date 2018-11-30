using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold,mse, crossentropy, throttle
using Base.Iterators: repeated
using Juno: @progress

using CuArrays

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()[1:1000]
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()[1:1000]
# One-hot-encode the labels
Y = onehotbatch(mod.(labels,2), 0:1) |> gpu
N=32
encoder = Dense(28^2, N, leakyrelu) |> gpu
decoder = Dense(N, 28^2, leakyrelu) |> gpu

a = Chain(encoder, decoder)

aloss(x,y) = mse(a(x), x)

pretrainloss(x,y) = crossentropy(m(x),y)
function nansafemin(a,b)
  if a == NaN
    return b
  elseif b == NaN
    return a
  end
  return min(a,b)
end

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 2),
  softmax) |> gpu
clippedloss(x, y) = nansafemin(mse(m(a(x)), y),crossentropy(m(a(x)),y))
loss(x, y) = min(clippedloss(x, y),clippedloss(x, -1*y.+1))

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

dataset = repeated((X, Y), 2)
dataseta = repeated((X,Y),2)

opt1 = ADAM(params(a))
opt2 = ADAM(params(m))

diffloss(x, y) = aloss(x,y) - 2*loss(x,y)
function evalcb()
   @show(diffloss(X, Y))
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

#println("Pretraining adversary")
#Flux.train!(pretrainloss, dataseta, opt2, cb=throttle(advcb,10))
#Flux.train!(aloss, dataset,opt2,cb=throttle(advcb,10))

@progress for k in 1:2000
@info "Epoch $k"
for i = 1:20
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
Flux.train!(loss, dataseta, opt2, cb=throttle(advcb,10))
end

save("sample"*lpad(string(k),4,string(0))*".png", sample())
end
end
