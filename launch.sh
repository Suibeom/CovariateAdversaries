#!/bin/bash
cd ~
wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.2-linux-x86_64.tar.gz
tar -xf julia-1.0.2-linux-x86_64.tar.gz
sudo ln -s ~/julia-1.0.2/bin/julia /bin/julia
cd ~/CovariateAdversaries
julia ./pkggrab.jl
julia ./mlp.jl
