#!/bin/bash
cd ~
wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.2-linux-x86_64.tar.gz
tar -xf julia-1.0.2-linux-x86_64.tar.gz
ln -s ~/julia-1.0.2-linux-x86_64.tar.gz/bin/julia /usr/bin/julia
cd ~/CovariateAdversaries
julia ./pkggrab.jl
julia ./mlp.jl
