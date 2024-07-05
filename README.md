# Code accompanying ``Discrete inverse problems with internal functionals''
This code illustrates examples from the paper ``Discrete inverse problems with internal functionals'' by Marcus Corbett, Fernando Guevara Vasquez, Alexander Royzman and Guang Yang (_insert arXiV link_)

The code that is provided is in Julia and it can be used to generate annotated HTML or interactive Jupyter notebooks.

Instructions:
* Change directory to this repository
* run Julia and activate the current environment (`] activate .`). This will install any needed dependencies
* in the Julia prompt: `include("process.jl")` to convert all the `.jl` files into interactive Jupyter notebooks

A list of the files that will be converted to Jupyter notebooks
* `example1.jl`: one single Dirichlet boundary condition is needed for local uniqueness
* `example2.jl`: a case where the linearized inverse problem is not injective
* `example3.jl`: a case where the linearized inverse problem is ill-conditioned
* `example4.jl`: an illustration of the complex conductivity result
* `thermal_noise.jl`: code comparing thermal noise and dissipated power
* `gauss_newton.jl`: an illustration of how to use Gauss-Newton method to solve the problem

## Funding
This project was partially funded by the National Science Foundation Grant DMS-2008610.

## License
Unless noted otherwise, the code is released under the following license (see also `LICENSE` file)
```
BSD 3-Clause License

Copyright (c) 2024, Fernando Guevara Vasquez
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.