# # Example 4: complex conductivities
using Plots, LinearAlgebra, Test
⊗ = kron
x =  ones(3,1)*[0 1 2]; 
y = [0;1;2]*ones(1,3);  
D = [ -1 1 0
       0 -1 1 ]
∇ = [ D ⊗ I(3) ;  I(3) ⊗ D ]
σ = [1+1im,2+2im] ⊗ ones(6) 
𝐁 = [1,2,3,4,6,7,8,9];
𝐈 = [5];
n𝐈 =length(𝐈); n𝐁 = length(𝐁); 
n𝐄, n𝐕 = size(∇)

f1 = [ 1, 1, 1, 1/2, 1/2, 0, 0, 0];
f2 = [ 1, 1/2, 0, 1, 0, 1, 1/2, 0];

L(σ) = ∇'*diagm(σ)*∇;