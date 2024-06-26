# # Example of non-uniqueness
# Here we show a graph where the linearization of the inverse
# problem of finding the conductivities from power measurements
# does not admit a unique solution.
# 
# ## Graph setup
# First we setup the graph, boundary conditions and 
# graph Laplacian
using Plots, LinearAlgebra, Test
⊗ = kron
x =  ones(3,1)*[0 1 2]; 
y = [0;1;2]*ones(1,3);  
D = [ -1 1 0
       0 -1 1 ]
∇ = [ D ⊗ I(3) ;  I(3) ⊗ D ]
σ = ones(12);
𝐁 = [1,2,3,4,6,7,8,9];
𝐈 = [5];

f1 = [ 1, 1, 1, 1/2, 1/2, 0, 0, 0];
f2 = [ 1, 1/2, 0, 1, 0, 1, 1/2, 0];

L = ∇'*diagm(σ)*∇;

# ## Graph plotting
scatter(x[𝐈],y[𝐈], color="blue",markersize=7,markeralpha=0.5);
scatter!(x[𝐁],y[𝐁], color="red",markersize=7,markeralpha=0.5);
for i in eachindex(x)
  annotate!(x[i], y[i], text("$i",:black,:top,:right,9))
end
for (i, r) in enumerate(eachrow(∇))
  i1, i2 = findall(abs.(r) .> 0)
  plot!([x[i1], x[i2]], [y[i1], y[i2]], color="black", lw=1)
  annotate!((x[i1]+x[i2])/2, (y[i1]+y[i2])/2, text("$i", :black, :top,:right,9))
end
p=plot!(legend=:none, aspect_ratio=:equal, axis=false, grid=false,size=(300,300))

# ## Solve the Dirichlet problems
u1 = zeros(9)
u1[𝐁] = f1;
u1[𝐈] = -L[𝐈,𝐈]\(L[𝐈,𝐁]*f1);

u2 = zeros(9)
u2[𝐁] = f2;
u2[𝐈] = -L[𝐈,𝐈]\(L[𝐈,𝐁]*f2);

# ## Non-uniqueness
# We give an example of a family of $\delta\sigma$ of the form
# $\delta\sigma = c_1 v_1 + c_2 v_2$ which cannot recovered from power
# measurements in the linearized sense. We test this by plugging  this
# (non-zero) $\delta\sigma, \delta u_1, \delta u_2$ into the linearized
# problem and showing we get zero.
v1 = 4*[0,1,0,0,-1,0,0,0,0,0,0,0]
v2 = 4*[0,0,0,0,0,0,0,0,1,-1,0,0]
c1,c2 = randn(2)
δσ = c1*v1 + c2*v2
δu1 = zeros(9); δu1[𝐈] .= c1;
δu2 = zeros(9); δu2[𝐈] .= c2;

## test linearized problem equations give zero
@testset begin
  @test norm((∇'*( (∇*u1) .* δσ ) + L*δu1)[𝐈]) ≈ 0
  @test norm((∇'*( (∇*u2) .* δσ ) + L*δu2)[𝐈]) ≈ 0
  @test norm((∇*u1).^2 .* δσ + 2σ.*(∇*u1).*(∇*δu1)) ≈ 0
  @test norm((∇*u2).^2 .* δσ + 2σ.*(∇*u2).*(∇*δu2)) ≈ 0
end;
