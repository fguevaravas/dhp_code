# # Example 2: uniqueness with one single boundary condition
# Here we show a graph where one single boundary condition is sufficient
# to recover conductivities from power data in the linearized sense.

# ## Graph setup
# First we setup the graph, boundary conditions and 
# graph Laplacian
using Plots, LinearAlgebra, Test
x = vcat([cos(2π*i/3) for i=0:2],0)
y = vcat([sin(2π*i/3) for i=0:2],0)
∇ = [1 0 0 -1
     0 1 0 -1
     0 0 1 -1 ]
σ = ones(3);
𝐁 = [1,2,3];
𝐈 = [4];
f1 = [ 1, 0,0];
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

# ## Solve the Dirichlet problem
u1 = zeros(4)
u1[𝐁] = f1;
u1[𝐈] = -L[𝐈,𝐈]\(L[𝐈,𝐁]*f1)
@test u1[𝐈][1] ≈ 1/3
