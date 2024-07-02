# # An illustration of Theorem and Proposition 1
# Here we simulate thermal noise currents and show that the problem
# is equivalent to a deterministic problem
# ## Graph setup
# Define graph and graph Laplacian
using Plots, LinearAlgebra, Test, Random
⊗ = kron
Nx = 5; Ny = 5; # number of nodes
x = (0:(Nx-1))*ones(1,Ny)
y = ones(Nx)*((0:(Ny-1))') 
D(N) = [ (i+1==j) - (i==j) for i=1:N-1,j=1:N]
# diagonal edges
idx = LinearIndices(x)
idxm1 = LinearIndices(x[1:Nx-1,1:Ny-1])
DE = zeros((Nx-1)*(Ny-1),Nx*Ny)
for i=1:(Nx-1), j=1:(Ny-1)
    DE[idxm1[i,j],idx[i,j]] = -1; DE[idxm1[i,j],idx[i+1,j+1]]=1;
end
∇ = [ I(Ny) ⊗ D(Nx) # horizontal edges
      D(Ny) ⊗ I(Nx) # vertical edges
     DE ]

𝐁 = findall( (x[:].==0) .| (x[:].==Nx-1) .| (y[:].==0) .| (y[:].==Ny-1) )
𝐈 = setdiff(1:Nx*Ny,𝐁)
n𝐈 =length(𝐈); n𝐁 = length(𝐁); 
n𝐄, n𝐕 = size(∇)

#f = (y[𝐁] .== Ny-1) - (y[𝐁] .== 0) # boundary condition
f = x[𝐁] + y[𝐁]/2
σ = ones(n𝐄)

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

# ## Generate random noise currents
# Note: this code stores all realizations and is un-necessarily memory intensive
#Random.seed!(17) # initialize seed
#κ = 1.380649e−23 # J/K
κ = π
T0 = 1; δT = 1e6; # background temp and perturbation
Nrel = 10000; # number of realizations

L = ∇'*diagm(σ)*∇ # Laplacian
udet = zeros(n𝐕)
udet[𝐁] = f
udet[𝐈] = - L[𝐈,𝐈]\(L[𝐈,𝐁]*udet[𝐁])

## background temperature experiment
m0 = zeros(Nrel)
for k ∈ 1:Nrel
    Jstd = sqrt.((κ/π)*T0*σ)
    J0 = Jstd.*randn(n𝐄)
    u0𝐈 = L[𝐈,𝐈]\((∇'*J0)[𝐈])
    g0 = L[𝐁,𝐈]*u0𝐈
    m0[k] = f'*g0
end
fg0 = var(m0)

## perturbed experiment
me = zeros(n𝐄,Nrel)
for e ∈ 1:n𝐄
   Jstd = sqrt.((κ/π)*[(T0+δT*(j==e))*σ[j] for j=1:n𝐄])
   for k ∈ 1:Nrel  
    Je = Jstd.*randn(n𝐄)
    ue𝐈 = L[𝐈,𝐈]\((∇'*Je)[𝐈])
    ge = L[𝐁,𝐈]*ue𝐈
    me[e,k] = f'*ge
   end
end
fge = var(me,dims=2)

stoch = fge .- fg0
deter = (δT*κ/π)*σ.*abs2.(∇*udet)

println(stoch)
println(deter)

plot(stoch,label="stoch")
plot!(deter,label="deter")