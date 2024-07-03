# # Example : complex conductivities
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
σr = [1,2] ⊗ ones(6)
σi = [1/2,1] ⊗ ones(6)
ω = 1/2

𝐁 = [1,2,3,4,6,7,8,9];
𝐈 = [5];
n𝐈 =length(𝐈); n𝐁 = length(𝐁); 
n𝐄, n𝐕 = size(∇)

# f1 = [ 1, 1, 1, 1/2, 1/2, 0, 0, 0];
# f2 = [ 1, 1/2, 0, 1, 0, 1, 1/2, 0];
f1 = [1, 2, 3, 4, 5, 6, 7, 8]
f2 = [1, 2, 3, 2, 2, 1, 2, 3]

L(σ) = ∇'*diagm(σ)*∇;

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

# ## Jacobian and injectivity test
R𝐈= I(n𝐕)[𝐈,:] # restriction to interior nodes
R𝐁= I(n𝐕)[𝐁,:] # restriction to boundary nodes

## Dirichlet problem solve
function dirsolve(σ,f)
  u = zeros(n𝐕)
  u[𝐁] = f
  u[𝐈] = -L(σ)[𝐈,𝐈]\(L(σ)[𝐈,𝐁]*f)
  return u
end

## Forward problem and Jacobian
ℒ(σr,σi,u0,u1,u1b) = [ 
            (L(σr)*u0)[𝐈]
            (L(σr+im*ω*σi)*u1)[𝐈]
            (L(σr-im*ω*σi)*u1b)[𝐈]
            u0[𝐁]
            u1[𝐁]
            u1b[𝐁]
]
ℳ(σr,σi,u0,u1,u1b) = 
[ σr .* abs2.(∇*u0)
  σr .* (∇*u1) .* (∇*u1b) ]

Dℒ(σr,σi,u0,u1,u1b) = [ 
    R𝐈*∇'*diagm(∇*u0)   zeros(n𝐈,n𝐄)            R𝐈*L(σr)      zeros(n𝐈,n𝐕)     zeros(n𝐈,n𝐕)
    R𝐈*∇'*diagm(∇*u1)   im*ω*R𝐈*∇'*diagm(∇*u1)  zeros(n𝐈,n𝐕)  R𝐈*L(σr+im*ω*σi) zeros(n𝐈,n𝐕)
    R𝐈*∇'*diagm(∇*u1b) -im*ω*R𝐈*∇'*diagm(∇*u1b) zeros(n𝐈,n𝐕)  zeros(n𝐈,n𝐕)     R𝐈*L(σr-im*ω*σi)
    zeros(n𝐁,n𝐄)        zeros(n𝐁,n𝐄)           R𝐁            zeros(n𝐁,n𝐕)    zeros(n𝐁,n𝐕)
    zeros(n𝐁,n𝐄)        zeros(n𝐁,n𝐄)           zeros(n𝐁,n𝐕)  R𝐁              zeros(n𝐁,n𝐕)
    zeros(n𝐁,n𝐄)        zeros(n𝐁,n𝐄)           zeros(n𝐁,n𝐕)  zeros(n𝐁,n𝐕)    R𝐁 
]

Dℳ(σr,σi,u0,u1,u1b) = [
 diagm(abs2.(∇*u0))     zeros(n𝐄,n𝐄) 2diagm(σr .* (∇*u0))*∇ zeros(n𝐄,n𝐕)           zeros(n𝐄,n𝐕)
 diagm((∇*u1).*(∇*u1b)) zeros(n𝐄,n𝐄) zeros(n𝐄,n𝐕)           diagm(σr .* (∇*u1b))*∇ diagm(σr .* (∇*u1))*∇
]

function jacobian(σr,σi)
  u0_1 = dirsolve(σr,f1)
  u1_1 = dirsolve(σr+im*ω*σi,f1)
  u0_2 = dirsolve(σr,f2)
  u1_2 = dirsolve(σr+im*ω*σi,f2)
  Dℒ1 = Dℒ(σr,σi,u0_1,u1_1,conj(u1_1))
  Dℒ2 = Dℒ(σr,σi,u0_2,u1_2,conj(u1_2))
  Dℳ1 = Dℳ(σr,σi,u0_1,u1_1,conj(u1_1))
  Dℳ2 = Dℳ(σr,σi,u0_2,u1_2,conj(u1_2))
  𝒜 = [ Dℒ1[:,1:2n𝐄] Dℒ1[:,2n𝐄 .+ (1:3n𝐕)] zeros(3n𝐕,3n𝐕)
        Dℒ2[:,1:2n𝐄] zeros(3n𝐕,3n𝐕)        Dℒ2[:,2n𝐄 .+ (1:3n𝐕)]
        Dℳ1[:,1:2n𝐄] Dℳ1[:,2n𝐄 .+ (1:3n𝐕)] zeros(2n𝐄,3n𝐕)
        Dℳ2[:,1:2n𝐄] zeros(2n𝐄,3n𝐕)        Dℳ2[:,2n𝐄 .+ (1:3n𝐕)]
      ]
  return 𝒜
end

# ## Note: this seems impossible
# after more inspection it seems this problem does not admit a unique solution.
J = jacobian(σr,σi)
heatmap(abs.(J).>1e-6)