# # Reconstructions using Gauss-Newton method
# Here we give an example of reconstructing the conductivity by successive linearization
# ## Graph setup
# Define graph and graph Laplacian
using Plots, LinearAlgebra, Test
⊗ = kron
Nx = 10; Ny = 10; # number of nodes
x = (0:(Nx-1))*ones(1,Ny)/(Nx-1)
y = ones(Nx)*((0:(Ny-1))') /(Ny-1)
D(N) = [ (i+1==j) - (i==j) for i=1:N-1,j=1:N]
## Discrete gradient
∇ = [ I(Ny) ⊗ D(Nx) # horizontal edges
      D(Ny) ⊗ I(Nx) # vertical edges
]
𝐁 = findall( (x[:].==0) .| (x[:].==Nx-1) .| (y[:].==0) .| (y[:].==Ny-1))
𝐈 = setdiff(1:Nx*Ny,𝐁)
n𝐈 =length(𝐈); n𝐁 = length(𝐁); 
n𝐄, n𝐕 = size(∇)
R𝐈 = I(n𝐕)[𝐈,:]  # restriction to interior nodes

x𝐄 = abs.(∇)*x[:]/2; y𝐄 = abs.(∇)*y[:]/2 # edge centers

indisk(c,r,x) = (x[1]-c[1])^2 + (x[2]-c[2])^2  <= r^2
σ_true =
   [ 1 + indisk((0.2,0.2),0.1,(x,y)) + 
         indisk((0.5,0.5),0.2,(x,y)) +
         indisk((0.75,0.6),0.2,(x,y)) 
    for  (x,y) ∈ zip(x𝐄,y𝐄) ]

L(σ) = ∇'*diagm(σ)*∇ # Laplacian

## Boundary conditions and data
# The Dirichlet boundary conditions we use are similar to $x + y$ and $x-y$ in the continuum. They are on purpose not aligned with the grid edges, so that we do not end up with edges where there are no currents flowing.
fs = [x[𝐁]+y[𝐁] x[𝐁]-y[𝐁]]; N = size(fs,2)

## Dirichlet problem solve
function dirsolve(σ,f)
    u = zeros(ComplexF64,n𝐕)
    u[𝐁] = f
    u[𝐈] = -L(σ)[𝐈,𝐈]\(L(σ)[𝐈,𝐁]*f)
    return u
end

## Plotting
# We plot the conductivity and the dissipated currents
function plot_edge_quantity(f;lw=6)
    p = plot()
    minf, maxf = extrema(f)
    for (i, r) in enumerate(eachrow(∇))
      i1, i2 = findall(abs.(r) .> 0)
      if (maxf-minf)/(maxf+minf) < 1e-6
        c = "black"
      else
        c = get(cgrad(:thermal),(f[i]-minf)/(maxf-minf))
      end
      plot!(p,[x[i1], x[i2]], [y[i1], y[i2]], linecolor=c, lw=lw)
    end
    plot!(p,legend=:none, aspect_ratio=:equal, axis=false, grid=false)
    return p
  end

  p = plot(
  plot_edge_quantity(σ,lw=4),
  plot_edge_quantity(Hs[1],lw=4),
  plot_edge_quantity(Hs[2],lw=4), 
  layout=grid(1,3) 
  )

# ## Jacobian computation
## Forward problem and Jacobian for one measurement
ℒ(σ,u) = [ 
        (L(σ)*u0)[𝐈]
        u0[𝐁]
]
ℳ(σ,u) = σ .* abs2.(∇*u)

Dℒ(σ,u) = [ 
    R𝐈*∇'*diagm(∇*u)  R𝐈*L(σ)
    zeros(n𝐁,n𝐄)      R𝐁     
]

Dℳ(σ) = [ diagm(abs2.(∇*u0)) 2diagm(σ .* (∇*u0))*∇ ];
## Assemble forward map
fwd(σ,us) = [ vcat([ℒ(σ,us[:,j]) for j=1:N]...)
              vcat([ℳ(σ,us[:,j]) for j=1:N]...)
            ]

## Assemble rhs
b(fs,Hs) = [ vcat([R𝐁'*fs[:,j] for j=1:N]...)
             Hs[:]
    ]

## Assemble Jacobian and injectivity matrix for all boundary conditions
function jacobian(σ,us)  
    N = size(us,2) # number of Dirichlet boundary conditions
    ## Solve Dirichlet problems and calculate Jacobians for each boundary condition
    us = zeros(n𝐕,N)
    Dℒs = Vector{Any}(undef,N)
    Dℳs = Vector{Any}(undef,N)
    for j=1:N
        Dℒs[j] = Dℒ(σ,us[:,j])
        Dℳs[j] = Dℳ(σ,us[:,j])
    end

    ## Assemble full Jacobian
    𝒜 = zeros(N*n𝐕+N*n𝐄,n𝐄+N*n𝐕)
    for j=1:N
        𝒜[ (j-1)*n𝐕 .+ (1:n𝐕)        , 1:n𝐄 ] = Dℒs[j][:,1:n𝐄]
        𝒜[ N*n𝐕 + (j-1)*n𝐄 .+ (1:n𝐄) , 1:n𝐄 ] = Dℳs[j][:,1:n𝐄]
        𝒜[ (j-1)*n𝐕 .+ (1:n𝐕)        , n𝐄 .+ (j-1)*n𝐕 .+ (1:n𝐕) ] = Dℒs[j][:,n𝐄 .+ (1:n𝐕)]
        𝒜[ N*n𝐕 .+ (j-1)*n𝐄 .+ (1:n𝐄), n𝐄 .+ (j-1)*n𝐕 .+ (1:n𝐕) ] = Dℳs[j][:,n𝐄 .+ (1:n𝐕)]
    end
    return 𝒜
end;

# ## Gauss-Newton method
# Here we solve the optimization problem
# $$
# \min_\sigma \| R(\sigma) \|^2 + \alpha^2 \| \sigma \|^2,
# $$
# where $R$ is the residual of a (nonlinear) system of equations describing the problem and $\alpha$ is a regularization parameter.
# The Gauss-Newton method consists of the update:
# $$
# \sigma^{(n+1)} = \sigma^{(n)} - (DR(\sigma)DR^T(\sigma) + \alpha^2 I)^{-1} R(\sigma) ,
# $$
# where $DF(\sigma)$ is the Jacobian of $F$ evaluated at $\sigma$.
function gauss_newton(R,DR,x0;maxiter=100,tol=1e-4,α=0)
    x = x0
    for n=1:maxiter
        J = DR(x)
        xnew = x - (J*J' + α^2*I)\R(x)
        norm(xnew - x)/norm(x) < tol && return xnew,n
        x = xnew
    end
    return x,maxiter
end

# ## Setup data and residual
unpack(x)  = (σ=x[1:n𝐄],us=reshape(x[(n𝐄+1):end]),n𝐕,N) # go from x to σ,us
pack(σ,us) = vcat(σ,vec(us)) # go from (σ,us) to x

## true data
us_true = hcat([ dirsolve(σ_true,f) for f ∈ eachcol(fs)]...)
Hs_true = hcat([ σ.*abs2.(∇*u) for f ∈ eachcol(us_true)]...)

R(x)  = fwd(unpack(x)...) - b(fs,Hs_true)
DR(x) = jacobian(unpack(x)...)


## test a Jacobian against Taylor's theorem
ϵs = 10.0 .^ (0:-0.5:-16)
jacobian_test(F,DF,x0,δx) =
 [ norm(F(x0 + ϵ*δx) - (F(x0) + ϵ*DF(x0)*δx))/ϵ^2/norm(δx) for ϵ ∈ ϵs ]

 plot(ϵs, jacobian_test(R,DR,ones(n𝐄),randn(n𝐄)))