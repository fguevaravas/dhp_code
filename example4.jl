# # Example 4: ill-posedness
# We give an example of a graph where the linearized inverse problem can be
# arbitrarily ill-posed. The reason for this is that there is an edge with
# gradient that is of order ε which can be made arbitrarily small.
using LinearAlgebra, Symbolics, Plots
@variables ε # for symbolic computations
∇ = [ 1 0 0 0 -1  0
      0 1 0 0 -1  0
      0 0 1 0  0 -1
      0 0 0 1  0 -1
      0 0 0 0  1 -1 ]
σ = [1+ε,1,1,1,1]
𝐁 = [1,2,3,4];
𝐈 = [5,6];
f1 = [ 1, 0, 1,0];
L= ∇'*diagm(σ)*∇;
u1 = zeros(Num,6)
u1[𝐁] = f1;
u1[𝐈] = -L[𝐈,𝐈]\(L[𝐈,𝐁]*f1);
u1 = simplify.(u1);
println(u1[𝐈])

# ## Calculate condition number
# The condition seems to be inversely proportional to ε, so the inverse problem becomes increasingly ill-posed for smaller ε.
R𝐈 = I(6)[𝐈,:] # restriction operator to 𝐈
function condest(ε)
    σ = [1+ε,1,1,1,1]
    L= ∇'*diagm(σ)*∇;
    u1 = zeros(6)
    u1[𝐁] = f1;
    u1[𝐈] = -L[𝐈,𝐈]\(L[𝐈,𝐁]*f1);
    ## Jacobian
    DH = [diagm((∇*u1).^2) diagm(2σ.*(∇*u1))*∇*R𝐈'
          (∇'*diagm(∇*u1))[𝐈,:] L[𝐈,𝐈] ]
    cond(DH)
end
εs = 10.0 .^ (-1:-0.5:-16)
cs = condest.(εs)
plot(εs,cs,xaxis=:log,yaxis=:log,
xlabel="ε",
ylabel="Jacobian conditioning",legend=:none)