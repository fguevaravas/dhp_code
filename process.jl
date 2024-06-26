import Literate

inputs = ["example3"]
for i ∈ inputs
  Literate.notebook("$(i).jl",i)
end
