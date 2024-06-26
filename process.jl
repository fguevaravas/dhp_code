import Literate

inputs = ["example2","example3"]
for i ∈ inputs
  Literate.notebook("$(i).jl",i)
end
