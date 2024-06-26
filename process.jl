import Literate

inputs = ["example2","example3","example4"]
for i ∈ inputs
  Literate.notebook("$(i).jl",i)
end
