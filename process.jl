import Literate

inputs = ["thermal_noise","example1","example2","example3","example4"]
for i ∈ inputs
  Literate.notebook("$(i).jl",i)
end
