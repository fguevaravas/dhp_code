# This file converts the scripts into Jupyter notebooks
import Literate

inputs = ["thermal_noise","example1","example2","example3","example4","gauss_newton"]
outdir = "notebooks"

for i ∈ inputs
  Literate.notebook("$(i).jl",outdir)
end
