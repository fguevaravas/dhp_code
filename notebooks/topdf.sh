#!/bin/bash
# This script converts the generated Jupyter notebooks to PDF
# A plain vanilla jupyter installation may not be enough to have this
# run. Try:
#   pip install nbconvert webpdf playwright
#   playwright install
NOTEBOOKS="example1.ipynb example2.ipynb example3.ipynb example4.ipynb
gauss_newton.ipynb thermal_noise.ipynb"

for n in $NOTEBOOKS;
do
 jupyter nbconvert --to=webpdf $n
done
