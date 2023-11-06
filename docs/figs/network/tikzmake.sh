#!/bin/bash


python arch.py $1
pdflatex $1.tex

rm *.aux *.log
# rm *.tex
# if [[ "$OSTYPE" == "darwin"* ]]; then
#     open $1.pdf
# else
#     xdg-open $1.pdf
# fi
