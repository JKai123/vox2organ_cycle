#!/bin/bash

pdflatex draw_network.tex

rm *.aux *.log

xdg-open draw_network.pdf
