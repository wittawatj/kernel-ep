#!/bin/bash 

# Break the paper into the main part and its supplementary material
pdftk kernel_ep_uai2015.pdf cat 1-10 output kernel_ep_uai2015_main.pdf
pdftk kernel_ep_uai2015.pdf cat 11-end output kernel_ep_uai2015_sup.pdf

