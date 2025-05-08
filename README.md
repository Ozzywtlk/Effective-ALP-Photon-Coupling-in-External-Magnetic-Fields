# Effective-ALP-Photon-Coupling-in-External-Magnetic-Fields
This repository aims to provide supplementary files for generating the results from our paper "Effective ALP-Photon Coupling in External Magnetic Fields". 

The most important files in this rep are the .nb files, explicit formulation of the 114 terms mentioned in the paper, including all the coefficients, for all orders l. One can vary the l depending on the accuracy goal. The file "SUMMED_RESIDUES.nb" contain the closed form of the infinite sum over all the residues, for the case l=0. 

It is adviced to convert the results to C code, and compute the numerical integrals using the precompiled files. An example Python code, which imports the compiled .dll files and performs numerical integration over VEGAS is provided in the file "Numerical.py". For the integration over the imaginary axis, we do not provide a conversion to C code, as both VEGAS and Mathematica have similar performances.
