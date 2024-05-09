<h1 align='center'>Cod for the paper "Single-seed generation of Brownian paths and integrals
for adaptive and high order SDE solvers"</h1>
<h2 align='center'>by Andraž Jelinčič, James Foster and Ptrick Kidger</h2>

The paper is available at TBA.

This repository is based on the Diffrax package by Patrick Kidger, which is available at [github.com/patrick-kidger/diffrax](https://github.com/patrick-kidger/diffrax).
The documentation for Diffrax can be found at [docs.kidger.site/diffrax](https://docs.kidger.site/diffrax).

The code for the CIR model example is in `notebooks/cir_model.ipynb`.

Neal's funnel example is split between two notebooks:
 - `notebooks/langevin.ipynb` contains the computation of strong orders of solvers on the Neal's funnel Langevin SDE
 - `notebooks/mcmc.ipynb` compares the output distribution of Langevin Monte Carlo against the No-U-Turn Sampler on the Neal's funnel model
