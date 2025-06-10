<h1 align='center'>Code for the paper "Single-seed generation of Brownian paths and integrals
for adaptive and high order SDE solvers"</h1>
<h2 align='center'>by Andraž Jelinčič, James Foster and Patrick Kidger</h2>

The paper is available at [arxiv.org/abs/2405.06464](https://arxiv.org/abs/2405.06464).

This repository is based on the Diffrax package by Patrick Kidger, which is available at [github.com/patrick-kidger/diffrax](https://github.com/patrick-kidger/diffrax).
The documentation for Diffrax can be found at [docs.kidger.site/diffrax](https://docs.kidger.site/diffrax).

The code for the CIR model example is in `notebooks/cir_model.ipynb`.

Neal's funnel example is split between two notebooks:
 - `notebooks/langevin_order.ipynb` contains the computation of strong orders of solvers on the Neal's funnel Langevin SDE
 - `notebooks/funnel_mcmc.ipynb` compares the output distribution of Langevin Monte Carlo against the No-U-Turn Sampler on the Neal's funnel model

The main entry point for the Bayesian Logistic Regression example is in `mcmc/progressive_run.py`.
Pregenerated plots can be found in `mcmc/progressive_results/good_plots/`.
For an experiment showing the strong order of convergence on the Bayesian Logistic Regression model, see `notebooks/logreg_order.ipynb`.

To use the code, clone this repository and install the requirements:

```bash
git clone https://github.com/andyElking/Single-seed_BrownianMotion.git
cd Single-seed_BrownianMotion/
pip install -e .
```
