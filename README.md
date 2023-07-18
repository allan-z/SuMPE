# Mixture Proportion Estimation Beyond Irreducibility

Yilun Zhu, Aaron Fjeldsted, Darren Holland, George Landon, Azaree Lintereur, and Clayton Scott, ``Mixture Proportion Estimation Beyond Irreducibility,'' accepted to International Conference on Machine Learning, 2023.
[[Paper]](https://arxiv.org/abs/2306.01253) [[Poster]](https://drive.google.com/file/d/1qqqi_Hvpa7LY5tLT1xgi2ts5NlspZODp/view?usp=sharing)

**To do**
- [x] release the code that could reproduce experimental result
- [ ] clear-up the implementation

**2023-6-1**: released the code (users needs to specify the parameters to match the test setup); further clean-up needed 

This folder includes 4 baseline MPE algorithms: DPL, EN, KM, TIcE, \
extracted from https://github.com/dimonenka/DEDPUL and https://web.eecs.umich.edu/~cscott/code.html#kmpe. \
The Regrouping algorithm was extracted from https://openreview.net/forum?id=aYAA-XHKyk.

The Subsampling algorithm (SuMPE) was implemented directly in the `experiments_xxx.py` files. \
We also re-implemented DPL, EN, KM that leverage histogram implementation \
(can be found in `experiments_nuclear_SuMPE.py` and `KMPE_discrete.py`)

All the results shown in experiment section and appendix in the paper can be reproduced by running the `experiments_xxx.py` files. 
To be specific:
- Unfolding, Gamma Ray Spectra Data: 
  - run `experiments_nuclear_SuMPE.py`
- Domain Adaptation, Synthetic Data: 
  - run `experiments_synth_SuMPE.py`
  - specify `scenario = 'Domain adaptation'`
- Domain Adaptation, Benchmark Data: 
  - run `experiments_UCI_MNIST_SuMPE.py`
  - specify `scenario = 'Domain adaptation'`, `data_mode = xxx`
- Selected/Reported at Random, Benchmark Data: 
  - run `experiments_UCI_MNIST_SuMPE.py`
  - specify `scenario = 'Reported at random'`, `data_mode = xxx`
- Appendix, When Irreducibility Holds: 
  - run `experiments_synth_SuMPE.py`
  - specify `scenario = 'irreducible'`

with some hyperparameter (like sample sizes) may need to change to match experimental section exactly.
