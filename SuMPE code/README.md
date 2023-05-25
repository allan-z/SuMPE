# Mixture Proportion Estimation Beyond Irreducibility

This folder includes 4 baseline MPE algorithms: DPL, EN, KM, TIcE, extracted from https://github.com/dimonenka/DEDPUL and https://web.eecs.umich.edu/~cscott/code.html#kmpe. 
The Regrouping algorithm was extracted from https://openreview.net/forum?id=aYAA-XHKyk.

The Subsampling algorithm was implemented directly in the experiments_xxx.py files. We also re-implemented DPL, EN, KM that leverage histogram implementation (can be find in experiments_nuclear_SuMPE.py and KMPE_discrete.py)

All the results shown in experiment section and appendix in the paper can be reproduced by running the experiments_xxx.py files. 
To be specific:
- Unfolding, Gamma Ray Spectra Data: 
  - run experiments_nuclear_SuMPE.py
- Domain Adaptation, Synthetic Data: 
  - run experiments_synth_SuMPE.py
  - specify scenario = 'Domain adaptation'
- Domain Adaptation, Benchmark Data: 
  - run experiments_UCI_SuMPE.py
  - specify scenario = 'Domain adaptation'
- Selection/Reported at Random, Benchmark Data: 
  - run experiments_UCI_SuMPE.py
  - specify scenario = 'Reported at random'
- Appendix, When Irreducibility Holds: 
  - run experiments_synth_SuMPE.py
  - specify scenario = 'irreducible'

with some hyperparameter (like sample sizes) may need to change to match experimental section exactly.