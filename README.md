

# JaxKineticModel
Python package to simulate, build, and train kinetic models.

This fork updates the project to Python 3.12+ and JAX 0.12, with a locked
environment, independent RoadRunner/COPASI validation, shared BioModels fixtures,
and a differentiable MAPK intervention example. See the
[migration and validation guide](validation/README.md) for the equations,
reproduction commands, numerical results and remaining SBML limitations.

### Installation
Install the package using
```
pip install jaxkineticmodel
```
Or the latest version using
```
git clone https://github.com/AbeelLab/jaxkineticmodel.git
python -m pip install .
```


### Documentation package 
Documentation on jaxkineticmodel can be found [here](https://abeellab.github.io/jaxkineticmodel/)

### Running experiments as reported in paper
- Datasets generated from SBML models can be found in [datasets](datasets/)
- Glucose pulse datasets can be found [here](datasets/VanHeerden_Glucose_Pulse/)
- Experiments as reported in [1] can be found in [scripts](scripts/). Results from these experiments can found in the 
[4tu-repository](https://data.4tu.nl/private_datasets/o-HY8kDJhoCXyNOijO9Eaylje7E2dU-ex-edboPBDZ8)

### Source code
All data, source code, scripts, and results can be found in the [4tu-repository](https://data.4tu.nl/private_datasets/o-HY8kDJhoCXyNOijO9Eaylje7E2dU-ex-edboPBDZ8)

### Reference
If you use this package in your work, please reference:

[1] van Lent, P., Bunkova, O., Magyar, B., Planken, L., Schmitz, J., & Abeel, T. (2025). 
Jaxkineticmodel: Neural ordinary differential equations inspired parameterization of kinetic models. 
PLOS Computational Biology, 21(7), e1012733.
