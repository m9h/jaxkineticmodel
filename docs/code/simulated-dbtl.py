from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import numpy as np
from jaxkineticmodel.utils import get_logger
from jaxkineticmodel.simulated_dbtl.dbtl import DesignBuildTestLearnCycle
logger = get_logger(__name__)

# Reproducible demonstration. Broad random perturbations can exhaust the ODE
# solver and produce NaN labels, which cannot be used to train XGBoost.
np.random.seed(0)

# load model (Messiha et. al (2013))
filepath = ("models/sbml_models/working_models/Messiha2013.xml")

model = SBMLModel(filepath)
# we retrieve parameters from the model


# timespan of model
# A finite observation window for this small tutorial, not a steady-state claim.
ts = jnp.linspace(0, 1000, 201)

dbtl_cycle = DesignBuildTestLearnCycle(model=model,
                                       parameters=model.parameters,
                                       initial_conditions=model.y0,
                                       timespan=ts,
                                       target=['PEP'])

# design phase
# These are additive promoter increments: the resulting multiplier is
# 1 + the sum of increments assigned to that parameter, not the increment alone.
parameter_perturbations = {name: [0.02, 0.05, 0.1] for name in (
    'lp_ADH__Kacald', 'lp_ENO__kcat_ENO1', 'lp_FBA__Kdhap',
    'lp_HXK__kcat_HXK1', 'lp_PGK__kcat', 'lp_HXT__Vmax', 'lp_GND__Kp6g_GND1')}

dbtl_cycle.design_establish_library_elements(parameter_perturbations=parameter_perturbations)

_ = dbtl_cycle.design_assign_positions(n_positions=6)
_ = dbtl_cycle.design_assign_probabilities(probabilities_per_position=None)
strain_designs = dbtl_cycle.design_generate_strains(samples=24)

# build phase
values = dbtl_cycle.build_simulate_strains(strain_designs, plot=False)
if not all(np.isfinite(v).all() for v in values.values()):
    raise RuntimeError('A simulated strain failed; inspect its ODE solve before training')


# test phase
noised_values = dbtl_cycle.test_add_noise(values, percentage=0.1, noise_type='heteroschedastic')
data = dbtl_cycle.test_format_dataset(strain_designs=strain_designs,
                                      production_values=noised_values,
                                      reference_parameters=dbtl_cycle.parameters)

# learn phase
xgbparameters = {'tree_method': 'auto', 'reg_lambda': 1, 'max_depth': 2,
                 'nthread': 1, 'seed': 0, "disable_default_eval_metric": 0}
alternative_params = {'num_boost_round': 10, 'early_stopping_rounds': 40}
bst, r2_scores = dbtl_cycle.learn_train_model(data=data,
                                              target="PEP",
                                              model_type="XGBoost",
                                              args=(xgbparameters, alternative_params), test_size=0.20)

dbtl_cycle.learn_validate_model(samples=12,
                                target='PEP',
                                plotting=True
                                )
