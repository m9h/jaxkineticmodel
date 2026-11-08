import re

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import sympy
import collections
from jaxkineticmodel.load_sbml.sbml_load_utils import construct_param_point_dictionary, separate_params
from jaxkineticmodel.utils import get_logger

jax.config.update("jax_enable_x64", True)

logger = get_logger(__name__)


class JaxKineticModel:
    def __init__(
            self,
            fluxes,
            stoichiometric_matrix,
            species_names,
            reaction_names,
            compartment_values,
            species_compartments,
            boundary_conditions,
            rate_divisors=None,
    ):
        """Initialize given the following arguments:
        v: the flux functions given as lambdified jax functions,
        S: a stoichiometric matrix. For now only support dense matrices, but later perhaps add for sparse
        params: kinetic parameters
        flux_point_dict: a dictionary for each vi that tells what the
          corresponding metabolites should be in y. Should be matched to S.
        ##A pointer dictionary?
        """
        self.func = fluxes
        self.stoichiometry = stoichiometric_matrix
        self.species_names = np.array(species_names)
        self.reaction_names = np.array(reaction_names)
        self.compartment_values = jnp.array(compartment_values)
        self.rate_divisors = self.compartment_values if rate_divisors is None else jnp.array(rate_divisors)
        self.species_compartments = species_compartments
        self.boundary_conditions = boundary_conditions
        self.argument_names = self._get_co_varnames()

    def _get_co_varnames(self):
        """helper function to directly pass arguments to evaluate properly
        """
        argument_names = {}
        for name, mechanism in self.func.items():
            argument_names[name] = mechanism.__code__.co_varnames
        return argument_names

    def __call__(self, t, y, args):
        """compute dMdt"""
        global_params, local_params, flux_array= args
        y = dict(zip(self.species_names, y))

        reaction_names = list(self.func.keys())

        # function evaluates the flux vi given y, parameter, local parameters, time dictionary
        def apply_func(t,
                       func: dict,
                       argument_names: dict,
                       y: jnp.ndarray,
                       global_params: dict,
                       local_params: dict, ):
            eval_dict = {**y, **global_params, **local_params}
            eval_dict['t'] = t
            eval_dict = {i: eval_dict[i] for i in argument_names}
            vi = func(**eval_dict)
            return vi

        for k, i in enumerate(self.reaction_names):

            v_i = apply_func(t=t, func=self.func[i], argument_names=self.argument_names[i],
                       y=y, global_params=global_params[i], local_params=local_params[i])
            flux_array= flux_array.at[k].set(v_i)

        # v = jnp.asarray(
        #     [apply_func(t=t, func=self.func[i], argument_names=self.argument_names[i],
        #                 y=y, global_params=global_params[i], local_params=local_params[i])
        #      for i in reaction_names]
        # )  # perhaps there is a way to vectorize this in a better way
        dY = jnp.matmul(self.stoichiometry, flux_array)  # dMdt=S*v(t)

        dY /= self.rate_divisors
        return dY


class NeuralODE:
    """ Class that wraps the kinetic model for simulation """
    func: JaxKineticModel

    def __init__(
            self,
            fluxes: list,
            stoichiometric_matrix: pd.DataFrame,
            compartment_values: list,
            species_compartments: list,
            boundary_conditions: dict,
            assignment_rules: dict,
            lambda_functions: dict,
            event_rules: dict,
            compartments: dict,
            compile: bool,
            **kwargs

    ):
        self.compile_status = compile
        self.fluxes = fluxes
        self.fluxes_symbolic = fluxes.copy() #for exporting the model to sbml

        self.reaction_names = list(stoichiometric_matrix.columns)
        self.species_names = list(stoichiometric_matrix.index)

        self.stoichiometry = stoichiometric_matrix

        self.compartment_values = compartment_values
        self.species_compartments = species_compartments
        self.lambda_functions = lambda_functions
        self.event_rules = event_rules
        self.compartments = compartments
        self.assignment_rules = assignment_rules
        self.boundary_conditions = boundary_conditions

        self.y0 = kwargs['y0'] #not used in training, but only to process the initial conditions in Trainer object
        self.parameters = kwargs['parameters'] #not used in training,
        self.amount_species = kwargs.get('amount_species', ())
        # but only to process the initial conditions in Trainer object

        #defined after compilation

        self.v_symbols = {}

        #hyperparameters for simulation
        self.max_steps = 300000
        self.rtol = 1e-10
        self.atol = 1e-12
        self.dt0 = 1e-12
        self.solver = diffrax.Kvaerno5()
        self.stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol,
                                                         pcoeff = 0.4, icoeff = 0.3)
        self.adjoint = diffrax.RecursiveCheckpointAdjoint()

        if self.compile_status:
            self._prepare_model()

        else:
            logger.info("Assignment rules, boundary conditions, lambda functions,"
                        "and compartment is not compiled. Run ._prepare_model() for simulation")

    def _prepare_model(self,
                       constant_parameters={},
                       ):
        """Substitutes assignment rules, boundary conditions,lambda functions, compartments.
        constant_parameters: dictionary with parameters that need to be replaced
        (e.g. they should not be considered during optimization)
        """
        # arguments from the lambda expression are mapped to their respective symbols.
        for reaction_name, equation in self.fluxes.items():
            for func in equation.atoms(sympy.Function):
                if hasattr(func, 'name'):
                    variables = self.lambda_functions[func.name].variables
                    variable_substitution = dict(zip(variables, func.args))
                    expression = self.lambda_functions[func.name].expr
                    expression = expression.subs(variable_substitution)
                    equation = equation.subs({func: expression})



            equation = equation.subs(self.assignment_rules)
            equation = equation.subs(self.boundary_conditions)
            if constant_parameters:
                equation = equation.subs(constant_parameters)
            equation = equation.subs(self.compartments)


            free_symbols = list(equation.free_symbols)
            equation = sympy.lambdify(free_symbols, equation, "jax")
            filtered_dict = dict(zip([str(i) for i in free_symbols], free_symbols))
            #maps back the filled in equations, lambdified.
            self.fluxes[reaction_name] = equation
            # all symbols that should be mapped to the equation
            self.v_symbols[reaction_name] = filtered_dict

        self.compile_status = True
        self.func = JaxKineticModel(fluxes=self.fluxes,
                                    stoichiometric_matrix=jnp.array(self.stoichiometry),
                                    species_names=self.stoichiometry.index,
                                    reaction_names=self.stoichiometry.columns,
                                    compartment_values=self.compartment_values,
                                    species_compartments=self.species_compartments,
                                    boundary_conditions=self.boundary_conditions,
                                    rate_divisors=[1.0 if s in self.amount_species else v
                                                   for s, v in zip(self.species_names, self.compartment_values)])
        for key in constant_parameters.keys():
            self.parameters.pop(key)

        # for each flux, metabolites are retrieved and mapped to the respective values in y0
        return logger.info("Compile complete")

    def _change_solver(self, solver, **kwargs):
        """To change the ODE solver object to any solver class from diffrax
        Does not support multiterm objects yet."""

        if isinstance(solver, diffrax.AbstractAdaptiveSolver):
            # for what I recall, only the adaptive part is important to ensure
            #it can be loaded properly
            self.solver = solver
            step_size_control_parameters = {'rtol': self.rtol, 'atol': self.atol,
                                            "pcoeff": 0.4, "icoeff": 0.3, "dcoeff": 0}
            for key in kwargs:
                if key in step_size_control_parameters:
                    step_size_control_parameters[key] = kwargs[key]
            self.stepsize_controller = diffrax.PIDController(**step_size_control_parameters)
        elif not isinstance(solver, diffrax.AbstractAdaptiveSolver):
            self.solver = solver
            self.stepsize_controller = diffrax.ConstantStepSize()
        else:
            logger.error(f"solver {type(solver)} not support yet")

        return logger.info(f"solver changed to {type(solver)}")

    def __call__(self, ts, y0, params):
        """Forward simulation step"""
        global_params, local_params = separate_params(params)

        # ensures that global params are loaded flux specific (necessary for jax)
        global_params = construct_param_point_dictionary(
            self.v_symbols, self.reaction_names, global_params
        )  # this is required,

        flux_array= jnp.zeros(len(self.reaction_names))

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.func),
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0,
            y0=y0,
            args=(global_params, local_params, flux_array),
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=self.max_steps,
            adjoint=self.adjoint
        )

        return solution.ys


def separate_params(params):
    """Seperates the global from local parameters using a identifier (lp.[Enz].)"""
    global_params = {}
    local_params = collections.defaultdict(dict)

    for key in params.keys():
        if re.match("lp_*_", key):
            fkey = key.removeprefix("lp_")
            splitted_list = fkey.split("__") #splits only on the first
            value = params[key]
            newkey = splitted_list[1]
            local_params[splitted_list[0]][newkey] = value
        else:
            global_params[key] = params[key]
    return global_params, local_params
