from typing import Union

import sympy as sp
import libsbml
import jax.numpy as jnp
import numpy as np
import pandas as pd
import re
import collections
import os
from jaxkineticmodel.utils import get_logger
from jaxkineticmodel.load_sbml.jax_kinetic_model import NeuralODE
from jaxkineticmodel.load_sbml.sympy_converter import SympyConverter, LibSBMLConverter

logger = get_logger(__name__)


class SBMLModel:
    """Loads SBML models for simulation purposes using jax/diffrax"""
    S: Union[pd.DataFrame, None]

    def __init__(self, file_path):
        self.model = self._load_model(file_path)
        unsupported_rules = [r for r in self.model.getListOfRules() if not r.isAssignment()]
        if unsupported_rules:
            raise NotImplementedError("Rate rules and algebraic rules are not supported; only assignment rules are supported")
        self.S = self._get_stoichiometric_matrix()
        # TODO the following two lines assume that self.S will never change
        self.reaction_names = list(self.S.columns)
        self.species_names = list(self.S.index)

        self.parameters = self._get_parameters()
        self.initial_assignments = self._get_initial_assignments()
        self.y0 = self._get_initial_conditions()
        self.amount_species = tuple(s.getId() for s in self.model.getListOfSpecies()
                                    if species_uses_amount(s, self.model))

        self.y0, self.parameters = self._update_with_initial_assignments()
        self.y0 = jnp.array(list(self.y0.values()))

        # species compartments are the string names, compartments is a dictionary,
        # and species_compartment_values are the values corresponding to each species
        self.species_compartments, self.compartments, self.compartment_values = self._get_compartments()

        self.v = self._get_fluxes()

        self.boundary_conditions, boundary_compartments = get_constant_boundary_species(self.model)

        self.species_compartments = {**self.species_compartments, **boundary_compartments}

        self.lambda_functions = get_lambda_function_dictionary(self.model)
        self.assignment_rules = get_assignment_rules_dictionary(self.model)
        self.event_rules = get_events_dictionary(self.model)



    @staticmethod
    def _load_model(file_path):
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError()

        reader = libsbml.SBMLReader()
        document = reader.readSBML(file_path)
        inconsistencies = document.checkInternalConsistency()
        if inconsistencies == 0:
            logger.info("No internal inconsistencies found")
        else:
            logger.warning(f"Number of internal inconsistencies: {document.checkInternalConsistency()}")

        model = document.getModel()
        logger.info("Model loaded.")
        logger.info(f" number of species: {model.getNumSpecies()}")
        logger.info(f" number of reactions: {model.getNumReactions()}")
        logger.info(f" number of global parameters: {model.getNumParameters()}")
        logger.info(f" number of constant boundary metabolites: {model.getNumSpeciesWithBoundaryCondition()}")
        logger.info(f" number of lambda function definitions: {len(model.function_definitions)}")
        logger.info(f" number of assignment rules: {model.getNumRules()}")
        logger.info(f" number of event rules: {model.getNumEvents()}")

        return model

    def _update_with_initial_assignments(self):
        """Update parameters and species with initial assignments."""

        #As far as I am aware, only parameters and species can be updated through initial assignments"""
        parameters = self.parameters
        y0 = self.y0

        for key, value in self.initial_assignments.items():
            if key in self.parameters.keys():
                value = value.subs(self.parameters)  #substitutes parameters
                value = value.subs(self.parameters)  # substitutes parameters
                parameters[key] = float(value)  #force it to be a float
            elif key in self.y0.keys():
                value = value.subs(self.parameters)
                value = value.subs(self.y0)
                y0[key] = float(value)  # force it to be a float
            else:
                logger.warning(f"initial assignment rule not in species or parameters. Assignment for"
                               "{key} is ignored and simulation might be wrong")
        return y0, parameters

    def _get_initial_assignments(self):
        "retrieve all rules that are defined outside of species and parameters, we then use these rules to update y0 and parameters"
        libsbml_converter = LibSBMLConverter()
        initial_assignments = {}
        for assignment in self.model.getListOfInitialAssignments():
            expression = assignment.getMath()
            expression = libsbml_converter.libsbml2sympy(expression)
            initial_assignments[assignment.id] = expression
        return initial_assignments

    def _get_compartments(self):
        """retrieves compartments and the compartment values of species. The latter is necessary for proper scaling
        of rate laws."""

        compartments = self.model.getListOfCompartments()
        compartments = {cmp.id: cmp.size for cmp in compartments}
        species_compartments, compartment_list = self._get_compartments_initial_conditions(compartments)

        return species_compartments, compartments, compartment_list

    def _get_stoichiometric_matrix(self) -> object:
        """Retrieves the stoichiometric matrix from the model."""
        species_ids = []
        reduced_species_list = []
        for s in self.model.getListOfSpecies():
            # these conditions do not have a rate law
            if s.getConstant() and s.getBoundaryCondition():
                continue
            elif s.getBoundaryCondition() and not s.getConstant():
                continue
            elif s.getConstant():
                continue
            else:
                reduced_species_list.append(s)
                species_ids.append(s.getId())

        # species = [s.getName() for s in model.getListOfSpecies()]
        reactions = [r.getId() for r in self.model.getListOfReactions()]

        stoichiometry_matrix = np.zeros((len(species_ids), len(reactions)))
        for reaction_index, reaction in enumerate(self.model.getListOfReactions()):
            reactants = {r.getSpecies(): r.getStoichiometry() for r in reaction.getListOfReactants()}
            products = {p.getSpecies(): p.getStoichiometry() for p in reaction.getListOfProducts()}

            for species_index, species_node in enumerate(reduced_species_list):
                species_id = species_node.getId()

                net_stoichiometry = -int(reactants.get(species_id, 0)) + int(products.get(species_id, 0))
                stoichiometry_matrix[species_index, reaction_index] = net_stoichiometry

        species_names = [s.getId() for s in reduced_species_list]
        reaction_names = [r.getId() for r in self.model.getListOfReactions()]

        return pd.DataFrame(stoichiometry_matrix, index=species_names, columns=reaction_names)

    def _get_initial_conditions(self):
        """Retrieves the species initial concentrations
        from the SBML model. If a species is a constant boundary condition,
        then it should be passed as a parameter instead of an initial condition,
        since it does not have a rate law

        also"""
        species = self.model.getListOfSpecies()
        initial_concentration_dict = {}
        for specimen in species:
            if specimen.isSetConstant() and specimen.isSetBoundaryCondition():
                # there are also non-stationary boundary conditions, deal with this later.
                if specimen.getConstant() and specimen.getBoundaryCondition():
                    logger.info(f"Constant Boundary Specimen {specimen.id}")
                    continue
                elif specimen.getBoundaryCondition() and not specimen.getConstant():
                    continue
                elif not specimen.getBoundaryCondition() and specimen.getConstant():
                    continue  # not a boundary, but still a constant
                elif not specimen.getBoundaryCondition() and not specimen.getConstant():
                    if specimen.isSetInitialConcentration():

                        initial_concentration_dict[specimen.id] = initial_species_value(specimen, self.model)
                    elif specimen.isSetInitialAmount():
                        initial_concentration_dict[specimen.id] = initial_species_value(specimen, self.model)
                    else:
                        logger.error(f"specimen {specimen.id} has no initial amount or concentration set")
            else:
                logger.warn(f"{specimen.id} constant/boundary attribute not set. Assume that boundary is constant")
                continue

        return initial_concentration_dict

    def _get_parameters(self):
        """Retrieves the parameters from the SBML model. Both local (with a identifier lp.{reaction.id}.) and global."""

        parameters = {}
        #retrieve global parameters
        global_params = self.model.getListOfParameters()
        global_parameters = {param.id: param.value for param in global_params}
        parameters.update(global_parameters)

        #retrieve local parameters
        for reaction in self.model.reactions:
            r = reaction.getKineticLaw()
            local_parameters = {"lp_" + str(reaction.id) + "__" + param.id: param.value for param in
                                r.getListOfParameters()}
            parameters.update(local_parameters)
        return parameters

    def _get_compartments_initial_conditions(self, compartments):
        """Returns a list of the compartment names and values of
        the initial conditions. This is necessary in the dMdt to properly scale."""
        species = self.model.getListOfSpecies()
        compartment_list = []
        species_compartments = {}

        for specimen in species:
            if specimen.isSetConstant() and specimen.isSetBoundaryCondition():
                # there are also non-stationary boundary conditions, deal with this later.
                if specimen.getConstant() and specimen.getBoundaryCondition():
                    logger.info(f"Constant Boundary Specimen {specimen.id}")
                    continue
                elif specimen.getBoundaryCondition() and not specimen.getConstant():
                    continue
                elif not specimen.getBoundaryCondition() and specimen.getConstant():
                    continue  # not a boundary, but still a constant
                elif not specimen.getBoundaryCondition() and not specimen.getConstant():
                    compartment_list.append(compartments[specimen.compartment])
                    species_compartments[specimen.id] = specimen.getCompartment()

        compartment_list = jnp.array(compartment_list)
        return species_compartments, compartment_list

    def _get_fluxes(self):
        """Retrieves flux functions from the SBML model and converts to sympy expressions"""
        libsbml_converter = LibSBMLConverter()
        v = {}
        for reaction in self.model.reactions:
            astnode_reaction = reaction.getKineticLaw().math
            equation = libsbml_converter.libsbml2sympy(astnode_reaction)  #sympy type
            v[reaction.id] = equation
            # v_symbol_dict[reaction.id] = filtered_dict
        return v



    def get_kinetic_model(self, compile=True):
        """returns a NeuralODE object with keywords arguments"""
        assert isinstance(compile, bool)
        kwargs = vars(self)

        include = ['y0', 'parameters', 'species_compartments',
                   'compartment_values', 'boundary_conditions',
                   'lambda_functions', 'assignment_rules', 'event_rules', 'compartments', 'amount_species']
        kwargs = {i: kwargs[i] for i in include}

        return NeuralODE(
            fluxes=self.v.copy(),
            stoichiometric_matrix=self.S,
            compile = compile,
            **kwargs)


def get_lambda_function_dictionary(model):
    """Stop giving these functions confusing names...
    it returns a dictionary with all lambda functions"""
    functional_dict = {}
    libsbml_converter = LibSBMLConverter()

    for function in model.function_definitions:
        id = function.getId()
        math = function.getMath()
        equation = libsbml_converter.libsbml2sympy(math)

        functional_dict[id] = equation
    return functional_dict


def get_compartments(model):
    """Some sbml models have compartments, retrieves them"""
    compartments = model.getListOfCompartments()
    compartment_dict = {cmp.id: cmp.size for cmp in compartments}
    return compartment_dict


def species_uses_amount(species, model):
    """The SBML species symbol denotes amount for substance-only/0-D species."""
    return (species.getHasOnlySubstanceUnits()
            or model.getCompartment(species.getCompartment()).getSpatialDimensions() == 0)


def initial_species_value(species, model):
    """Convert the initial value to the units of its SBML species symbol."""
    volume = model.getCompartment(species.getCompartment()).getSize()
    amount = species_uses_amount(species, model)
    if species.isSetInitialAmount():
        return species.getInitialAmount() if amount else species.getInitialAmount() / volume
    if species.isSetInitialConcentration():
        return species.getInitialConcentration() * volume if amount else species.getInitialConcentration()
    raise ValueError(f"Species {species.getId()} has no initial amount or concentration")


def get_constant_boundary_species(model):
    """Species that are boundary conditions should be fed as fixed, non-learnable parameters
    https://synonym.caltech.edu/software/libsbml/5.18.0/docs/formatted/python-api/classlibsbml_1_1_species.html"""
    constant_boundary = {}
    boundary_compartments = {}
    species = model.getListOfSpecies()
    for specimen in species:
        if specimen.getBoundaryCondition() or specimen.getConstant():
            boundary_compartments[specimen.id] = specimen.getCompartment()
            if specimen.isSetInitialConcentration():
                constant_boundary[specimen.id] = initial_species_value(specimen, model)
            elif specimen.isSetInitialAmount():
                constant_boundary[specimen.id] = initial_species_value(specimen, model)
            # else:
            #     logger.error(f"specimen {specimen.id} has no initial concentration or amount.")

    return constant_boundary, boundary_compartments


def get_local_parameters(reaction):
    """Some sbml models also have local parameters (locally defined for reactions),
    this function retrieves them for an individual reaction, removing the chance
    similarly named parameters are overwritten"""
    r = reaction.getKineticLaw()
    local_parameter_dict = {param.id: param.value for param in r.getListOfParameters()}
    return local_parameter_dict


def get_reaction_species(reaction):
    """Retrieves the substrates, products, and modifiers from sbml format.
    These will be passed to the Torch Kinetic Model class."""
    sub = reaction.getListOfReactants()
    prod = reaction.getListOfProducts()
    mod = reaction.getListOfModifiers()

    substrates = [s.species for s in sub]
    products = [p.species for p in prod]
    modifiers = [m.species for m in mod]

    species = substrates + products + modifiers
    return species


def get_reaction_symbols_dict(eval_dict):
    """This functions works on the local_dictionary passed in the sympify function
    It ensures that sympy symbols for parameters and y-values are properly passed,
    while the rest is simply substituted in the expression."""
    symbol_dict = {i: sp.Symbol(i) for i in eval_dict.keys() if not callable(eval_dict[i])}  # skip functions symbols
    return symbol_dict


def species_match_to_S(initial_conditions, species_names):
    """Small helper function ensures that y0 is properly matched to rows of S
    Input: initial conditions is a dictionary of initial conditions
    species_names: is the order of the rows in S"""
    y0 = []
    for species in species_names:
        if species in initial_conditions.keys():
            y0.append(initial_conditions[species])
    y0 = jnp.array(y0)
    return y0


def reaction_match_to_S(flux_funcs, reaction_names):
    """Small helper function ensures that y0 is properly matched to rows of S
    Input: initial conditions is a dictionary of initial conditions
    species_names: is the order of the rows in S"""
    v = []
    for reaction in reaction_names:
        v.append(flux_funcs[reaction])
    return v


def get_leaf_nodes(node, leaf_nodes):
    """Finds the leaf nodes of the mathml expression."""
    if node.getNumChildren() == 0:
        name = node.getName()
        if name is not None:
            leaf_nodes.append(name)
    else:
        for i in range(node.getNumChildren()):
            get_leaf_nodes(node.getChild(i), leaf_nodes)
    leaf_nodes = np.array(leaf_nodes)
    leaf_nodes = np.unique(leaf_nodes)
    leaf_nodes = leaf_nodes.tolist()
    return leaf_nodes


def get_assignment_rules_dictionary(model):
    """Get rules that assign to variables. I did not lambdify here"""
    libsbml_converter = LibSBMLConverter()
    assignment_dict = {}
    sorted_rules = sorted(model.rules, key=lambda r: r.getMetaId())
    for rule in sorted_rules:
        id = rule.getId()
        expr = rule.getMath()
        expr = libsbml_converter.libsbml2sympy(expr)

        assignment_dict[id] = expr

    return assignment_dict


def get_events_dictionary(model):
    """There are many type of events. For now, I only add a few and the rest will throw a warning"""
    num_events = model.getNumEvents()
    events_dict = {}
    if model.getLevel() == 2 and num_events != 0:
        for i in range(num_events):
            event = model.getEvent(i)

            # Print the trigger
            trigger = event.getTrigger()
            if trigger is not None:
                logger.info(f"Trigger: {libsbml.formulaToString(trigger.getMath())}")
                trigger_event = libsbml.formulaToL3String(trigger.getMath())

                # need to replace && for proper reading in sympy
                trigger_event = trigger_event.replace("&&", "&")
                leaf_nodes = []
                leaf_nodes = get_leaf_nodes(trigger.getMath(), leaf_nodes)
                symbols = {leaf_node: sp.Symbol(leaf_node) for leaf_node in leaf_nodes}

                expr = sp.sympify(trigger_event, locals=symbols)
                if event.isSetId():
                    events_dict[event.id] = expr
                if event.isSetName():
                    events_dict[event.name] = expr

            # Print the delay (if any)
            delay = event.getDelay()
            if delay is not None:
                logger.warn("sbml model has delay event, but this is not supported yet, output might be different")

            # Print the priority (if any)
            priority = event.getPriority()
            if priority is not None:
                logger.warn("sbml model has priority event,but this is not supported yet, output might be different")
    return events_dict
