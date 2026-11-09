"""This code runs an export function on the kinetic model
We need from model.get_kinetic_model() the Jaxkmodel.func before jit compiling """

import libsbml
import math
import jax
import jax.numpy as jnp
from typing import Union
from jaxkineticmodel.load_sbml.sympy_converter import SympyConverter, LibSBMLConverter
from jaxkineticmodel.load_sbml.jax_kinetic_model import NeuralODE, expand_functions
from jaxkineticmodel.building_models.JaxKineticModelBuild import NeuralODEBuild, BoundaryCondition
from jaxkineticmodel.utils import get_logger
import sympy
from jaxkineticmodel.load_sbml.sbml_load_utils import separate_params

jax.config.update("jax_enable_x64", True)

logger = get_logger(__name__)


# design choice: separate export class instead of integrated with the SBML document
# this seems the way to go because when we use self-made models the sbml export


class SBMLExporter:
    """class used to export SBML model from a NeuralODE.JaxKineticModel object"""

    def __init__(self,
                 model: Union[NeuralODE, NeuralODEBuild]):
        assert isinstance(model, (NeuralODE, NeuralODEBuild))
        logger.info(f"Exporting Neural ODE model of instance {type(model)}")
        self.kmodel = model
        self.sympy_converter = SympyConverter()
        self.libsbml_converter = LibSBMLConverter()

        # we need to add this to NeuralODE object
        self.compartment_values = {}  # Need to deal with compartments for species for both NeuralODE and NeuralODEBuild

    def export(self,
               initial_conditions: jnp.ndarray,
               parameters: dict,
               output_file: str):
        """Exports model based on the input arguments to .xml file
        Input:
        - initial_conditions: initial conditions of the model
        - parameters: global parameters of the model"""
        try:
            document = libsbml.SBMLDocument(2, 4)
        except:
            raise SystemExit('Could not create SBML document')

        export_model = document.createModel()

        # initial conditions and the compartments species belong to
        # (non-constant, non-boundary species)

        initial_conditions = dict(zip(self.kmodel.species_names, initial_conditions))
        species_compartments = self.kmodel.func.species_compartments  #same for both
        species_reference = {}

        # compartments: we need to retrieve compartment dictionary without interacting with
        if isinstance(self.kmodel, NeuralODE):
            compartments = self.kmodel.compartments
        else:
            compartments = [float(i) for i in self.kmodel.func.compartment_values]
            compartments = dict(zip(species_compartments.values(), compartments))

        # needs to be assessed separately
        if isinstance(self.kmodel, NeuralODE):
            boundaries = self.kmodel.boundary_conditions
            boundaries = {key: BoundaryCondition(value) for key, value in boundaries.items()}

        elif isinstance(self.kmodel, NeuralODEBuild):
            boundaries = self.kmodel.boundary_conditions

        for (c_id, c_size) in compartments.items():
            # Create a compartment inside this model, and set the required
            # attributes for an SBML compartment in SBML Level 3.
            c1 = export_model.createCompartment()
            check(c1, 'create compartment')
            check(c1.setId(c_id), 'set compartment id')
            check(c1.setConstant(True), 'set compartment "constant"')
            check(c1.setSize(c_size), 'set compartment "size"')
            check(c1.setSpatialDimensions(3), 'set compartment dimensions')

        # boundary conditions and species
        boundary_species = {}
        amount_species = set(getattr(self.kmodel, 'amount_species', ()))
        for (s_id, s_comp) in species_compartments.items():
            s1 = export_model.createSpecies()
            check(s1, 'create species')
            check(s1.setId(s_id), 'set species id')
            check(s1.setCompartment(s_comp), 'set species s1 compartment')
            check(s1.setHasOnlySubstanceUnits(s_id in amount_species), 'set "hasOnlySubstanceUnits" on s1')
            set_initial = s1.setInitialAmount if s_id in amount_species else s1.setInitialConcentration
            species_reference[s_id] = s1

            if isinstance(self.kmodel, NeuralODE) and s_id in self.kmodel.assignment_rules:
                check(s1.setConstant(False), 'set assigned species constant flag')
                check(s1.setBoundaryCondition(False), 'set assigned species boundary flag')
                boundary_species[s_id] = s1
            elif s_id not in boundaries.keys():
                check(s1.setConstant(False), 'set "constant" attribute on s1')
                check(set_initial(float(initial_conditions[s_id])), 'set initial value for s1')
                check(s1.setSubstanceUnits('mole'), 'set substance units for s1')
                check(s1.setBoundaryCondition(False), 'set "boundaryCondition" on s1')


            elif s_id in boundaries.keys():
                check(s1.setBoundaryCondition(True), 'set "boundaryCondition" on s1')
                check(s1, 'create species')
                check(s1.setCompartment(s_comp), 'set "compartment" attribute on s1')
                condition = boundaries[s_id]
                if condition.is_constant:
                    assert isinstance(condition.sympified, sympy.Number)
                    check(s1.setConstant(True), 'set "constant" attribute on s1')

                    check(set_initial(float(condition.sympified)), 'set initial value for s1')
                elif not condition.is_constant:
                    logger.info(f"boundary not constant, support not tested yet")
                    check(s1.setConstant(False), 'set "constant" attribute on s1')
                    check(s1.setInitialAmount(jnp.nan), 'set "initialAmount" attribute on s1')
                    math_ast = self.sympy_converter.sympy2libsbml(condition.sympified)
                    orig = self.libsbml_converter.libsbml2sympy(math_ast)
                    print('orig spline', str(orig))
                    print('spline sympified', str(condition.sympified))
                    # interesting, for the piece wise,
                    # condition.sympified is actually different

                    rule = export_model.createAssignmentRule()
                    check(rule.setVariable(s1.id), 'set "rule" attribute on s1')
                    check(rule.setMath(math_ast), 'set "math" attribute on s1')

                boundary_species[s_id] = s1  # need to save this for later use in reactions

        #we use an parameter dict input argument for export
        global_parameters, local_parameters = separate_params(parameters)


        for p_name, p_value in global_parameters.items():
            p1 = export_model.createParameter()
            check(p1.setId(str(p_name)), 'set parameter name')
            if p_name in self.kmodel.assignment_rules.keys():
                check(p1.setConstant(False), 'set "constant" attribute on s1 for assignment rule')
            else:
                check(p1.setConstant(True), 'set parameter name')
            if p_name not in self.kmodel.assignment_rules or math.isfinite(float(p_value)):
                check(p1.setValue(float(p_value)), 'set parameter value')

        # reactions and boundaries are dealt with slightly different right now between two objects.
        # NeuralODE objects has sympy expressions for reactions, while in the NeuralODEBuild class
        # they are implemented through the Mechanism/Reaction objects.
        # We therefore deal with them (for now) separately
        if isinstance(self.kmodel, NeuralODE):
            for (reaction, mechanism) in self.kmodel.fluxes_symbolic.items():
                mechanism = expand_functions(mechanism, self.kmodel.lambda_functions)

                r1 = export_model.createReaction()
                check(r1, 'create reaction')
                check(r1.setId(str(reaction)), 'set reaction name')
                check(r1.setReversible(False), 'set reversible')  # required

                # include species reference based on stoichiometry
                stoichiometry = self.kmodel.stoichiometry[reaction].to_dict()
                for (s_id, stoich) in stoichiometry.items():
                    if stoich < 0:
                        species_ref1 = r1.createReactant()

                    elif stoich > 0:
                        species_ref1 = r1.createProduct()
                    else:
                        continue

                    specimen = species_reference[s_id]

                    check(species_ref1, 'create species')
                    check(species_ref1.setSpecies(specimen.getId()), 'set reactant species id')
                    # check(species_ref1.setConstant(specimen.constant), 'set reactant species id')
                    check(species_ref1.setStoichiometry(abs(stoich)), 'set absolute reactant/product stoichiometry')

                str_mechanism = [str(i) for i in mechanism.free_symbols]
                for (s_id, specimen) in boundary_species.items():
                    if str(s_id) in str_mechanism:
                        species_ref1 = r1.createModifier()
                        check(species_ref1, 'create boundary species')
                        check(species_ref1.setSpecies(specimen.getId()), 'set boundary species id')

                #include modifiers reference based on boundary
                # mechanism.free_symbols
                math_ast = self.sympy_converter.sympy2libsbml(mechanism)
                orig = self.libsbml_converter.libsbml2sympy(math_ast)
                assert str(mechanism) == str(orig)

                kinetic_law = r1.createKineticLaw()
                check(kinetic_law, 'create kinetic law')
                check(kinetic_law.setMath(math_ast), 'set math on kinetic law')

                for p_name, p_value in local_parameters[reaction].items():
                    local_param = kinetic_law.createParameter()
                    check(local_param, 'create local parameter')
                    check(local_param.setId(str(p_name)), 'set parameter name')
                    check(local_param.setValue(float(p_value)), 'set parameter name')

                #local parameters (based on identifier)

            # Preserve assignment rules; user-defined function calls are expanded.
            for target, expression in self.kmodel.assignment_rules.items():
                rule = export_model.createAssignmentRule()
                check(rule.setVariable(target), 'set assignment target')
                expression = expand_functions(expression, self.kmodel.lambda_functions)
                check(rule.setMath(self.sympy_converter.sympy2libsbml(expression)), 'set assignment math')

        elif isinstance(self.kmodel, NeuralODEBuild):
            for reaction in self.kmodel.func.reactions:
                r1 = export_model.createReaction()
                check(r1, 'create reaction')
                check(r1.setId(str(reaction.name)), 'set reaction id')
                check(r1.setReversible(False), 'set reversible')  # required
                for (s_id, stoich) in reaction.stoichiometry.items():
                    if stoich < 0:
                        species_ref1 = r1.createReactant()
                        check(species_ref1.setStoichiometry(abs(stoich)), 'set absolute reactant/product stoichiometry')

                    elif stoich > 0:
                        species_ref1 = r1.createProduct()
                        check(species_ref1.setStoichiometry(abs(stoich)), 'set absolute reactant/product stoichiometry')

                    else:
                        logger.info(f"modifier {s_id} is added")
                        species_ref1 =r1.createModifier()

                    # use the dictionary with species references
                    specimen = species_reference[s_id]
                    check(species_ref1, 'create reactant')
                    check(species_ref1.setSpecies(specimen.getId()), 'set reactant species id')
                    # check(species_ref1.setConstant(specimen.getConstant()), 'set reactant species id')


                math_ast = self.sympy_converter.sympy2libsbml(reaction.mechanism.symbolic())
                orig = self.libsbml_converter.libsbml2sympy(math_ast)
                assert str(reaction.mechanism.symbolic()) == str(orig)

                kinetic_law = r1.createKineticLaw()
                check(kinetic_law, 'create kinetic law')
                check(kinetic_law.setMath(math_ast), 'set math on kinetic law')

            if self.kmodel.assignment_rules:
                assignment_rules = self.kmodel.assignment_rules
                for key, assignment_rule in assignment_rules.items():
                    math_ast = self.sympy_converter.sympy2libsbml(assignment_rule.equation)
                    orig = self.libsbml_converter.libsbml2sympy(math_ast)
                    assert str(assignment_rule.equation) == str(orig)
                    rule = export_model.createAssignmentRule()
                    check(rule.setVariable(key), 'set variable name')
                    check(rule.setMath(math_ast), "set math on kinetic law")






        # file = open(output_file, "w")

        sbml = libsbml.writeSBMLToFile(document, output_file)
        return print((f"{sbml}: succesful export. Please check whether its correct"
                f"SBML using the online sbml validator"))


def check(value, message):
    """Check output from libSBML functions for errors.
   If 'value' is None, prints an error message constructed using 'message' and then raises SystemExit.
   If 'value' is an integer, it assumes it is a libSBML return status code.
   If 'value' is any other type, return it unchanged and don't do anything else.
   For the status code, if the value is LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
   prints an error message constructed using 'message' along with text from libSBML explaining the meaning of the
   code, and raises SystemExit.
   """
    if value is None:
        raise SystemExit('LibSBML returned a null value trying to ' + message + '.')
    elif type(value) is int:
        if value == libsbml.LIBSBML_OPERATION_SUCCESS:
            return
        else:
            err_msg = 'Error encountered trying to ' + message + '.' \
                      + 'LibSBML returned error code ' + str(value) + ': "' \
                      + libsbml.OperationReturnValue_toString(value).strip() + '"'
            raise SystemExit(err_msg)
