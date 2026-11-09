"""Regression cases exposed by the larger biochemical-network catalog."""

import jax.numpy as jnp
import libsbml
import numpy as np
import pytest

from jaxkineticmodel.load_sbml.export_sbml import SBMLExporter
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from validation.conformance import FIXTURES, load, references, to_amounts


def document(name="constant"):
    return libsbml.readSBML(str(FIXTURES / "assignment_rules" / f"{name}.xml"))


def save(doc, path):
    assert doc.checkInternalConsistency() == 0, doc.getErrorLog().toString()
    assert libsbml.writeSBMLToFile(doc, str(path)) == 1
    return path


def check_oracles(path, times, expected):
    rr, cp = references(path, np.asarray(times), ["A", "B"], amounts=True)
    for values in (rr, cp):
        np.testing.assert_allclose(values, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("level,version", [(2, 4), (3, 2)])
def test_events_are_rejected_instead_of_ignored(tmp_path, level, version):
    doc = document()
    model = doc.getModel()
    model.getParameter("k0").setConstant(False)
    event = model.createEvent()
    event.setId("change_rate")
    event.setUseValuesFromTriggerTime(True)
    trigger = event.createTrigger()
    trigger.setInitialValue(True)  # L2 semantics; the actual trigger is false at t=0.
    trigger.setPersistent(True)
    trigger.setMath(libsbml.parseL3Formula("time >= 0.5"))
    assignment = event.createEventAssignment()
    assignment.setVariable("k0")
    assignment.setMath(libsbml.parseL3Formula("2"))
    assert doc.setLevelAndVersion(level, version)
    path = save(doc, tmp_path / "event.xml")
    with pytest.raises(NotImplementedError, match="events require"):
        SBMLModel(str(path))
    rr, cp = references(path, [0., 1.], ["A", "B"], amounts=True)
    expected = [2*np.exp(-1.5), 2-2*np.exp(-1.5)]
    np.testing.assert_allclose(rr[-1], expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(cp[-1], expected, rtol=1e-7, atol=1e-9)


def test_delay_is_rejected_instead_of_becoming_subtraction(tmp_path):
    doc = document()
    doc.getModel().getReaction(0).getKineticLaw().setMath(libsbml.parseL3Formula("k0 * delay(A, 0.2)"))
    path = save(doc, tmp_path / "delay.xml")
    with pytest.raises(NotImplementedError, match="history-dependent solver"):
        SBMLModel(str(path))


def test_fractional_stoichiometry_and_repeated_references(tmp_path):
    doc = document()
    reaction = doc.getModel().getReaction(0)
    reaction.getReactant(0).setStoichiometry(.25)
    extra = reaction.createReactant()
    extra.setSpecies("A")
    extra.setStoichiometry(.25)
    extra.setConstant(True)
    reaction.getProduct(0).setStoichiometry(1.5)
    path = save(doc, tmp_path / "fractional.xml")
    model, solve = load(path)
    ts = jnp.linspace(0., 1., 11)
    a = 2*np.exp(-np.asarray(ts)/2)
    expected = np.column_stack((a, 3*(2-a)))
    np.testing.assert_allclose(to_amounts(model, solve(ts, model.y0, model.parameters)),
                               expected, rtol=1e-7, atol=1e-9)
    check_oracles(path, ts, expected)


@pytest.mark.parametrize("boundary", [False, True])
def test_assigned_species_is_derived_without_an_initial_value(tmp_path, boundary):
    doc = document("assigned_species")
    species = doc.getModel().getSpecies("U")
    species.setBoundaryCondition(boundary)
    species.unsetInitialAmount()
    species.unsetInitialConcentration()
    path = save(doc, tmp_path / "assigned.xml")
    model, solve = load(path)
    assert model.species_names == ["A", "B"]
    ts = jnp.linspace(0., 1., 11)
    a = 2*np.exp(-np.asarray(ts)-np.asarray(ts)**2/2)
    expected = np.column_stack((a, 2-a))
    np.testing.assert_allclose(to_amounts(model, solve(ts, model.y0, model.parameters)),
                               expected, rtol=1e-7, atol=1e-9)
    check_oracles(path, ts, expected)
    exported = tmp_path / "roundtrip.xml"
    SBMLExporter(model.get_kinetic_model()).export(model.y0, model.parameters, str(exported))
    again, repeat = load(exported)
    np.testing.assert_allclose(to_amounts(again, repeat(ts, again.y0, again.parameters)),
                               expected, rtol=1e-7, atol=1e-9)


def test_rule_dependency_order_and_function_calls_survive_export(tmp_path):
    doc = document()
    model = doc.getModel()
    function = model.createFunctionDefinition()
    function.setId("sum_two")
    function.setMath(libsbml.parseL3Formula("lambda(a, b, a+b)"))
    for name, formula in [("z", "y"), ("y", "x"), ("x", "sum_two(k0, time)")]:
        param = model.createParameter()
        param.setId(name)
        param.setConstant(False)
        rule = model.createAssignmentRule()
        rule.setVariable(name)
        rule.setMath(libsbml.parseL3Formula(formula))
    model.getReaction(0).getKineticLaw().setMath(libsbml.parseL3Formula("z*A"))
    path = save(doc, tmp_path / "chained.xml")
    loaded, solve = load(path)
    ts = jnp.linspace(0., 1., 11)
    a = 2*np.exp(-np.asarray(ts)-np.asarray(ts)**2/2)
    expected = np.column_stack((a, 2-a))
    np.testing.assert_allclose(to_amounts(loaded, solve(ts, loaded.y0, loaded.parameters)),
                               expected, rtol=1e-7, atol=1e-9)
    check_oracles(path, ts, expected)
    exported = tmp_path / "roundtrip.xml"
    SBMLExporter(loaded.get_kinetic_model()).export(loaded.y0, loaded.parameters, str(exported))
    again, repeat = load(exported)
    np.testing.assert_allclose(to_amounts(again, repeat(ts, again.y0, again.parameters)),
                               expected, rtol=1e-7, atol=1e-9)
