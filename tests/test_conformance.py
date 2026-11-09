"""Numerical evidence for the JAX migration, using unchanged reference fixtures."""

import hashlib
import json

import jax
import jax.numpy as jnp
import libsbml
import numpy as np
import optax
import pytest

from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from jaxkineticmodel.load_sbml.export_sbml import SBMLExporter
from jaxkineticmodel.parameter_estimation.training import update_linear, update_log
from validation.conformance import (ANALYTIC, BIOMODELS, FIXTURES, exact, load,
                                    run_case, to_amounts)


@pytest.mark.parametrize('case', ANALYTIC)
@pytest.mark.parametrize('intervals', [1, 10])
def test_analytic_and_independent_oracles(case, intervals):
    pytest.importorskip('basico')
    pytest.importorskip('roadrunner')
    result = run_case(FIXTURES/'assignment_rules'/f'{case}.xml',
                      np.linspace(0, 1, intervals+1), analytic=case)
    assert all(c['passed'] for c in result['comparisons'].values()), result


@pytest.mark.parametrize('number', BIOMODELS)
def test_biomodel_baseline(number):
    pytest.importorskip('basico')
    pytest.importorskip('roadrunner')
    result = run_case(FIXTURES/'biomodels'/f'BIOMD{number:010}.xml', np.linspace(0, 50, 5001))
    assert all(c['passed'] for c in result['comparisons'].values()), result


def test_biomodel_source_hashes():
    for item in json.loads((FIXTURES/'biomodels/manifest.json').read_text()):
        assert hashlib.sha256((FIXTURES/'biomodels'/item['file']).read_bytes()).hexdigest() == item['sha256']


def test_rate_rule_fails_explicitly():
    with pytest.raises(NotImplementedError, match='Rate rules'):
        SBMLModel(str(FIXTURES/'assignment_rules/rate_rule.xml'))


def test_jit_vmap_and_parameter_gradient():
    model, solve = load(FIXTURES/'assignment_rules/time_dependent.xml')
    def endpoint(alpha):
        return solve(jnp.array([0., 1.]), model.y0, model.parameters | {'alpha': alpha})[-1, 0]
    alphas = jnp.array([.5, 1., 2.])
    np.testing.assert_allclose(jax.jit(jax.vmap(endpoint))(alphas),
                               2*np.exp(-1-np.asarray(alphas)/2), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(jax.jit(jax.grad(endpoint))(1.), -np.exp(-1.5), rtol=1e-7, atol=1e-9)


def test_nonzero_time_restart():
    model, solve = load(FIXTURES/'assignment_rules/time_dependent.xml')
    y = np.asarray(solve(jnp.array([1., 2.]), model.y0, model.parameters))[-1]
    np.testing.assert_allclose(y, [2*np.exp(-2.5), 2-2*np.exp(-2.5)], rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('space', ['linear', 'log'])
def test_optax_training_reduces_loss(space):
    model, solve = load(FIXTURES/'assignment_rules/time_dependent.xml')
    def loss(params, ts, ys):
        alpha = jnp.exp2(params['alpha']) if space == 'log' else params['alpha']
        pred = solve(ts, model.y0, model.parameters | {'alpha': alpha})
        return jnp.mean((pred-ys)**2)
    ts = jnp.linspace(0, 1, 11)
    ys = jnp.asarray(exact('time_dependent', ts))
    optimizer = optax.adam(.05)
    params = {'alpha': jnp.array(.4)}
    state = optimizer.init(params)
    update = update_log if space == 'log' else update_linear
    step = jax.jit(lambda state, params: update(state, params, ts, ys, loss, optimizer))
    losses = []
    for _ in range(20):
        state, params, value, grads = step(state, params)
        assert np.isfinite(float(grads['alpha']))
        losses.append(float(value))
    assert losses[-1] < .4*losses[0], losses


@pytest.mark.parametrize('amount_symbol,initial_as_amount', [(True, True), (True, False), (False, True), (False, False)])
def test_initial_units_and_export_roundtrip(tmp_path, amount_symbol, initial_as_amount):
    doc = libsbml.readSBML(str(FIXTURES/'assignment_rules/constant.xml'))
    m = doc.getModel()
    for s in m.getListOfSpecies():
        amount = 2. if s.getId() == 'A' else 0.
        s.unsetInitialAmount()
        s.setHasOnlySubstanceUnits(amount_symbol)
        if initial_as_amount:
            s.setInitialAmount(amount)
        else:
            s.setInitialConcentration(amount/2)
    # A concentration-valued symbol needs V*k*A to describe the same amount flux.
    formula = 'k0 * A' if amount_symbol else 'cell * k0 * A'
    m.getReaction(0).getKineticLaw().setMath(libsbml.parseL3Formula(formula))
    path = tmp_path/'units.xml'
    libsbml.writeSBMLToFile(doc, str(path))
    model, solve = load(path)
    ts = jnp.linspace(0, 1, 11)
    np.testing.assert_allclose(to_amounts(model, solve(ts, model.y0, model.parameters)),
                               exact('constant', ts), rtol=1e-7, atol=1e-9)
    exported = tmp_path/'export.xml'
    SBMLExporter(model.get_kinetic_model()).export(initial_conditions=model.y0,
        parameters=model.parameters, output_file=str(exported))
    roundtrip, solve_again = load(exported)
    np.testing.assert_allclose(to_amounts(roundtrip, solve_again(ts, roundtrip.y0, roundtrip.parameters)),
                               exact('constant', ts), rtol=1e-7, atol=1e-9)
