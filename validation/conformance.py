"""Compare unchanged SBML fixtures against analytic solutions and two engines."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import jax
import jax.numpy as jnp
import libsbml
import numpy as np

from jaxkineticmodel.load_sbml.sbml_model import SBMLModel

FIXTURES = Path(__file__).resolve().parents[1] / 'tests/fixtures/conformance'
ANALYTIC = ('constant', 'time_dependent', 'state_dependent', 'chained',
            'assigned_species', 'concentration_species')
BIOMODELS = (10, 26, 27, 29, 31)
RTOL, ATOL = 1e-9, 1e-11
COMPARE_RTOL, COMPARE_ATOL = 2e-5, 1e-7


def source_hashes():
    import jaxkineticmodel.load_sbml as source
    root = Path(source.__file__).parent
    return {f'jaxkineticmodel/load_sbml/{p}': hashlib.sha256((root/p).read_bytes()).hexdigest()
            for p in ('sbml_model.py', 'jax_kinetic_model.py', 'export_sbml.py',
                      'sympy_converter.py', 'sbml_load_utils.py')}


def load(path):
    model = SBMLModel(str(path))
    solve = model.get_kinetic_model()
    solve._change_solver(solve.solver, rtol=RTOL, atol=ATOL)
    solve.dt0 = 1e-4
    return model, jax.jit(solve)


def references(path, times, ids, *, amounts=False, initial=None):
    """Adapters for the checked-in fixed-volume fixtures, in native model units."""
    import roadrunner
    import basico
    import COPASI

    rr = roadrunner.RoadRunner(str(path))
    rr.conservedMoietyAnalysis = False
    rr.integrator.relative_tolerance = RTOL
    rr.integrator.absolute_tolerance = ATOL
    for s, value in (initial or {}).items():
        rr[f'[{s}]'] = float(value)
    selections = ids if amounts else [f'[{s}]' for s in ids]
    r = np.asarray(rr.simulate(times=list(times), selections=['time'] + selections))
    np.testing.assert_allclose(r[:, 0], times, rtol=0, atol=1e-9)
    model = basico.load_model(str(path))
    if model is None:
        raise ValueError(f'COPASI could not load {path}')
    try:
        if initial:
            species = {s.getSBMLId(): s for s in model.getModel().getMetabolites()}
            for sid, value in initial.items():
                species[sid].setInitialConcentration(float(value))
                model.getModel().updateInitialValues(species[sid].getInitialConcentrationReference())
        c = basico.run_time_course(model=model, values=list(times), method='deterministic',
                                  use_initial_values=True, automatic=False, use_sbml_id=True,
                                  use_concentrations=True, r_tol=RTOL, a_tol=ATOL)
        np.testing.assert_allclose(c.index, times, rtol=0, atol=1e-9)
        c = c.loc[:, ids].to_numpy()
        if amounts:
            doc = libsbml.readSBML(str(path))
            sbml = doc.getModel()
            c *= [sbml.getCompartment(sbml.getSpecies(s).getCompartment()).getSize() for s in ids]
        return r[:, 1:], c
    finally:
        COPASI.CRootContainer.removeDatamodel(model)


def to_amounts(model, values):
    sbml = model.model
    factors = [1 if s in model.amount_species else
               sbml.getCompartment(sbml.getSpecies(s).getCompartment()).getSize()
               for s in model.species_names]
    return np.asarray(values) * np.asarray(factors)


def exact(case, times):
    t = np.asarray(times)
    a = 2 * np.exp(-t) if case == 'constant' else (
        2 / (1 + 2*t) if case == 'state_dependent' else 2 * np.exp(-t - t*t/2))
    return np.column_stack((a, 2-a))


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Incompatible or nonfinite trajectories')
    error = np.abs(a-b)
    scaled = error / (COMPARE_ATOL + COMPARE_RTOL*np.maximum(np.abs(a), np.abs(b)))
    return {'max_absolute_error': float(error.max()), 'max_scaled_error': float(scaled.max()),
            'passed': bool((scaled <= 1).all())}


def run_case(path, times, *, analytic=None):
    model, solve = load(path)
    ys = np.asarray(solve(jnp.asarray(times), model.y0, model.parameters))
    if analytic:
        ys = to_amounts(model, ys)
    rr, cp = references(path, times, model.species_names, amounts=bool(analytic))
    comparisons = {'jax_roadrunner': compare(ys, rr), 'jax_copasi': compare(ys, cp),
                   'roadrunner_copasi': compare(rr, cp)}
    if analytic:
        truth = exact(analytic, times)
        comparisons.update({f'{name}_exact': compare(a, truth)
                            for name, a in [('jax', ys), ('roadrunner', rr), ('copasi', cp)]})
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'species': model.species_names, 'samples': len(times),
            'time_start': float(times[0]), 'time_end': float(times[-1]),
            'quantity': 'amount' if analytic else 'concentration', 'comparisons': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('validation/results/report.json'))
    args = parser.parse_args()
    packages = ['jaxkineticmodel', 'jax', 'jaxlib', 'diffrax', 'equinox', 'optax', 'numpy',
                'libroadrunner', 'copasi-basico', 'python-copasi', 'python-libsbml']
    report = {'versions': {p: version(p) for p in packages}, 'source_sha256': source_hashes(), 'platform': platform.platform(),
              'python': platform.python_version(), 'devices': [str(d) for d in jax.devices()],
              'x64': jax.config.x64_enabled, 'solver': 'Kvaerno5',
              'solver_rtol': RTOL, 'solver_atol': ATOL,
              'comparison_rtol': COMPARE_RTOL, 'comparison_atol': COMPARE_ATOL, 'cases': {}}
    for name in ANALYTIC:
        for n in (1, 10):
            print(name, n, flush=True)
            report['cases'][f'{name}_{n}'] = run_case(
                FIXTURES / 'assignment_rules' / f'{name}.xml', np.linspace(0, 1, n+1), analytic=name)
    for number in BIOMODELS:
        print('BioModel', number, flush=True)
        report['cases'][str(number)] = run_case(
            FIXTURES / 'biomodels' / f'BIOMD{number:010}.xml', np.linspace(0, 50, 5001))
    try:
        SBMLModel(str(FIXTURES / 'assignment_rules/rate_rule.xml'))
    except NotImplementedError as e:
        report['unsupported_rate_rule'] = str(e)
    else:
        raise AssertionError('Unsupported rate rule was silently accepted')
    report['all_comparisons_passed'] = all(
        c['passed'] for case in report['cases'].values() for c in case['comparisons'].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not report['all_comparisons_passed']:
        raise SystemExit('Conformance failed; inspect the report')


if __name__ == '__main__':
    main()
