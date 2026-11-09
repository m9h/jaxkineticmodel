"""Batch and optimize a pool-preserving MAPK pulse, checking independent gradients."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .conformance import FIXTURES, compare, load, references, source_hashes


def run():
    path = FIXTURES/'biomodels/BIOMD0000000031.xml'
    model, solve = load(path)
    baseline_times = jnp.linspace(0, 50, 501)
    baseline = solve(baseline_times, model.y0, model.parameters)
    state = baseline[-1]
    i, j = model.species_names.index('M'), model.species_names.index('Mpp')
    times = jnp.linspace(0, 20, 201)

    def pulse(fraction):
        # Transfer material within the MAPK pool, preserving M + Mp + Mpp.
        moved = fraction*state[i]
        return state.at[i].add(-moved).at[j].add(moved)

    def trajectory(fraction):
        return solve(times, pulse(fraction), model.parameters)

    def endpoint(fraction):
        return trajectory(fraction)[-1, j]

    fractions = jnp.array([0., .1, .2, .35, .6])
    batch = jax.jit(jax.vmap(trajectory))(fractions)
    np.testing.assert_allclose(np.asarray(batch).sum(axis=-1), float(state.sum()), rtol=1e-8, atol=1e-7)
    comparisons = {}
    for f, ys in zip(np.asarray(fractions), np.asarray(batch)):
        initial = dict(zip(model.species_names, np.asarray(pulse(f)).tolist()))
        rr, cp = references(path, np.asarray(times), model.species_names, initial=initial)
        comparisons[str(float(f))] = {'roadrunner': compare(ys, rr), 'copasi': compare(ys, cp)}

    fraction = .2
    gradient = float(jax.jit(jax.grad(endpoint))(fraction))
    finite_differences = {}
    for h in (1e-3, 1e-4, 1e-5):
        outputs = []
        for f in (fraction-h, fraction+h):
            initial = dict(zip(model.species_names, np.asarray(pulse(f)).tolist()))
            outputs.append(references(path, np.asarray(times), model.species_names, initial=initial))
        finite_differences[str(h)] = {
            engine: float((outputs[1][k][-1, j]-outputs[0][k][-1, j])/(2*h))
            for k, engine in enumerate(('roadrunner', 'copasi'))}
        for g in finite_differences[str(h)].values():
            np.testing.assert_allclose(gradient, g, rtol=2e-4, atol=1e-5)

    # Synthetic, attainable target: identify the pulse used to generate it.
    target = endpoint(.35)
    def loss(theta):
        return ((endpoint(jax.nn.sigmoid(theta))-target)/state.sum())**2
    optimizer = optax.adam(.1)
    theta = jnp.log(.05/.95)
    opt_state = optimizer.init(theta)
    @jax.jit
    def step(theta, opt_state):
        value, grad = jax.value_and_grad(loss)(theta)
        updates, opt_state = optimizer.update(grad, opt_state)
        return optax.apply_updates(theta, updates), opt_state, value
    losses = []
    for _ in range(160):
        theta, opt_state, value = step(theta, opt_state)
        losses.append(float(value))
    fitted = float(jax.nn.sigmoid(theta))
    final_loss = float(loss(theta))
    assert final_loss < losses[0]*1e-4
    assert abs(fitted-.35) < .002
    assert all(c['passed'] for row in comparisons.values() for c in row.values())
    report = {'model': path.name, 'source_sha256': source_hashes(), 'baseline_duration': 50, 'observation_duration': 20,
              'intervention': 'Transfer fraction of M to Mpp; preserve total MAPK',
              'initial_fraction': .05, 'target_generating_fraction': .35,
              'target_Mpp': float(target), 'optimized_fraction': fitted,
              'initial_loss': losses[0], 'final_loss': final_loss,
              'gradient_at_fraction_0.2': gradient, 'finite_difference_gradients': finite_differences,
              'batch_fractions': np.asarray(fractions).tolist(),
              'batch_endpoint_Mpp': np.asarray(batch)[:, -1, j].tolist(),
              'comparisons': comparisons, 'all_checks_passed': True}
    return report, np.asarray(times), np.asarray(batch)[:, :, j], losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('validation/results'))
    args = parser.parse_args()
    report, times, batch, losses = run()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'mapk_intervention.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for f, ys in zip(report['batch_fractions'], batch):
        axes[0].plot(times, ys, label=f'{f:g}')
    axes[0].set(xlabel='Time after pulse (model units)', ylabel='Mpp (native concentration)',
                title='Transfer a fraction of M to Mpp')
    axes[0].legend(title='Fraction')
    axes[1].semilogy(losses)
    axes[1].set(xlabel='Optimizer step', ylabel='Squared normalized target error',
                title=f"Recovered fraction: {report['optimized_fraction']:.4f} (target 0.35)")
    fig.savefig(args.output/'mapk_intervention.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
