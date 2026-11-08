# Shared simulator conformance fixtures

The XML files are unchanged copies of the cases used in
[m9h/sbmltoodejax](https://github.com/m9h/sbmltoodejax/tree/af646fcf4142734e8836be692478325cfec2c49b/test/fixtures/assignment_rules)
and [m9h/GRN_ODE](https://github.com/m9h/GRN_ODE/tree/9f36653/python/models).
They are loaded independently by Jaxkineticmodel, RoadRunner and COPASI.

## Analytic models

Each model transfers material from A to B in a **two-litre** compartment, with
initial amounts A = 2 mol and B = 0 mol. The exact amount solutions are:

| Case | A(t) | Purpose |
|---|---|---|
| constant | 2 exp(-t) | Constant-rate control |
| time_dependent | 2 exp(-t-t²/2) | Parameter assignment evaluated continuously |
| state_dependent | 2/(1+2t) | Assignment depends on current state |
| chained | 2 exp(-t-t²/2) | Assignment dependency ordering |
| assigned_species | 2 exp(-t-t²/2) | Amount-valued assigned species |
| concentration_species | 2 exp(-t-t²/2) | Concentration-valued assigned species |
| rate_rule | 2 exp(-t-t²/2) | Unsupported rate rule must raise explicitly |

The rate-rule model is not counted as a successful Jaxkineticmodel simulation.
The other six are checked at two output grids, against both independent engines
and the exact solution. Rules do not change the conserved total A + B = 2 mol.

## Biological models

These are MAPK **protein signaling** models, not literal gene-transcription
networks. All have fixed, size-one compartments. Comparisons use the original
initial concentrations and all dynamic species over 0–50 model-time units.

| BioModels ID | Model and distinctive feature |
|---|---|
| BIOMD0000000010 | Kholodenko (2000): three-stage MAPK cascade with negative feedback |
| BIOMD0000000026 | Markevich (2004): explicit enzyme–substrate complexes |
| BIOMD0000000027 | Markevich (2004): reduced ordered dual phosphorylation |
| BIOMD0000000029 | Markevich (2004): random phosphorylation order, ordered dephosphorylation |
| BIOMD0000000031 | Markevich (2004): ordered phosphorylation with two distinct kinases |

Models 27, 29 and 31 have constant enzyme species whose `boundaryCondition` is
false. The loader must still retain their values in reaction expressions.
The [manifest](biomodels/manifest.json) records SHA-256 hashes, exact species order,
initial values, source URLs and publication DOIs. Scientific references:
[Kholodenko](https://doi.org/10.1046/j.1432-1327.2000.01197.x),
[Markevich et al.](https://doi.org/10.1083/jcb.200308060).

## Licensing

The five BioModels files are covered by the BioModels repository's
[CC0 dedication](https://biomodels.org/termsofuse). Original XML annotations and
historical notices are preserved. See the manifest for the notice in each file.
CC0 applies to the encoded models, not to the cited journal articles.

The seven original analytic XML fixtures were authored by Morgan Hough and are
reused under the accompanying [MIT license](LICENSE-MIT). Jaxkineticmodel's
existing Apache-2.0 license remains unchanged.
