# Reproduction notes

## Scientific configuration

The selected code corresponds to the 100-scenario synthetic benchmark, the final hard-nonnegative backbone comparison, the two-stage 00/C/H/HC and dual H/HC models, and the four-arm source design in the October 23 manuscript files. Experimental routes that were not used in those manuscript versions are excluded.

The published model variants retain the actual implementation: the AFNO-style block uses dense spectral channel mixing; the FNO/UNO spectral implementation retains its original frequency selection; DeepONet uses the fixed-grid branch/trunk implementation. The release does not replace these with differently parameterized library models.

The numerical grid uses 20 m horizontal cells and 2, 5 or 9 numerical layers. The final manuscript allocation is recorded per scenario in `configs/paper.json`. The generator directly constructs those recorded subdivisions. It does not repeat the historical sequence of failed numerical attempts that led to selecting the subdivisions.

The simulator uses upstream advection, full XT3D dispersion, 180-day periods and 10-day internal steps. Exported concentrations use saturated pore-water-volume weighting and are clipped after aggregation. Both raw and clipped coarse concentrations are generated. Raw cell-budget residual and raw negative-mass acceptance rules remain in the generation code.

## Input and normalization conventions

The learning grid contains two layers and 30 × 50 cells. Static channels are conductivity, layer top, layer bottom, porosity, x and y; forcing channels are recharge, evapotranspiration, river stage/conductance/bottom, well flow, injected concentration, CHD mask and well mask. The numerical coordinate orientation and the original network coordinate features have different row conventions; the network input representation is retained.

The old export represented recharge as a placeholder and CHD membership in explicitly specified periods only. The training loader reconstructs actual prescribed recharge and inherited CHD values, then fits their corrected moments using training scenarios only. This behavior matches the original training workflow. No old solved states are needed to construct these inputs.

Head receives 36 channels; conditioned concentration receives 38; dual-branch concentration receives 36. The day-180 head and concentration initialize the trajectory. Subsequent steps use model predictions, without reference-state resets.

## Training and checkpoint selection

The unified `model/train.py` entry point calls the retained fit and rollout functions. The substantive training operations, model capacity, projection, loss weighting, checkpoint criterion and early-stopping convention are preserved. The packaging changes replace historical launchers and their dependence on existing run directories.

The entry point uses the selected learning rate 0.0003 for the final models. It does not repeat the two-candidate learning-rate search. The flow and transport residual scale constants in `configs/paper.json` are the fixed values used in the paper, rather than newly tuned values.

All concentration backbones share the same data-only AFNO head for a seed. Two-stage H/HC share the flow-guided AFNO head. Joint dual H/HC runs train both networks; their transport term detaches head. The original system-MSE checkpoint selection is retained even where the historical minimum concentration NRMSE occurred at a different epoch. An equal maximum epoch count is not an equal realized update budget.

## Checks performed during packaging

- Reconstructed static/forcing inputs, masks, time coordinates and cell volumes were compared directly with the historical prepared inputs for **all 100 baseline scenarios**. The arrays matched exactly.
- Source multipliers were compared with the original prescribed protocols for **20 bases × five source modes** (the four arms and intermittent release). They matched to floating-point precision.
- For `dual_0`, regenerated numerical-model geometry, materials, initial states and prescribed list boundaries matched the original accepted model; the seeded conductivity array matched exactly. Re-exporting its existing solver outputs through the new input-independent exporter reproduced all numerical arrays exactly. No solver was launched for this check.
- For the five-layer `dual_7` case, regenerated DIS, NPF, IC, STO, MST, DSP, WEL, CHD, RIV, EVT and RCHA input texts matched the accepted model after ignoring generated comments.
- Five backbone concentration rollouts completed backward propagation and an optimizer update; hard nonnegativity and finite gradients were checked.
- Two-stage and dual rollouts were exercised across a truncated-backpropagation boundary. Head gradients remained absent for the frozen two-stage route and present for the dual route.
- Existing local flow/transport tests exercise cellwise residuals, source/storage balance, CHD treatment, tensor-dispersion behavior and residual gradients. Early-stopping tests cover the original patience rule.
- All **13 included tests** passed in the checked Python 3.11 / PyTorch 2.9.1 CUDA environment. The Python distribution also built successfully as a wheel.

## Limitations

These checks verify code packaging and selected numerical contracts. They do not reproduce the paper's final accuracy tables: the complete MODFLOW simulation campaign, complete training matrix, retained-test inference and SHAP calculation were not rerun during packaging.

Exact earlier random draws for the original forcing templates are not recoverable from the available notebook. The fixed prescribed inputs are therefore included. Seeded heterogeneous conductivity generation remains executable. Solver output formatting and floating-point precision can introduce small differences between generated double-precision K arrays and single-precision learning-grid exports.

This release covers data generation and training as requested. Figure production and SHAP postprocessing are separate from this code package. No publication DOI or software license has yet been assigned.

## Source provenance

| Release component | Original implementation |
|---|---|
| Conductivity generation | `wr_jh_operator/random_k.py` |
| Horizontal and vertical refinement | `refine_two_layer_pilot.py`, `repair_vertical_r3.py` |
| Learning-grid export | `export_formal_scenario.py` |
| Source schedules | `run_source_timing.py`, four-arm logic in `wrr_revision/generate_data.py` |
| Head/backbone training | `run_e1_head_v3.py`, `run_e1_concentration_v3.py`, `run_e1_five_models.py` |
| Framework comparison | `run_nonnegative_oneway.py` |
| Guided head and residuals | `run_pde_ablation_e1.py`, `pde_batch.py`, `e23_physics.py`, `src/gwno/physics/` |
| Architecture implementations | `src/gwno/models/` |

Fixed configurations were extracted from the 100 `new_dual_guass_conc_models/dual_*` input directories. Input files are stored once when their content is identical; only FloPy's generated timestamp comment was removed. The archive index restores the original package filenames.
