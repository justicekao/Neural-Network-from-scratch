"""
app.py

The visual fitting tool. Run it with:

    genmonod-app

(or `streamlit run src/genmonod/app.py` if you're developing locally).

This is the file you'll actually spend time in as a user — everything
else in the package is the engine underneath it. Sections below are
labeled so you can find your way around; there's very little here that
needs editing since the whole point is that you configure the system
FROM the app, not by editing this file.

NEEDS YOUR INPUT: nothing to run it. If you want to change bounds/
defaults for newly-created matrices, that's in config.py's
`default_config`, not here.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from genmonod.config import default_config, SystemConfig
from genmonod.data_io import Dataset
from genmonod.fitting import fit, pack_free_params
from genmonod.fit_store import record_fit, DEFAULT_STORE_PATH, _data_summary
from genmonod.amortized_model import guess_initial_params
from genmonod.plotting import plot_fit

st.set_page_config(page_title="genmonod — multi-strain Monod fitter", layout="wide")
st.title("Generalized Monod System Fitter")

# ---------------------------------------------------------------------------
# 1. SYSTEM SETTINGS — dimensions and which optional processes are on
# ---------------------------------------------------------------------------
st.sidebar.header("System settings")

n_strains = st.sidebar.number_input("Number of strains", min_value=1, max_value=12, value=2)
n_metabolites = st.sidebar.number_input("Number of metabolites", min_value=0, max_value=6, value=1)
n_toxins = st.sidebar.number_input("Number of toxins", min_value=0, max_value=6, value=0)
include_mutation = st.sidebar.checkbox("Include mutation (strain-to-strain)", value=False)
include_translocation = st.sidebar.checkbox("Include translocation (strain-to-strain, contact-based)", value=False)

dims_key = (n_strains, n_metabolites, n_toxins, include_mutation, include_translocation)
if "dims_key" not in st.session_state or st.session_state.dims_key != dims_key:
    # dimensions changed -> rebuild a fresh, fully-free config.
    # NOTE: this resets any matrix edits you made under the old dimensions.
    st.session_state.cfg = default_config(
        n_strains, n_metabolites, n_toxins, include_mutation, include_translocation
    )
    st.session_state.dims_key = dims_key

cfg: SystemConfig = st.session_state.cfg

with st.sidebar.expander("Rename strains / metabolites / toxins"):
    for i in range(n_strains):
        cfg.strain_names[i] = st.text_input(f"Strain {i+1} name", value=cfg.strain_names[i], key=f"sname_{i}")
    for k in range(n_metabolites):
        cfg.metabolite_names[k] = st.text_input(f"Metabolite {k+1} name", value=cfg.metabolite_names[k], key=f"mname_{k}")
    for l in range(n_toxins):
        cfg.toxin_names[l] = st.text_input(f"Toxin {l+1} name", value=cfg.toxin_names[l], key=f"tname_{l}")

# ---------------------------------------------------------------------------
# 2. MATRIX EDITORS — leave a cell blank to fit it freely, type a number to
#    fix it. This is the visual replacement for hand-editing config.py.
# ---------------------------------------------------------------------------
st.header("Parameters")
st.caption("Leave a cell blank to fit it. Type a number to fix it at that value.")


def _matrix_editor(label: str, spec, row_names: list[str], col_names: list[str], key: str):
    """Renders one matrix as an editable table and writes edits back into `spec.values`."""
    df = pd.DataFrame(spec.values, index=row_names, columns=col_names)
    edited = st.data_editor(df, key=key, use_container_width=True)
    spec.values[:, :] = edited.to_numpy()
    st.caption(f"{label} — bounds for free entries: [{spec.lower}, {spec.upper}]")


def _vector_editor(label: str, spec, names: list[str], key: str):
    df = pd.DataFrame({label: spec.values}, index=names)
    edited = st.data_editor(df, key=key, use_container_width=True)
    spec.values[:] = edited[label].to_numpy()
    st.caption(f"{label} — bounds for free entries: [{spec.lower}, {spec.upper}]")


tabs = st.tabs(["Growth", "Toxin", "Strain-strain", "Environment"])

with tabs[0]:
    if n_metabolites > 0:
        st.subheader("Growth rate (r)")
        _matrix_editor("growth_rate", cfg.growth_rate, cfg.strain_names, cfg.metabolite_names, "r_edit")
        st.subheader("Half-saturation (Ks)")
        _matrix_editor("growth_half_sat", cfg.growth_half_sat, cfg.strain_names, cfg.metabolite_names, "ks_edit")
        st.subheader("Consumption")
        _matrix_editor("consumption", cfg.consumption, cfg.strain_names, cfg.metabolite_names, "cons_edit")
    else:
        st.info("Add at least one metabolite in the sidebar to configure growth terms.")

with tabs[1]:
    if n_toxins > 0:
        st.subheader("Toxin kill rate (P)")
        _matrix_editor("toxin_kill_rate", cfg.toxin_kill_rate, cfg.strain_names, cfg.toxin_names, "p_edit")
        st.subheader("Toxin half-saturation (K)")
        _matrix_editor("toxin_half_sat", cfg.toxin_half_sat, cfg.strain_names, cfg.toxin_names, "k_tox_edit")
        st.subheader("Secretion")
        _matrix_editor("secretion", cfg.secretion, cfg.strain_names, cfg.toxin_names, "secr_edit")
    else:
        st.info("Add at least one toxin in the sidebar to configure toxin terms.")

with tabs[2]:
    st.subheader("Mortality")
    _vector_editor("mortality", cfg.mortality, cfg.strain_names, "delta_edit")
    if include_mutation:
        st.subheader("Mutation matrix — entry [i, j] = per-capita rate strain j mutates into strain i")
        st.caption("Diagonal is always 0 (no self-mutation) regardless of what you type here.")
        _matrix_editor("mutation", cfg.mutation, cfg.strain_names, cfg.strain_names, "mut_edit")
        np.fill_diagonal(cfg.mutation.values, 0.0)
    if include_translocation:
        st.subheader("Translocation matrix — entry [i, j] = contact rate strain i converts strain j")
        st.caption("Diagonal is always 0 (no self-translocation) regardless of what you type here.")
        _matrix_editor("translocation", cfg.translocation, cfg.strain_names, cfg.strain_names, "tl_edit")
        np.fill_diagonal(cfg.translocation.values, 0.0)
    if not include_mutation and not include_translocation:
        st.info("Enable mutation and/or translocation in the sidebar to configure strain-strain transfer.")

with tabs[3]:
    if n_metabolites > 0:
        st.subheader("Metabolite supply")
        _vector_editor("metabolite_supply", cfg.metabolite_supply, cfg.metabolite_names, "msup_edit")
        st.subheader("Metabolite dilution")
        _vector_editor("metabolite_dilution", cfg.metabolite_dilution, cfg.metabolite_names, "mdil_edit")
    if n_toxins > 0:
        st.subheader("Toxin supply")
        _vector_editor("toxin_supply", cfg.toxin_supply, cfg.toxin_names, "tsup_edit")
        st.subheader("Toxin decay")
        _vector_editor("toxin_decay", cfg.toxin_decay, cfg.toxin_names, "tdec_edit")

n_free, _, _ = pack_free_params(cfg)
st.caption(f"Current system: {len(n_free)} free parameters to fit.")

# ---------------------------------------------------------------------------
# 3. DATA — upload one or more CSVs and map columns to strains/metabolites/
#    toxins. Multiple files are fit JOINTLY: one shared set of parameters
#    explaining all of them at once (e.g. several replicate tubes of the
#    same system), which is generally a much stronger fit than treating
#    each file in isolation.
# ---------------------------------------------------------------------------
st.header("Data")
st.caption(
    "Upload one file per experiment/replicate. All uploaded files are fit "
    "JOINTLY against one shared set of parameters. They must use the same "
    "column names as each other — the mapping below is set up once, using "
    "the first file, and applied to every file you upload."
)
uploaded_files = st.file_uploader("Upload CSV file(s)", type="csv", accept_multiple_files=True)

datasets: list[Dataset] = []
if uploaded_files:
    # NEEDS YOUR INPUT: nothing to use this normally. If your replicate
    # files genuinely use DIFFERENT column names per file (not just
    # different values), this shared-mapping approach won't work as-is —
    # rename columns to be consistent across files before uploading, or
    # extend this loop to build a separate column_map per file.
    first_df = pd.read_csv(uploaded_files[0])
    st.dataframe(first_df.head())

    time_col = st.selectbox("Which column is time?", first_df.columns)
    all_state_names = cfg.strain_names + cfg.metabolite_names + cfg.toxin_names
    column_map = {}
    st.caption("For each state variable, pick the matching CSV column (or 'None' if not measured). This mapping applies to every uploaded file.")
    cols = st.columns(3)
    for i, state_name in enumerate(all_state_names):
        options = ["None"] + [c for c in first_df.columns if c != time_col]
        choice = cols[i % 3].selectbox(state_name, options, key=f"map_{state_name}")
        if choice != "None":
            column_map[state_name] = choice

    for f in uploaded_files:
        f.seek(0)
        datasets.append(Dataset.from_csv(f, time_col, column_map, cfg, name=f.name))

    st.success(f"Loaded {len(datasets)} dataset(s): " + ", ".join(ds.name for ds in datasets))

# ---------------------------------------------------------------------------
# 4. FIT
# ---------------------------------------------------------------------------
st.header("Fit")
use_amortized = st.checkbox(
    "Use past fits to propose a starting guess (needs several prior fits of this same shape)",
    value=True,
)
save_to_store = st.checkbox("Save this fit for future guessing", value=True)

if st.button("Run fit", type="primary", disabled=(len(datasets) == 0)):
    n_datasets = len(datasets)
    with st.spinner(f"Fitting jointly across {n_datasets} dataset(s)..."):
        init_guess = None
        if use_amortized:
            n_free, _, _ = pack_free_params(cfg)
            # the guesser is queried using the FIRST dataset's summary —
            # with replicate data the datasets should look similar enough
            # that this is a reasonable stand-in for "what this shape of
            # data tends to fit to"
            init_guess = guess_initial_params(
                shape_signature=cfg.shape_signature(),
                data_summary=_data_summary(datasets[0]),
                n_free_params=len(n_free),
                store_path=DEFAULT_STORE_PATH,
            )
            if init_guess is not None:
                st.info("Using a starting guess learned from past fits of this same system shape.")

        result = fit(cfg, datasets, init_guess=init_guess)
        st.session_state.last_result = result

    if save_to_store:
        record_fit(result, DEFAULT_STORE_PATH)

if "last_result" in st.session_state:
    result = st.session_state.last_result
    st.success(f"Fit {'converged' if result.success else 'stopped early'} — cost: {result.cost:.4f}")

    for ds, traj in zip(result.datasets, result.trajectories):
        fig = plot_fit(result.config, ds, traj)
        st.pyplot(fig)

    st.subheader("Fitted parameter values (shared across all datasets above)")
    for attr in ["growth_rate", "growth_half_sat", "consumption", "toxin_kill_rate",
                 "toxin_half_sat", "secretion", "mutation", "translocation"]:
        spec = getattr(result.config, attr)
        if spec is not None:
            st.write(attr)
            st.dataframe(spec.values)
