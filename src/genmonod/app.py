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

from genmonod.config import default_config, SystemConfig, add_dosing_event, add_passage_event, add_production
from genmonod.data_io import Dataset
from genmonod.fitting import fit, pack_free_params, pack_free_params_with_y0
from genmonod.strain_library import record_strain_params, apply_library, DEFAULT_LIBRARY_PATH
from genmonod.fit_store import record_fit, DEFAULT_STORE_PATH, _data_summary
from genmonod.amortized_model import guess_initial_params
from genmonod.plotting import plot_fit
from genmonod.model_selection import compare_structures, summarize

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
    st.session_state.schedule_rows = []  # old events may reference now-removed strains/metabolites

cfg: SystemConfig = st.session_state.cfg

with st.sidebar.expander("Rename strains / metabolites / toxins"):
    for i in range(n_strains):
        cfg.strain_names[i] = st.text_input(f"Strain {i+1} name", value=cfg.strain_names[i], key=f"sname_{i}")
    for k in range(n_metabolites):
        cfg.metabolite_names[k] = st.text_input(f"Metabolite {k+1} name", value=cfg.metabolite_names[k], key=f"mname_{k}")
    for l in range(n_toxins):
        cfg.toxin_names[l] = st.text_input(f"Toxin {l+1} name", value=cfg.toxin_names[l], key=f"tname_{l}")

# keep cfg.subsets' keys in sync with strain_names -- renaming a strain
# above does NOT retroactively update subsets' dict keys on its own, so
# without this a rename would silently orphan any overlap grouping
# configured under the old name
_combined = cfg.combined_substrate_names()
for _s in cfg.strain_names:
    if _s not in cfg.subsets:
        cfg.subsets[_s] = [[n] for n in _combined]
for _stale in [k for k in cfg.subsets if k not in cfg.strain_names]:
    del cfg.subsets[_stale]

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


tabs = st.tabs(["Growth / Toxin (r, K, c)", "Subsets (metabolic overlap)", "Production", "Strain-strain", "Environment"])

combined_names = cfg.combined_substrate_names()

with tabs[0]:
    st.caption(
        "One combined axis: metabolites, then toxins. A toxin is just a substrate "
        "with a NEGATIVE r (it subtracts from growth instead of adding to it) — "
        "same matrices, no separate toxin-specific editor needed."
    )
    if combined_names:
        st.subheader("r — growth rate constant (metabolite: usually >0; toxin: usually <0)")
        _matrix_editor("r", cfg.r, cfg.strain_names, combined_names, "r_edit")
        st.subheader("K — Monod constant (shared between growth and consumption/uptake)")
        _matrix_editor("K", cfg.K, cfg.strain_names, combined_names, "K_edit")
        st.subheader("c — consumption / uptake rate")
        _matrix_editor("c", cfg.c, cfg.strain_names, combined_names, "c_edit")
    else:
        st.info("Add at least one metabolite or toxin in the sidebar to configure growth/uptake.")

with tabs[1]:
    st.caption(
        "Which substrates SHARE a saturation denominator (compete for uptake capacity) "
        "for a given strain — this is precisely what 'metabolic overlap' means here. "
        "Two substrates in the same group compete; substrates not grouped with anything "
        "saturate independently. Group a toxin with a metabolite to represent 'toxin "
        "competes with metabolites for uptake' instead of having an independent supply."
    )
    if len(combined_names) >= 2:
        sel_strain = st.selectbox("Strain", cfg.strain_names, key="subset_strain")
        current_groups = cfg.subsets.get(sel_strain, [[n] for n in combined_names])
        st.caption(f"Current groups for {sel_strain}: " + " | ".join("+".join(g) for g in current_groups))
        group_pick = st.multiselect(
            "Select substrates to group together (competing for shared uptake)",
            combined_names, key="subset_pick",
        )
        if st.button("Set this group") and len(group_pick) >= 2:
            others = [n for n in combined_names if n not in group_pick]
            cfg.subsets[sel_strain] = [group_pick] + [[n] for n in others]
            st.rerun()
        if st.button("Reset to independent (no overlap) for this strain"):
            cfg.subsets[sel_strain] = [[n] for n in combined_names]
            st.rerun()
    else:
        st.info("Need at least 2 metabolites/toxins combined to configure overlap.")

with tabs[2]:
    st.caption(
        "A strain SYNTHESIZING one substrate from another (the paper's 'ς' term) — "
        "different from growth/consumption: this saturates on the PRECURSOR's own "
        "concentration via its own Monod constant."
    )
    if "production_rows" not in st.session_state:
        st.session_state.production_rows = []
    if combined_names:
        pc = st.columns(3)
        p_product = pc[0].selectbox("Product", combined_names, key="prod_product")
        p_strain = pc[1].selectbox("Producing strain", cfg.strain_names, key="prod_strain")
        p_precursor = pc[2].selectbox("Precursor", combined_names, key="prod_precursor")
        if st.button("Add production pathway"):
            st.session_state.production_rows.append({"product": p_product, "strain": p_strain, "precursor": p_precursor})

    cfg.production = []
    if st.session_state.production_rows:
        for i, row in enumerate(st.session_state.production_rows):
            rcols = st.columns([5, 1])
            rcols[0].write(f"{row['strain']} produces {row['product']} from {row['precursor']}")
            add_production(cfg, row["product"], row["strain"], row["precursor"])
            if rcols[1].button("Remove", key=f"rm_prod_{i}"):
                st.session_state.production_rows.pop(i)
                st.rerun()
    else:
        st.caption("No production pathways yet.")

with tabs[3]:
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

with tabs[4]:
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

    st.subheader("Discrete events (dosing / passage)")
    st.caption(
        "The supply/dilution rates above model a CONTINUOUS process (e.g. a true "
        "chemostat feed). Use this instead for a DISCRETE one-time or repeated event — "
        "a sugar addition, or a passage/dilution step (which also dilutes strain "
        "populations, not just a metabolite/toxin — supply/dilution above can't do that)."
    )
    if "schedule_rows" not in st.session_state:
        st.session_state.schedule_rows = []

    ev_cols = st.columns([2, 2, 2, 2, 1])
    ev_type = ev_cols[0].selectbox("Event type", ["Dosing (add/set one metabolite or toxin)", "Passage (dilute everything)"], key="ev_type")
    ev_time = ev_cols[1].number_input("Time", value=0.0, key="ev_time")
    if ev_type.startswith("Dosing"):
        ev_target = ev_cols[2].selectbox("Target", cfg.metabolite_names + cfg.toxin_names, key="ev_target")
        ev_kind = ev_cols[3].selectbox("Kind", ["add", "set"], key="ev_kind")
        ev_amount = ev_cols[4].number_input("Amount", value=1.0, key="ev_amount")
    else:
        ev_target, ev_kind = None, None
        ev_amount = ev_cols[2].number_input("Dilution factor (e.g. 0.1 = 10-fold)", value=0.1, min_value=0.0, max_value=1.0, key="ev_dilfactor")

    if st.button("Add event"):
        if ev_type.startswith("Dosing"):
            st.session_state.schedule_rows.append({"type": "dosing", "time": ev_time, "target": ev_target, "kind": ev_kind, "amount": ev_amount})
        else:
            st.session_state.schedule_rows.append({"type": "passage", "time": ev_time, "dilution_factor": ev_amount})

    cfg.schedule = []
    if st.session_state.schedule_rows:
        for i, row in enumerate(st.session_state.schedule_rows):
            cols = st.columns([5, 1])
            if row["type"] == "dosing":
                cols[0].write(f"t={row['time']}: {row['kind']} {row['amount']} to {row['target']}")
                add_dosing_event(cfg, row["time"], row["target"], row["kind"], row["amount"])
            else:
                cols[0].write(f"t={row['time']}: dilute EVERYTHING by factor {row['dilution_factor']}")
                add_passage_event(cfg, row["time"], row["dilution_factor"])
            if cols[1].button("Remove", key=f"rm_ev_{i}"):
                st.session_state.schedule_rows.pop(i)
                st.rerun()
    else:
        st.caption("No discrete events yet — continuous supply/dilution only.")

n_free, _, _ = pack_free_params(cfg)
st.caption(f"Current system: {len(n_free)} free parameters to fit (plus one per unmeasured initial condition, once data is uploaded below).")

if st.button("Fill known strains from library"):
    cfg, filled = apply_library(cfg, store_path=DEFAULT_LIBRARY_PATH)
    st.session_state.cfg = cfg
    if filled:
        st.success(f"Filled {len(filled)} entries from previously-fit strains: " +
                   ", ".join(f"{attr}{subject}" for attr, subject, _ in filled[:8]) +
                   (" ..." if len(filled) > 8 else ""))
    else:
        st.info("No matching strain names found in the library yet — nothing to fill.")
    st.rerun()

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
    # 3b. STRUCTURE SEARCH — "should toxin/metabolic overlap/etc be in the
    # model at all?" answered automatically, ranked by AICc, instead of you
    # having to guess or build each variant by hand.
    # ---------------------------------------------------------------------------
    with st.expander("Not sure which processes to include? Search structures automatically"):
        st.caption(
            "Fits several structural variants (toxin present or not, one shared "
            "metabolite vs. one per strain) against the data above and ranks them "
            "by AICc — lower is better. This can take a while (each variant is a "
            "full fit) and reuses the matrix settings above as the base, except "
            "for the axes being searched."
        )
        search_toxin = st.checkbox("Search: toxin present vs. absent", value=(cfg.n_toxins > 0 or True))
        search_overlap = st.checkbox("Search: shared vs. separate metabolite pool", value=True)
        search_max_nfev = st.number_input("Max iterations per candidate fit", min_value=20, max_value=500, value=150)

        if st.button("Run structure search"):
            with st.spinner("Fitting each structural variant..."):
                # rebuild from the raw uploaded bytes each time, since
                # "separate metabolite" candidates need a different number
                # of metabolite columns than "shared" ones
                raw_bytes = [f.getvalue() for f in uploaded_files]

                def _builder(candidate_cfg):
                    import io
                    built = []
                    for name, content in zip([f.name for f in uploaded_files], raw_bytes):
                        built.append(Dataset.from_csv(io.BytesIO(content), time_col, column_map, candidate_cfg, name=name))
                    return built

                search_results = compare_structures(
                    _builder, n_strains=cfg.n_strains, strain_names=cfg.strain_names,
                    include_toxin=(False, True) if search_toxin else (cfg.n_toxins > 0,),
                    metabolic_overlap=("shared", "separate") if search_overlap else ("shared",),
                    max_nfev=search_max_nfev,
                )
            st.text(summarize(search_results))
            if search_results and not search_results[0].reliable:
                st.warning(
                    "The top-ranked structure has too few observed data points relative to its "
                    "free parameters for AICc to be a reliable basis for comparison. This is common "
                    "with a single small dataset — consider uploading more replicate files (fit "
                    "jointly, above) before trusting the ranking."
                )

# ---------------------------------------------------------------------------
# 4. FIT
# ---------------------------------------------------------------------------
st.header("Fit")
use_amortized = st.checkbox(
    "Use past fits to propose a starting guess (needs several prior fits of this same shape)",
    value=True,
)
save_to_store = st.checkbox("Save this fit for future guessing (same-shape systems)", value=True)
save_to_library = st.checkbox("Save this fit's strain values to the library (for building larger systems later)", value=True)

if st.button("Run fit", type="primary", disabled=(len(datasets) == 0)):
    n_datasets = len(datasets)
    with st.spinner(f"Fitting jointly across {n_datasets} dataset(s)..."):
        init_guess = None
        if use_amortized:
            n_free, _, _, _ = pack_free_params_with_y0(cfg, datasets)
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
    if save_to_library:
        n_written = record_strain_params(result, DEFAULT_LIBRARY_PATH)
        st.caption(f"Saved {n_written} strain-keyed values to the library.")

if "last_result" in st.session_state:
    result = st.session_state.last_result
    st.success(f"Fit {'converged' if result.success else 'stopped early'} — cost: {result.cost:.4f}")

    for ds, traj in zip(result.datasets, result.trajectories):
        fig = plot_fit(result.config, ds, traj)
        st.pyplot(fig)

    st.subheader("Fitted parameter values (shared across all datasets above)")
    for attr in ["r", "K", "c", "mutation", "translocation"]:
        spec = getattr(result.config, attr)
        if spec is not None:
            st.write(attr)
            st.dataframe(spec.values)
    if result.config.production:
        st.write("production")
        st.dataframe(pd.DataFrame(result.config.production)[["product", "strain", "precursor", "rate", "half_sat"]])
    if result.config.conjugative_transfer:
        st.write("conjugative_transfer")
        st.dataframe(pd.DataFrame(result.config.conjugative_transfer)[["product", "donor", "recipient", "rate"]])
