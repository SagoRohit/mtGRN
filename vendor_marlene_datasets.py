"""
Verbatim vendored copy of `load_trrust`/`load_regnetwork` from
`marlene/Marlene/marlene/datasets.py` -- FROZEN, byte-identical to the
source, never modified (same "adapt the caller, never the frozen source"
convention as every other frozen-code reuse in this project).

WHY VENDORED HERE instead of importing from the sibling marlene/ project:
train_mesc_validation.py originally reached across to
`marlene/Marlene/marlene/datasets.py` via a relative sys.path insert,
which meant running this on Kaggle required uploading marlene/Marlene/ as
a second, separately-structured folder alongside mtgrn/. Per user
instruction, mtgrn/ needs to be a single, self-contained folder that can
be pushed/uploaded on its own -- so this file copies ONLY the two
functions actually used (load_trrust, load_regnetwork; NOT
scRNATimeSeriesDataset or remove_bad_rows_columns, which mtgrn/ doesn't
need) verbatim from the source, plus vendors the two raw data files
themselves (`data/trrust_rawdata.mouse.tsv`, `data/RegNetwork-mouse.source`,
copied unmodified from marlene/Marlene/data/) into mtgrn/data/ so the
`data/trrust_rawdata.<species>.tsv` / `data/RegNetwork-<species>.source`
relative paths below resolve correctly when the caller chdirs into
mtgrn/ (this file's own directory) before calling these functions -- see
train_mesc_validation.py's call site.

If marlene/Marlene/marlene/datasets.py's load_trrust/load_regnetwork are
ever changed, this copy will silently drift -- same accepted risk as
ritini/'s vendored ritini_repo/ clone.

Source: marlene/Marlene/marlene/datasets.py (as of 2026-09-09).
"""
from typing import Literal

import numpy as np
import pandas as pd


def load_trrust(
    species: Literal['human', 'mouse'] = 'human',
    adata=None,
    index_adata: bool = True,
):
    """Loads the TRRUST database. If adata is provided and index_adata
    is True, will also remove non-overlapping genes from adata."""
    path = f'data/trrust_rawdata.{species}.tsv'
    trrust = pd.read_csv(path, sep='\t', header=None)
    if adata is None:
        return trrust

    trrust[0] = trrust[0].str.upper()
    trrust[1] = trrust[1].str.upper()

    adata.var_names = np.char.upper(adata.var_names.to_numpy().astype(str))

    # Keep only links where both the TF and target can be found in adata
    trrust = trrust[(np.isin(trrust[0], adata.var_names))
                    & (np.isin(trrust[1], adata.var_names))]
    trrust_links = list(set(zip(trrust[0], trrust[1])))

    all_unq_trrust_genes = pd.unique(trrust[[0, 1]].values.ravel('K'))
    common_genes = np.intersect1d(all_unq_trrust_genes, adata.var_names)
    print(f"Found {len(common_genes)} genes in common.")
    if index_adata:
        adata._inplace_subset_var(common_genes)
        adata.var['is_TF'] = np.isin(adata.var_names, trrust[0].unique())
        print(f"Using {adata.var['is_TF'].sum()} TFs")
    return trrust_links


def load_regnetwork(
    species: Literal['human', 'mouse'] = 'human',
    adata=None,
    index_adata: bool = True,
):
    """Loads the RegNetwork database. If adata is provided and index_adata
    is True, will also remove non-overlapping genes from adata."""
    path = f'data/RegNetwork-{species}.source'
    regnetwork = pd.read_csv(path, sep='\t', header=None)
    regnetwork = regnetwork[~(
        (regnetwork[0].str.startswith('hsa'))
        | (regnetwork[2].str.startswith('hsa'))
        | (regnetwork[0].str.startswith('mnu'))
        | (regnetwork[2].str.startswith('mnu'))
    )]
    if adata is None:
        return regnetwork

    regnetwork[0] = regnetwork[0].str.upper()
    regnetwork[2] = regnetwork[2].str.upper()

    adata.var_names = np.char.upper(adata.var_names.to_numpy().astype(str))

    tf_names = (
        adata.var_names if 'is_TF' not in adata.var
        else adata.var_names[adata.var['is_TF']]
    )
    regnetwork = regnetwork[
        (np.isin(regnetwork[0], tf_names))
        & (np.isin(regnetwork[2], adata.var_names))
    ]

    regnetwork_links = list(set(zip(regnetwork[0], regnetwork[2])))

    all_unq_regnetwork_genes = pd.unique(regnetwork[[0, 2]].values.ravel('K'))
    common_genes = np.intersect1d(all_unq_regnetwork_genes, adata.var_names)
    print(f"Found {len(common_genes)} genes in common.")
    if index_adata:
        adata._inplace_subset_var(common_genes)
        adata.var['is_TF'] = np.isin(adata.var_names, regnetwork[0].unique())
        print(f"Using {adata.var['is_TF'].sum()} TFs")
    return regnetwork_links
