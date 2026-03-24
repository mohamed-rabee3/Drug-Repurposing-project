"""
TxGNN still uses pandas.DataFrame.append (removed in pandas 2.0).

Call apply() before importing txgnn so complex_disease splits and related
utils keep working without editing site-packages.
"""

from __future__ import annotations

import pandas as pd


def apply() -> None:
    if hasattr(pd.DataFrame, "append"):
        return

    def append(  # noqa: ANN001 — mirrors legacy pandas API
        self,
        other,
        ignore_index: bool = False,
        verify_integrity: bool = False,
        sort: bool = False,
    ):
        return pd.concat(
            [self, other],
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            sort=sort,
        )

    pd.DataFrame.append = append  # type: ignore[method-assign]
