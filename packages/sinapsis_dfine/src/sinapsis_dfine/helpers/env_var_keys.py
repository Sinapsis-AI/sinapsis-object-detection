# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _DFineEnvVars(BaseModel):
    """Environment variables for the Sinapsis D-FINE package."""

    ALLOW_UNVETTED_DATASETS: EnvVarEntry = EnvVarEntry(
        var_name="ALLOW_UNVETTED_DATASETS",
        default_value=True,
        allowed_values=[True, False, 1, 0],
        description=(
            "If `True`, dataset license validation is skipped. This is the default to "
            "facilitate local development. For production environments, this must be "
            "explicitly set to `False` to enforce license checks."
        ),
        return_type=bool,
    )


DFineEnvVars = _DFineEnvVars()

doc_str = return_docs_for_vars(DFineEnvVars, docs=doc_str, string_for_doc="""Sinapsis D-FINE env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """To use as an import when updating the value is not important."""
    if name in DFineEnvVars.model_fields:
        return DFineEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


__all__ = (*list(DFineEnvVars.model_fields.keys()), "DFineEnvVars")
