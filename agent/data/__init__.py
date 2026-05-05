"""Loaders for the operational data files under `operational/`.

Each submodule wraps one JSON source with a typed query API. Loaders cache
the parsed file at module level — the agent reloads the catalog only on
process restart, matching how a real call-center service would treat the
CMMS export.
"""
