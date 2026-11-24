"""ntnconnect.common.capabilities - print the full menu of selectable components.

The simulator is general: a scenario composes a pipeline by NAME from the
registries (stages, validators, interference modes, channel models, steering /
ISL strategies, hooks ...). This module surfaces every option in one place so a
scenario author knows exactly what they can pick.

    python -m ntnconnect.common.capabilities            # human-readable
    python -m ntnconnect.common.capabilities --json     # machine-readable

Importing the package triggers stage/validator/mode self-registration first so
the lists are complete.
"""
from __future__ import annotations
import json
import sys


def _ensure_registered() -> None:
    """Force the side-effect imports that populate the registries."""
    import importlib
    mods = ["ntnconnect.common.validation",            # @register_validator
            "ntnconnect.registry.channels.access_models"]  # @register_access_model
    try:  # reuse the engine's own stage-module list so the menu == what runs
        from ntnconnect.engine.engine import (
            DEFAULT_STAGE_MODULES, DEFAULT_RADIO_ALGO_MODULES,
            DEFAULT_CHANNEL_PREDICTOR_MODULES, DEFAULT_HOOK_MODULES,
        )
        mods += (DEFAULT_STAGE_MODULES + DEFAULT_RADIO_ALGO_MODULES
                 + DEFAULT_CHANNEL_PREDICTOR_MODULES + DEFAULT_HOOK_MODULES)
    except Exception:
        mods += ["ntnconnect.engine.stages.external_interference"]
    for mod in mods:
        try:
            importlib.import_module(mod)
        except Exception:
            pass


def catalog() -> dict:
    _ensure_registered()
    from ntnconnect.registry.registry import describe_registries
    from ntnconnect.common.validation import validator_catalog
    cat = describe_registries()
    cat["_validator_docs"] = validator_catalog()
    return cat


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cat = catalog()
    if "--json" in argv:
        print(json.dumps(cat, indent=2))
        return 0
    docs = cat.pop("_validator_docs", {})
    print("=" * 64)
    print(" SkyNTNPro v1 — selectable components (pick by name per scenario)")
    print("=" * 64)
    for group, names in cat.items():
        print(f"\n{group}  ({len(names)})")
        for n in names:
            if group == "validators" and n in docs:
                print(f"    • {n:<24} [{docs[n]['category']}] {docs[n]['doc']}")
            else:
                print(f"    • {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
