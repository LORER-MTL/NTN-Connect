# Compatibility shim. Real implementation + registration is in._extras.
# Importing this module simply triggers _extras to load.
from ntnconnect.analysis.hooks import _extras  # noqa: F401
