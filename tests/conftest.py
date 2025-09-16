import warnings

# Suppress noisy external deprecation warnings that appear during imports in
# some environments (pkg_resources / scikits) so focused package tests stay clean.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*scikits.*")
# Suppress common sklearn Future/Deprecation warnings that are not actionable here
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sklearn.*")
