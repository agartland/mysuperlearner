import warnings

# Filter noisy external deprecation warnings early during interpreter startup.
# This file is imported by site if present on sys.path before other imports,
# which makes it a reliable place to suppress external package warnings that
# otherwise appear during pytest startup (e.g., pkg_resources, scikits).
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*scikits.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*sklearn.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*sklearn.*")
