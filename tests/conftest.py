import sys
import os
import warnings

# Ensure the project root is on sys.path so tests import the local package.
# This mirrors what pytest does for editable installs but is robust in CI
# where tests may be executed from other working directories.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

# Suppress noisy external deprecation warnings that appear during imports in
# some environments (pkg_resources / scikits) so focused package tests stay clean.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*scikits.*")
# Suppress common sklearn Future/Deprecation warnings that are not actionable here
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sklearn.*")
