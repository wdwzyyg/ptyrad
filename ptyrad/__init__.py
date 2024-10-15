# __init__.py is not strictly needed since Python 3.3 for the intepreter to recognize it as a package 
# However, this is still included for clarity and for initialization purpose

from ptyrad.utils import vprint

__version__ = "v0.1.0-beta2.8" # 2024.10.14
vprint(f"\nPtyRAD Version: {__version__}")