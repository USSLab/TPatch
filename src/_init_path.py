import os
import sys
import warnings

try:
    warnings.filterwarnings("ignore")
    root = os.path.dirname(__file__) + "/.."
    os.chdir(root)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
except Exception as e:
    print(e)
    exit(0)
