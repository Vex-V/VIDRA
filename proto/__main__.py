"""`python -m proto` -> the pipeline over one folder.

A module rather than running `driver.py` directly: importing the package
already loads `driver`, so `python -m proto.driver` would execute it twice
and Python warns about exactly that.
"""

from .driver import main

raise SystemExit(main())
