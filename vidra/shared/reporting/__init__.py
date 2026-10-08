"""How a run tells its caller something.

    errors      what was refused, and why -- every deliberate failure is a
                `VidraError`, keeping its builtin base. Imports nothing
    logs        what happened -- records at the choke points
    progress    how far it has got -- events from inside the long stages
"""
