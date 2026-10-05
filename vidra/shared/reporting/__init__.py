"""How a run tells its caller something.

    errors      what was refused, and why -- every deliberate failure is a
                `VidraError`, keeping its builtin base
    logs        what happened -- records at the choke points, never a warning
                about something that works
    progress    how far it has got -- events from inside the long stages

`errors` imports nothing, so a document can refuse with the library's own
error without growing an edge.
"""
