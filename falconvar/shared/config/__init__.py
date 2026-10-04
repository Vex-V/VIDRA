"""Where things are, and what a process was told before it ran.

    paths       where things live; the only module that knows a filename
    env         reading a `.env` -- by an entry point, or when asked
    settings    `falconvar.configure()`: the data root, the weights, which
                `.env`, the Hugging Face token
"""
