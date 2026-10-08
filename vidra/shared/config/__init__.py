"""Where things are, and what a process was told before it ran.

    paths       where things live; the only module that knows a filename
    env         reading a `.env` -- by an entry point, or when asked
    settings    `vidra.configure()`: the data root, the weights, which
                `.env`, the Hugging Face token
    lookup      a class named as `module:Class`, imported when first asked
                for, and what its constructor accepts
"""
