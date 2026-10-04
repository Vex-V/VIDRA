"""What every aggregator shares -- the way video_rag's components share `shared/`.

    base        the protocols, `Context` (a record as one object) and
                `DefinitionRunner`, which every definition kind subclasses
    inputs      the selection grammar: `,` `+` `[a,b]`, and reading one
                against a record into rows
    record      a video's folder -- or a combination's, or a mapping of
                document paths -- opened as a `Context`
    rendering   how rows and spans read to a model

Nothing here is an aggregator. Each aggregator is a folder beside this one.
"""
