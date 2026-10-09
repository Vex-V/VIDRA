"""The batch pipeline: a whole file in, every stage in order, one folder of
documents out.

    media        1  what streams the file carries
    audio        2  the soundtrack, scanned whole
    boundaries 3+4  the grid, from the picture or the soundtrack
    video        5  which frames each sampler keeps
    cut          6  the transcript, onto the grid
    describe     7  one model answer per (chunk, sampler:question)
    glance      7b  kept frames to vectors, with no answer (beside or instead)
    embed        8  both modalities to vectors

`driver.video_rag()` runs them all. Everything a stage shares with other
pipelines -- the samplers, the question vocabulary, the frame store, search --
is in `core`.
"""
