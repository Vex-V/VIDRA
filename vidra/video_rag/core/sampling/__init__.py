"""Which frames are kept: the decimator, the frame reader, the samplers, and the
spec that names them (`"clip:[text,scene],yolo"`).

A sampler sees one decimated frame at a time and cannot look ahead, so the same
samplers serve a file read end to end and a stream read as it arrives.
"""
