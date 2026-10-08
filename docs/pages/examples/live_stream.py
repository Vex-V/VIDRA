"""A live run end to end: a file sent to a port at the pace it was recorded,
each kept frame answered as it arrives, then both searches -- frame by frame
over the last ten minutes, and chunk by chunk once the stream has ended.

    python live_stream.py path/to/video.mp4
"""

import multiprocessing
import sys
from datetime import datetime, timedelta, timezone

import vidra
from vidra import Folder, LocalEmbedder, Models, OpenAI
from vidra.video_rag import live, search, search_observations, video_rag_live

vidra.configure(env_file=".env")


def show(o):
    """Every answer, as soon as it is ready."""
    print(f"{o.seen_at[11:19]}  t={o.media_ts:5.1f}s  lag {o.lag_s:4.1f}s  "
          f"{o.sampler_id}: {o.description[:70]}")


if __name__ == "__main__":
    video = sys.argv[1]
    models = Models(vlm=OpenAI("gpt-5.4-mini"), embedder=LocalEmbedder())
    db = Folder("data/out")             # reads observations.jsonl while it grows

    # The camera: its own process, sending to a port this run listens on.
    camera = multiprocessing.Process(target=live.send,
                                     args=(video, "tcp://127.0.0.1:9000"))
    camera.start()
    run = video_rag_live("tcp://127.0.0.1:9000?listen=1", "data/out",
                         video_id="camera-1", sampler="clip:overview",
                         context=(1, 0),          # the frame before, for motion
                         stop_after_s=60,         # a minute of the stream
                         on_unit=show, models=models, database=db)
    camera.join()                       # the sender stops when the run closes
    print(f"\n{run.observations} answers, {run.dropped} dropped, lag {run.lag_s}, "
          f"ended: {run.ended}")

    # One hit per kept frame, by when it arrived.
    since = datetime.now(timezone.utc) - timedelta(minutes=10)
    print("\nframes, last ten minutes:")
    for hit in search_observations("a person at the counter", "camera-1", since=since,
                                   limit=3, models=models, database=db):
        print(f"  t={hit.media_ts:5.1f}s  {hit.description[:80]}")

    # The stream ended as an ordinary video, so moment search reads it too.
    moments, notes = search("a person at the counter", "camera-1", models=models,
                            database=db)
    print("\nmoments:")
    for m in moments[:3]:
        print(f"  chunk {m.chunk_id}  {m.start_ts:.0f}-{m.end_ts:.0f}s  {m.score:.3f}")
