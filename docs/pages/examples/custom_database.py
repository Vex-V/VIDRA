"""A database of your own: SQLite, holding the grid, the moment vectors and
the aggregate vectors, and searching both.

Fill in only the hooks you need. A write hook left out does nothing (that
artifact is not copied); a read hook left out raises `Unsupported`.

    python custom_database.py path/to/video.mp4
"""

import json
import math
import sqlite3
import sys

import vidra
from vidra import Database, LocalEmbedder, Models, OpenAI, aggregates
from vidra.video_rag import search, video_rag

vidra.configure(env_file=".env")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def matches(fields, wanted):
    """Every wanted field equal to its value, or a list holding it."""
    return all(fields.get(k) == v or (isinstance(fields.get(k), list) and v in fields[k])
               for k, v in wanted.items())


class SQLite(Database):
    name = "sqlite"                         # what run.problems calls it

    def __init__(self, path="vidra.sqlite"):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            create table if not exists chunks (video_id text, chunk_id int,
                start_ts real, end_ts real, primary key (video_id, chunk_id));
            create table if not exists units (video_id text, chunk_id int,
                sampler_id text, sampler text, question text, content text,
                structured text, embedder text, vector text,
                primary key (video_id, chunk_id, sampler_id, embedder));
            create table if not exists sources (source_id text primary key, video_ids text);
            create table if not exists aggregate_units (source_id text, aggregate_id text,
                item_id text, level text, content text, start_ts real, end_ts real,
                embedder text, vector text);
        """)

    # ---- writes: called as each stage finishes, with the file's JSON as a dict

    def write_timeline(self, video_id, document):
        self.db.execute("delete from chunks where video_id = ?", (video_id,))
        self.db.executemany("insert into chunks values (?, ?, ?, ?)",
                            [(video_id, c["chunk_id"], c["start_ts"], c["end_ts"])
                             for c in document["chunks"]])
        self.db.commit()

    def write_embedded(self, video_id, document):
        self.db.executemany(
            "insert or replace into units values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(video_id, u["chunk_id"], u["sampler_id"], u["sampler"], u["question"],
              u["content"], json.dumps(u["structured"]), document["embedder"],
              json.dumps(u["vector"]))
             for u in document["units"] if u.get("vector")])
        self.db.commit()

    def write_source(self, source):
        self.db.execute("insert or replace into sources values (?, ?)",
                        (source["source_id"], json.dumps(source["video_ids"])))
        self.db.commit()

    def write_aggregate_units(self, source_id, units, embedder_key):
        # Defining this hook is what makes a run embed summaries and chapters.
        ids = {u["aggregate_id"] for u in units}
        self.db.executemany("delete from aggregate_units where source_id = ? "
                            "and aggregate_id = ?", [(source_id, a) for a in ids])
        self.db.executemany(
            "insert into aggregate_units values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(source_id, u["aggregate_id"], u["item_id"], u["level"], u["content"],
              u.get("start_ts"), u.get("end_ts"), embedder_key, json.dumps(u["vector"]))
             for u in units])
        self.db.commit()

    # ---- reads: needed only to search through this database

    def video_ids(self):
        return [row[0] for row in self.db.execute("select distinct video_id from chunks")]

    def spans(self, video_id):
        return [(start, end) for start, end in self.db.execute(
            "select start_ts, end_ts from chunks where video_id = ? order by chunk_id",
            (video_id,))]

    def search(self, vector, query, embedder_key, limit=20, video_ids=None, sampler=None,
               question=None, strategy=None, chunk_ids=None, structured=None):
        rows = []
        for (vid, chunk, sampler_id, sampler_name, asked, content, fields,
             vec) in self.db.execute(
                "select video_id, chunk_id, sampler_id, sampler, question, content, "
                "structured, vector from units where embedder = ?", (embedder_key,)):
            fields = json.loads(fields)
            if ((video_ids and vid not in video_ids)
                    or (sampler and sampler_id != sampler)
                    or (question and asked != question)
                    or (strategy and sampler_name != strategy)
                    or (chunk_ids and chunk not in chunk_ids)
                    or (structured and not matches(fields, structured))):
                continue
            rows.append({"video_id": vid, "chunk_id": chunk, "sampler_id": sampler_id,
                         "sampler": sampler_name, "question": asked, "content": content,
                         "structured": fields,
                         "score": cosine(vector, json.loads(vec))})
        rows.sort(key=lambda r: r["score"], reverse=True)
        return [{**r, "dense_rank": rank, "text_rank": None}
                for rank, r in enumerate(rows[:limit], start=1)]

    def search_aggregates(self, vector, query, embedder_key, level, limit=5,
                          source_ids=None):
        sources = {sid: json.loads(vids)
                   for sid, vids in self.db.execute("select * from sources")}
        rows = []
        for sid, agg, item, lvl, content, start, end, _, vec in self.db.execute(
                "select * from aggregate_units where embedder = ? and level = ?",
                (embedder_key, level)):
            if source_ids and sid not in source_ids:
                continue
            rows.append({"source_id": sid, "video_ids": sources.get(sid, [sid]),
                         "aggregate_id": agg, "item_id": item, "level": lvl,
                         "content": content, "start_ts": start, "end_ts": end,
                         "score": cosine(vector, json.loads(vec))})
        rows.sort(key=lambda r: r["score"], reverse=True)
        return [{**r, "dense_rank": rank, "text_rank": None}
                for rank, r in enumerate(rows[:limit], start=1)]

    def close(self):
        self.db.close()


models = Models(vlm=OpenAI("gpt-5.4-mini"), llm=OpenAI("gpt-5.4-mini"),
                embedder=LocalEmbedder())

with SQLite("data/vidra.sqlite") as database:
    run = video_rag(sys.argv[1], "data/out", sampler="clip", models=models,
                    database=database)
    moments, notes = search("a person at a counter", run.video_id,
                            models=models, database=database)
    for m in moments:
        print(f"chunk {m.chunk_id}: {m.start_ts:.1f}-{m.end_ts:.1f}s")

    d = run.folder
    video = aggregates.record(timeline=d / "timeline.json",
                              descriptions=d / "descriptions.json")
    told = video.excerpt(answers={"clip": ["summary"]})
    aggregates.aggregate(out=d / "aggregates", models=models, database=database,
                         summary=told, chapters=told)
    for hit in aggregates.search("the opening", level="span", models=models,
                                 database=database):
        print(f"{hit['aggregate_id']} {hit['item_id']}: {hit['start_ts']}-{hit['end_ts']}s")
