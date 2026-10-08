"""Models of your own: subclass VLM, LLM or Embedder and fill in one method.

These three wrap the OpenAI SDK and sentence-transformers directly, to show
what a method receives and must return; any other client works the same way.

    python custom_models.py path/to/video.mp4
"""

import base64
import os
import sys

import vidra
from vidra import LLM, VLM, Embedder, Folder, Models, aggregates
from vidra.video_rag import search, video_rag

vidra.configure(env_file=".env")


def json_format(schema):
    """`schema` is {"name": ..., "schema": <JSON Schema>}, or None for prose."""
    if schema is None:
        return {}
    return {"response_format": {"type": "json_schema",
                                "json_schema": {**schema, "strict": True}}}


class MyVLM(VLM):
    """Describes frames. `generate` may be async, as here, or a plain def."""

    key = "mine:gpt-5.4-mini"       # the identity stored with every answer
    concurrency = 4                 # calls in flight at once

    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()

    def problems(self):
        # Checked before any work starts, so a missing key fails at once.
        return [] if os.environ.get("OPENAI_API_KEY") else ["OPENAI_API_KEY is not set"]

    async def generate(self, parts, schema=None, system=None, max_output_tokens=4000):
        # parts: {"type": "text", "text": ...} and
        #        {"type": "image", "data": <bytes>, "mime": "image/jpeg"}, in order.
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            else:
                data = base64.b64encode(part["data"]).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{part['mime']};base64,{data}"}})
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": content})
        response = await self.client.chat.completions.create(
            model="gpt-5.4-mini", messages=messages,
            max_completion_tokens=max_output_tokens, **json_format(schema))
        return response.choices[0].message.content    # JSON text is parsed for you


class MyLLM(LLM):
    """Writes summaries, chapters, events and entity accounts. A plain def runs
    in a thread."""

    key = "mine:gpt-5.4-mini"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def complete(self, prompt, schema=None, system=None, max_output_tokens=4000):
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model="gpt-5.4-mini", messages=messages,
            max_completion_tokens=max_output_tokens, **json_format(schema))
        return response.choices[0].message.content


class MyEmbedder(Embedder):
    """Text to vectors. The key names the vector space: include the width."""

    key = "mine:all-MiniLM-L6-v2:384"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts):
        return self.encoder.encode(list(texts), normalize_embeddings=True).tolist()

    # Override embed_query only when a query is embedded differently.


models = Models(vlm=MyVLM(), llm=MyLLM(), embedder=MyEmbedder())
database = Folder("data/out")

run = video_rag(sys.argv[1], "data/out", sampler="clip", use_audio=False,
                models=models, database=database)
moments, _ = search("people at a counter", run.video_id, models=models, database=database)
print([(m.chunk_id, round(m.start_ts, 1)) for m in moments])

d = run.folder
video = aggregates.record(timeline=d / "timeline.json", descriptions=d / "descriptions.json")
aggregates.summary(input=video.excerpt(answers={"clip": ["summary"]}),
                   out=d / "answers" / "summary.json", models=models)
print(aggregates.load(d / "answers" / "summary.json").stats["model"])   # mine:gpt-5.4-mini
