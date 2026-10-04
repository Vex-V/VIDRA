-- Drop everything the pipeline owns, then run video_rag.sql and aggregates.sql.
--
-- Destroys every row, including descriptions and embeddings that cost money to
-- produce. Run by hand only. The later statements remove leftovers of older
-- versions (a `ver3` schema, tables in `public`); harmless when absent.

drop schema if exists falconvar cascade;

drop schema if exists ver3 cascade;

drop table if exists
  public.video_manifests, public.video_chunks, public.video_descriptions,
  public.chunk_embeddings, public.audio_transcripts, public.audio_chunks,
  public.video_aggregates, public.video_embeddings
cascade;

drop function if exists public.search_embeddings(
  text, vector, text, text, text, int, int);
