-- FalCONvar · the aggregates on Supabase. Run with video_rag.sql, in either
-- order; idempotent.
--
-- An answer belongs to a source: one video, or several laid end to end
-- (`aggregates.combination`). `video_ids` lists them and is not a foreign key
-- to `vr_videos`, so deleting a video never deletes an answer.
--
-- The tables are generic over every aggregator:
--
--   ag_sources       one per source                 the equivalent of `vr_videos`
--   ag_answers       one per answer, payload whole
--   ag_items         one per thing an answer places in time: a chapter, an
--                    event, a name, a linked entity
--   ag_mentions      one per sighting of a linked entity
--   ag_embeddings    summaries, chapters, entities -- one level at a time
--   ag_definitions   what a prompt or link profile said, per version
--
-- Every table here starts with ag_; every video_rag table with vr_.
--
-- Rows are built by `falconvar.aggregates.export`; `aggregate(..., database=)`
-- writes them.

-- ===========================================================================
-- 0 · schema and extension
-- ===========================================================================
create schema if not exists falconvar;
grant usage on schema falconvar to anon, service_role;
create extension if not exists vector;

-- ===========================================================================
-- 1 · sources and answers
-- ===========================================================================
create table if not exists falconvar.ag_sources (
  source_id    text primary key,          -- a video id, or combined-<hash>
  video_ids    text[] not null,           -- what it draws from, in order
  members      jsonb not null default '[]'::jsonb,  -- each: video_id, first_chunk, chunk_count, offset_s
  duration_s   numeric,
  chunk_count  int,
  created_at   timestamptz not null default now()
);
-- "every source involving test": video_ids @> '{test}'
create index if not exists ag_sources_videos
  on falconvar.ag_sources using gin (video_ids);

create table if not exists falconvar.ag_answers (
  source_id    text not null references falconvar.ag_sources on delete cascade,
  aggregate_id text not null,             -- summary · summary~severity · entities:people
  aggregator   text not null,             -- summary · entities:people (the definition)
  tier         text not null,             -- free | local | llm
  payload      jsonb not null,            -- the answer, as its file holds it
  inputs       text,                      -- the selection read; null for a counter
  model        text,                      -- who made it
  version      text,                      -- the definition's hash when it was asked
  inputs_fingerprint text not null,
  built_at     timestamptz not null default now(),
  primary key (source_id, aggregate_id)
);

-- ===========================================================================
-- 2 · what an answer places in time, unpacked from its payload; deleted with it
-- ===========================================================================
create table if not exists falconvar.ag_items (
  source_id    text not null,
  aggregate_id text not null,
  item_id      text not null,             -- chapter-2 · event-14 · name-17 · e003
  item_kind    text not null,             -- chapter · event · name · entity
  label        text,                      -- a chapter's title, a name's label, an entity's identity
  text         text,                      -- what it says, as it is embedded
  chunk_ids    int[] not null default '{}',
  start_ts     numeric,
  end_ts       numeric,
  data         jsonb not null default '{}'::jsonb,  -- the rest of the item
  primary key (source_id, aggregate_id, item_id),
  foreign key (source_id, aggregate_id) references falconvar.ag_answers on delete cascade
);
create index if not exists ag_items_kind
  on falconvar.ag_items (source_id, item_kind);
-- "what is in chunk 6": chunk_ids @> '{6}'
create index if not exists ag_items_chunks
  on falconvar.ag_items using gin (chunk_ids);

create table if not exists falconvar.ag_mentions (
  source_id    text not null,
  aggregate_id text not null,
  mention_key  text not null,             -- c3/yolo/people/1
  item_id      text not null,             -- the entity it was linked into
  chunk_id     int not null,
  sampler_id   text not null,
  entry        jsonb not null default '{}'::jsonb,
  doubt        text,                      -- why the account left it out; null if undisputed
  primary key (source_id, aggregate_id, mention_key),
  foreign key (source_id, aggregate_id, item_id)
    references falconvar.ag_items on delete cascade
);
create index if not exists ag_mentions_chunk
  on falconvar.ag_mentions (source_id, chunk_id);

-- ===========================================================================
-- 3 · the aggregate index: one table, three levels, any embedder and width.
--     A search filters on embedder and level before taking a distance.
-- ===========================================================================
create table if not exists falconvar.ag_embeddings (
  source_id    text not null,
  aggregate_id text not null,
  item_id      text not null default '',  -- '' for a summary, which is the whole answer
  level        text not null check (level in ('source', 'span', 'entity')),
  embedder     text not null,             -- provider:model:dims
  text_hash    text not null,
  content      text not null,
  start_ts     numeric,
  end_ts       numeric,
  embedding    vector not null,
  fts          tsvector generated always as (to_tsvector('english', content)) stored,
  embedded_at  timestamptz not null default now(),
  primary key (source_id, aggregate_id, item_id, embedder),
  foreign key (source_id, aggregate_id) references falconvar.ag_answers on delete cascade
);
create index if not exists ag_embeddings_level
  on falconvar.ag_embeddings (embedder, level);
create index if not exists ag_embeddings_fts
  on falconvar.ag_embeddings using gin (fts);

-- ===========================================================================
-- 4 · definitions: what a prompt or link profile said at each version an
--     answer records in `ag_answers.version`. Written, never read back.
-- ===========================================================================
create table if not exists falconvar.ag_definitions (
  name        text not null,          -- summary · entities:people
  version     text not null,
  kind        text not null,          -- fold | spans | items | link
  definition  jsonb not null,
  builtin     boolean not null default false,
  first_seen  timestamptz not null default now(),
  primary key (name, version)
);

-- ===========================================================================
-- 5 · the search: video_rag.sql's hybrid search, over one level
-- ===========================================================================
create or replace function falconvar.ag_search(
  p_embedder     text,
  p_level        text,                 -- source | span | entity
  p_query_vector vector,
  p_query_text   text default null,
  p_source_ids   text[] default null,  -- null = every source
  p_limit        int  default 5,
  p_rrf_k        int  default 60
)
returns table (
  source_id text, aggregate_id text, item_id text, level text, content text,
  label text, video_ids text[], start_ts numeric, end_ts numeric,
  vector_rank int, text_rank int, score double precision
)
language sql stable as $$
  with candidates as (
    select e.* from falconvar.ag_embeddings e
    where e.embedder = p_embedder
      and e.level = p_level
      and (p_source_ids is null or e.source_id = any(p_source_ids))
  ),
  by_vector as (
    select c.source_id, c.aggregate_id, c.item_id,
           row_number() over (order by c.embedding <=> p_query_vector) as rank
    from candidates c
    order by c.embedding <=> p_query_vector
    limit greatest(p_limit * 4, 40)
  ),
  query_or as (
    select nullif(replace(
             websearch_to_tsquery('english', coalesce(p_query_text, ''))::text,
             '&', '|'), '')::tsquery as q
  ),
  query_terms as (
    select array_agg(lexeme) as lexemes
    from unnest(to_tsvector('english', coalesce(p_query_text, '')))
  ),
  by_text as (
    select c.source_id, c.aggregate_id, c.item_id,
           row_number() over (order by ts_rank_cd(c.fts, o.q) desc) as rank
    from candidates c
    cross join query_or o
    cross join query_terms t
    where o.q is not null
      and c.fts @@ o.q
      and (select count(*) from unnest(c.fts) d
            where d.lexeme = any(t.lexemes))
          >= least(2, coalesce(cardinality(t.lexemes), 1))
    limit greatest(p_limit * 4, 40)
  ),
  fused as (
    select coalesce(v.source_id,    t.source_id)    as source_id,
           coalesce(v.aggregate_id, t.aggregate_id) as aggregate_id,
           coalesce(v.item_id,      t.item_id)      as item_id,
           v.rank as vector_rank, t.rank as text_rank,
           coalesce(1.0 / (p_rrf_k + v.rank), 0)
         + coalesce(1.0 / (p_rrf_k + t.rank), 0) as score
    from by_vector v
    full outer join by_text t
      on  v.source_id = t.source_id and v.aggregate_id = t.aggregate_id
      and v.item_id   = t.item_id
  )
  select f.source_id, f.aggregate_id, f.item_id, c.level, c.content,
         i.label, s.video_ids, c.start_ts, c.end_ts,
         f.vector_rank::int, f.text_rank::int, f.score
  from fused f
  join candidates c
    on  c.source_id = f.source_id and c.aggregate_id = f.aggregate_id
    and c.item_id   = f.item_id
  join falconvar.ag_sources s on s.source_id = f.source_id
  left join falconvar.ag_items i
    on  i.source_id = f.source_id and i.aggregate_id = f.aggregate_id
    and i.item_id   = f.item_id
  order by f.score desc, f.vector_rank asc nulls last,
           f.source_id, f.aggregate_id, f.item_id
  limit p_limit;
$$;

-- ===========================================================================
-- 6 · row level security and grants: read-only for `anon`. Last, as in
--     video_rag.sql.
-- ===========================================================================
do $$
declare t text;
begin
  foreach t in array array[
    'ag_sources','ag_answers','ag_items','ag_mentions',
    'ag_embeddings','ag_definitions'
  ] loop
    execute format('alter table falconvar.%I enable row level security', t);
    execute format('drop policy if exists "public read" on falconvar.%I', t);
    execute format(
      'create policy "public read" on falconvar.%I for select to anon using (true)', t);
  end loop;
end $$;

grant select  on all tables in schema falconvar to anon;
grant all     on all tables in schema falconvar to service_role;
grant execute on function falconvar.ag_search(
  text, text, vector, text, text[], int, int)
  to anon, service_role;

-- ===========================================================================
-- Verify under the PUBLISHABLE key:  select count(*) from falconvar.ag_answers;
-- ===========================================================================
