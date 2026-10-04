-- FalCONvar · video_rag on Supabase: the tables `video_rag` exports and the
-- search over them. Every table here starts with vr_; the aggregates' (ag_)
-- are in aggregates.sql.
--
-- Run it in the SQL editor; it is idempotent. Then add `falconvar` under
-- Dashboard > Settings > API > Exposed schemas.
--
-- Writes use the secret key; reads use the publishable key (`anon`), which is
-- read-only. Tables that are cheap to rebuild cascade from their parent;
-- `vr_descriptions` and `vr_embeddings` have no foreign key and are never
-- deleted by a cascade.

-- ===========================================================================
-- 0 · schema and extension
-- ===========================================================================
create schema if not exists falconvar;
grant usage on schema falconvar to anon, service_role;
create extension if not exists vector;

-- ===========================================================================
-- 1 · the file
-- ===========================================================================
create table if not exists falconvar.vr_videos (
  video_id      text primary key,
  path          text not null,
  name          text,                     -- the filename unless given
  recorded_at   timestamptz,              -- the container's creation time, or given
  container     text not null,
  duration_s    numeric,
  has_video     boolean not null,
  has_audio     boolean not null,
  video_stream  jsonb,
  audio_stream  jsonb,
  seen_at       timestamptz not null default now()
);

-- ===========================================================================
-- 2 · the grid: one per video; chunk tables join it on (video_id, chunk_id)
-- ===========================================================================
create table if not exists falconvar.vr_timelines (
  video_id      text primary key references falconvar.vr_videos on delete cascade,
  policy        text not null,            -- uniform | scene | vad | speaker
  derived_from  text not null,            -- video | audio | grid
  params        jsonb not null default '{}'::jsonb,
  fingerprint   text not null,
  duration_s    numeric not null,
  chunk_count   int not null,
  built_at      timestamptz not null default now()
);

create table if not exists falconvar.vr_chunks (
  video_id  text not null references falconvar.vr_timelines on delete cascade,
  chunk_id  int  not null,
  start_ts  numeric not null,
  end_ts    numeric not null,
  primary key (video_id, chunk_id),
  check (end_ts > start_ts)
);

-- ===========================================================================
-- 3 · the soundtrack: the raw transcript, and its text per chunk
-- ===========================================================================
create table if not exists falconvar.vr_transcripts (
  video_id     text primary key references falconvar.vr_videos on delete cascade,
  timeline_fingerprint text,
  model        jsonb not null default '{}'::jsonb,
  track        jsonb not null default '{}'::jsonb,
  stats        jsonb not null default '{}'::jsonb,
  segments     jsonb not null default '[]'::jsonb,
  words        jsonb not null default '[]'::jsonb,
  turns        jsonb not null default '[]'::jsonb,
  heard_at     timestamptz not null default now()
);

create table if not exists falconvar.vr_transcript_chunks (
  video_id    text not null,
  chunk_id    int  not null,
  text        text not null default '',
  word_count  int  not null default 0,
  structured  jsonb not null default '{}'::jsonb,   -- {speakers}
  turns       jsonb not null default '[]'::jsonb,
  primary key (video_id, chunk_id),
  foreign key (video_id, chunk_id) references falconvar.vr_chunks on delete cascade
);

-- ===========================================================================
-- 4 · the picture: which frames each sampler kept
-- ===========================================================================
create table if not exists falconvar.vr_manifests (
  video_id     text primary key references falconvar.vr_videos on delete cascade,
  timeline_fingerprint text not null,
  manifest_fingerprint text not null,
  source       jsonb not null,
  config       jsonb not null,
  stats        jsonb not null default '{}'::jsonb,
  ingested_at  timestamptz not null default now()
);

-- One row per (chunk, sampler run); `questions` is what was asked of its frames.
create table if not exists falconvar.vr_chunk_samplers (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- the sampler's name
  questions   text[] not null default '{}',
  frame_count int  not null,
  frames      jsonb not null,
  primary key (video_id, chunk_id, sampler_id),
  foreign key (video_id, chunk_id) references falconvar.vr_chunks on delete cascade
);

-- ===========================================================================
-- 5 · descriptions: one model answer per (chunk, sampler:question). No foreign key.
-- ===========================================================================
create table if not exists falconvar.vr_descriptions (
  video_id      text not null,
  chunk_id      int  not null,
  sampler_id    text not null,
  question      text not null,
  frame_indexes int[] not null,
  frame_count   int  not null,
  description   text,
  structured    jsonb not null default '{}'::jsonb,
  model         jsonb not null default '{}'::jsonb,
  elapsed_s     numeric,
  timeline_fingerprint text,
  manifest_fingerprint text,
  described_at  timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id)
);

-- ===========================================================================
-- 6 · embeddings: the moment index. No foreign key.
--
-- One vector space per embedder: searches filter on `embedder` before taking a
-- distance, so the column holds any width. No HNSW index (it needs a fixed
-- width); a large single space could add a partial one:
-- `using hnsw ((embedding::vector(N)) vector_cosine_ops) where embedder = '...'`.
-- ===========================================================================
create table if not exists falconvar.vr_embeddings (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- the pairing: "clip:text"
  sampler     text not null,              -- the sampler half
  question    text not null,              -- the question half
  embedder    text not null,              -- provider:model:dims
  text_hash   text not null,              -- a hash of content
  content     text not null,
  structured  jsonb not null default '{}'::jsonb,
  embedding   vector not null,            -- any width
  -- Full text over the content and every string in the structured answer.
  fts         tsvector generated always as (
                to_tsvector('english', content || ' ' || jsonb_path_query_array(
                  structured, 'strict $.**?(@.type() == "string")')::text)
              ) stored,
  embedded_at timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id, embedder)
);

create index if not exists vr_embeddings_question
  on falconvar.vr_embeddings (video_id, embedder, question);
create index if not exists vr_embeddings_fts on falconvar.vr_embeddings using gin (fts);
create index if not exists vr_embeddings_structured
  on falconvar.vr_embeddings using gin (structured jsonb_path_ops);

-- ===========================================================================
-- 7 · prompts: what each question said, at each version a run asked it under.
--     Written, never read back; rows are never deleted.
-- ===========================================================================
create table if not exists falconvar.vr_prompts (
  name        text not null,
  version     text not null,          -- instruction + shape + system
  instruction text not null,
  shape       jsonb not null default '{}'::jsonb,
  summary     text not null default 'standard',
  builtin     boolean not null default false,
  about       text,
  first_seen  timestamptz not null default now(),
  primary key (name, version)
);

-- ===========================================================================
-- 8 · the hybrid search: a vector ranking and a full-text ranking of the same
--     rows, fused by RRF. Changing the parameter list needs the old function
--     dropped first (`create or replace` would add an overload).
-- ===========================================================================
create or replace function falconvar.vr_search(
  p_embedder     text,
  p_query_vector vector,
  p_query_text   text default null,
  p_video_ids    text[] default null, -- null = every video
  p_sampler      text default null,   -- one pairing: sampler_id = 'clip:text'
  p_question     text default null,   -- the question, whoever asked it
  p_strategy     text default null,   -- one sampler's whole output: 'clip'
  p_chunk_ids    int[] default null,  -- a set of chunks; a time window resolves to this
  p_structured   jsonb default null,  -- exact values, e.g. {"severity":"severe"}
  p_limit        int  default 20,
  p_rrf_k        int  default 60
)
returns table (
  video_id text, chunk_id int, sampler_id text, sampler text, question text,
  content text, structured jsonb, start_ts numeric, end_ts numeric,
  vector_rank int, text_rank int, score double precision
)
language sql stable as $$
  with candidates as (
    select e.* from falconvar.vr_embeddings e
    where e.embedder = p_embedder
      and (p_video_ids  is null or e.video_id = any(p_video_ids))
      and (p_sampler    is null or e.sampler_id = p_sampler)
      and (p_question   is null or e.question = p_question)
      and (p_strategy   is null or e.sampler = p_strategy)
      and (p_chunk_ids  is null or e.chunk_id = any(p_chunk_ids))
      and (p_structured is null or e.structured @> p_structured)   -- GIN
  ),
  by_vector as (
    select c.video_id, c.chunk_id, c.sampler_id,
           row_number() over (order by c.embedding <=> p_query_vector) as rank
    from candidates c
    order by c.embedding <=> p_query_vector
    limit greatest(p_limit * 4, 40)
  ),
  -- Any query term may match (`&` -> `|`), keeping stemming and stopwords.
  query_or as (
    select nullif(replace(
             websearch_to_tsquery('english', coalesce(p_query_text, ''))::text,
             '&', '|'), '')::tsquery as q
  ),
  query_terms as (
    select array_agg(lexeme) as lexemes
    from unnest(to_tsvector('english', coalesce(p_query_text, '')))
  ),
  -- With two or more query lexemes a row must share at least two.
  by_text as (
    select c.video_id, c.chunk_id, c.sampler_id,
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
    select coalesce(v.video_id,   t.video_id)   as video_id,
           coalesce(v.chunk_id,   t.chunk_id)   as chunk_id,
           coalesce(v.sampler_id, t.sampler_id) as sampler_id,
           v.rank as vector_rank, t.rank as text_rank,
           coalesce(1.0 / (p_rrf_k + v.rank), 0)
         + coalesce(1.0 / (p_rrf_k + t.rank), 0) as score
    from by_vector v
    full outer join by_text t
      on  v.video_id   = t.video_id
      and v.chunk_id   = t.chunk_id
      and v.sampler_id = t.sampler_id
  )
  select f.video_id, f.chunk_id, f.sampler_id, c.sampler, c.question,
         c.content, c.structured,
         k.start_ts, k.end_ts,                -- from the grid
         f.vector_rank::int, f.text_rank::int, f.score
  from fused f
  join candidates c
    on  c.video_id   = f.video_id
    and c.chunk_id   = f.chunk_id
    and c.sampler_id = f.sampler_id
  left join falconvar.vr_chunks k
    on k.video_id = f.video_id and k.chunk_id = f.chunk_id
  -- Ties broken by the vector rank.
  order by f.score desc,
           f.vector_rank asc nulls last,
           f.video_id, f.chunk_id, f.sampler_id
  limit p_limit;
$$;

-- ===========================================================================
-- 9 · row level security and grants: read-only for `anon`. Last, so reads are
--     never denied between enabling RLS and adding its policy.
-- ===========================================================================
do $$
declare t text;
begin
  foreach t in array array[
    'vr_videos','vr_timelines','vr_chunks','vr_transcripts','vr_transcript_chunks',
    'vr_manifests','vr_chunk_samplers','vr_descriptions','vr_embeddings','vr_prompts'
  ] loop
    execute format('alter table falconvar.%I enable row level security', t);
    execute format('drop policy if exists "public read" on falconvar.%I', t);
    execute format(
      'create policy "public read" on falconvar.%I for select to anon using (true)', t);
  end loop;
end $$;

grant select  on all tables in schema falconvar to anon;
grant all     on all tables in schema falconvar to service_role;
grant execute on function falconvar.vr_search(
  text, vector, text, text[], text, text, text, int[], jsonb, int, int)
  to anon, service_role;

-- ===========================================================================
-- Verify reads under the publishable key (the SQL editor is a superuser):
--   select count(*) from falconvar.vr_videos;
-- ===========================================================================
