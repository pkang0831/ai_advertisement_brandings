CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE posts (
    post_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL CHECK (platform IN ('instagram', 'patreon')),
    audience_tiers TEXT NOT NULL DEFAULT '[]',
    format TEXT NOT NULL,
    story_week INTEGER,
    publish_at_local TEXT,
    publish_at_utc TEXT,
    timezone TEXT NOT NULL DEFAULT 'America/Toronto',
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    cta TEXT NOT NULL DEFAULT '',
    alt_text TEXT NOT NULL DEFAULT '',
    hashtags TEXT NOT NULL DEFAULT '[]',
    location_label TEXT NOT NULL DEFAULT '',
    disclosure TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    calendar_version INTEGER NOT NULL DEFAULT 1,
    state TEXT NOT NULL DEFAULT 'draft' CHECK (state IN (
        'draft','generating','assets_ready','content_approved',
        'schedule_approved','packaged','publishing','published',
        'needs_review','failed','needs_reconciliation'
    )),
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (instr(lower(post_id), 'mature') = 0),
    CHECK (substr(lower(post_id), 1, 4) <> 'mne_'),
    CHECK (instr(lower(title || body || cta), 'lingerie') = 0)
);

CREATE TABLE post_state_transitions (
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    PRIMARY KEY (from_state, to_state)
);

INSERT INTO post_state_transitions VALUES
('draft','generating'), ('generating','assets_ready'),
('assets_ready','content_approved'), ('content_approved','schedule_approved'),
('schedule_approved','packaged'), ('packaged','publishing'),
('publishing','published'),
('needs_review','generating'), ('needs_review','assets_ready'),
('failed','generating'), ('needs_reconciliation','publishing'),
('needs_reconciliation','published');

INSERT INTO post_state_transitions
SELECT s, 'needs_review' FROM (
    SELECT 'draft' s UNION SELECT 'generating' UNION SELECT 'assets_ready'
    UNION SELECT 'content_approved' UNION SELECT 'schedule_approved'
    UNION SELECT 'packaged' UNION SELECT 'publishing'
);
INSERT INTO post_state_transitions
SELECT s, 'failed' FROM (
    SELECT 'draft' s UNION SELECT 'generating' UNION SELECT 'assets_ready'
    UNION SELECT 'content_approved' UNION SELECT 'schedule_approved'
    UNION SELECT 'packaged' UNION SELECT 'publishing'
);
INSERT INTO post_state_transitions VALUES
('publishing','needs_reconciliation');

CREATE TRIGGER posts_state_machine
BEFORE UPDATE OF state ON posts
WHEN OLD.state <> NEW.state
 AND NOT EXISTS (
   SELECT 1 FROM post_state_transitions
   WHERE from_state = OLD.state AND to_state = NEW.state
 )
BEGIN
  SELECT RAISE(ABORT, 'invalid post state transition');
END;

CREATE TABLE assets (
    asset_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE CHECK (length(sha256) = 64),
    relative_path TEXT NOT NULL UNIQUE,
    media_type TEXT NOT NULL CHECK (media_type IN ('image','video')),
    width INTEGER,
    height INTEGER,
    duration_ms INTEGER,
    prompt_hash TEXT NOT NULL,
    workflow_hash TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    storage_profile TEXT NOT NULL DEFAULT 'platform'
        CHECK (storage_profile = 'platform'),
    created_at TEXT NOT NULL,
    CHECK (relative_path NOT LIKE '/%'),
    CHECK (relative_path NOT LIKE '%..%'),
    CHECK (instr(lower(relative_path), 'mature_non_explicit') = 0),
    CHECK (instr(lower(asset_id), 'mature') = 0),
    CHECK (substr(lower(asset_id), 1, 4) <> 'mne_')
);

CREATE TRIGGER assets_provenance_immutable
BEFORE UPDATE OF sha256, relative_path, prompt_hash, workflow_hash, model_hash,
 policy_version, storage_profile ON assets
BEGIN
  SELECT RAISE(ABORT, 'asset provenance is immutable; register a new asset');
END;

CREATE TABLE post_assets (
    post_id TEXT NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE RESTRICT,
    asset_slot TEXT NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 0,
    selected INTEGER NOT NULL DEFAULT 0 CHECK (selected IN (0,1)),
    PRIMARY KEY (post_id, asset_id),
    UNIQUE (post_id, asset_slot, ordinal)
);

CREATE TABLE generation_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE,
    asset_slot TEXT NOT NULL,
    generation_version INTEGER NOT NULL CHECK (generation_version > 0),
    candidate_index INTEGER NOT NULL CHECK (candidate_index >= 0),
    track TEXT NOT NULL CHECK (track IN ('ig','patreon_a','patreon_b','patreon_c')),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued','leased','succeeded','failed','cancelled')),
    lease_owner TEXT,
    lease_expires_at TEXT,
    prompt_hash TEXT NOT NULL,
    workflow_hash TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (post_id, asset_slot, generation_version, candidate_index),
    CHECK (
      (status = 'leased' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
      OR status <> 'leased'
    )
);

CREATE INDEX generation_jobs_claim
ON generation_jobs(status, lease_expires_at, job_id);

CREATE TABLE qc_results (
    qc_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    check_name TEXT NOT NULL,
    passed INTEGER NOT NULL CHECK (passed IN (0,1)),
    score REAL,
    details_json TEXT NOT NULL DEFAULT '{}',
    policy_version TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    UNIQUE(asset_id, check_name, policy_version)
);

CREATE TABLE approvals (
    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL REFERENCES posts(post_id) ON DELETE RESTRICT,
    approval_type TEXT NOT NULL CHECK (approval_type IN ('content','schedule')),
    decision TEXT NOT NULL CHECK (decision IN ('approved','rejected','invalidated')),
    approver TEXT NOT NULL,
    reason TEXT,
    snapshot_hash TEXT NOT NULL,
    invalidates_approval_id INTEGER REFERENCES approvals(approval_id),
    created_at TEXT NOT NULL
);

CREATE TRIGGER approvals_immutable_update
BEFORE UPDATE ON approvals BEGIN
  SELECT RAISE(ABORT, 'approvals are immutable');
END;
CREATE TRIGGER approvals_immutable_delete
BEFORE DELETE ON approvals BEGIN
  SELECT RAISE(ABORT, 'approvals are immutable');
END;

CREATE TABLE publication_records (
    publication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL CHECK (platform IN ('instagram','patreon')),
    post_id TEXT NOT NULL REFERENCES posts(post_id) ON DELETE RESTRICT,
    remote_media_id TEXT,
    remote_url TEXT,
    published_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(platform, post_id),
    UNIQUE(remote_media_id)
);

CREATE TABLE publish_attempts (
    publish_attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL CHECK (platform IN ('instagram','patreon')),
    post_id TEXT NOT NULL REFERENCES posts(post_id) ON DELETE RESTRICT,
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    container_id TEXT,
    request_hash TEXT NOT NULL,
    response_json TEXT,
    status TEXT NOT NULL CHECK (status IN (
      'started','container_created','succeeded','retryable_error',
      'permanent_error','needs_reconciliation','expired'
    )),
    lease_owner TEXT,
    lease_expires_at TEXT,
    error_class TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(platform, post_id, attempt_no)
);

CREATE TABLE audit_events (
    audit_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TRIGGER audit_events_immutable_update
BEFORE UPDATE ON audit_events BEGIN
  SELECT RAISE(ABORT, 'audit events are immutable');
END;
CREATE TRIGGER audit_events_immutable_delete
BEFORE DELETE ON audit_events BEGIN
  SELECT RAISE(ABORT, 'audit events are immutable');
END;

CREATE TABLE throughput_benchmarks (
    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,
    model_id TEXT,
    workload TEXT NOT NULL,
    samples INTEGER NOT NULL CHECK (samples > 0),
    p50_seconds REAL NOT NULL,
    p95_seconds REAL NOT NULL,
    failure_rate REAL NOT NULL CHECK (failure_rate BETWEEN 0 AND 1),
    peak_memory_mb REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    recorded_at TEXT NOT NULL
);

CREATE TRIGGER posts_hash_invalidation
AFTER UPDATE OF title, body, cta, alt_text, hashtags, location_label,
 disclosure, content_hash ON posts
WHEN OLD.state IN ('content_approved','schedule_approved','packaged','publishing')
 AND OLD.content_hash <> NEW.content_hash
BEGIN
  INSERT INTO approvals(
    post_id, approval_type, decision, approver, reason,
    snapshot_hash, created_at
  ) VALUES (
    NEW.post_id, 'content', 'invalidated', 'system',
    'approved post content hash changed', NEW.content_hash,
    strftime('%Y-%m-%dT%H:%M:%fZ','now')
  );
  UPDATE posts SET state = 'needs_review' WHERE post_id = NEW.post_id;
  INSERT INTO audit_events(entity_type, entity_id, event_type, actor, payload_json, created_at)
  VALUES('post', NEW.post_id, 'approval_invalidated', 'system',
    json_object('old_hash', OLD.content_hash, 'new_hash', NEW.content_hash),
    strftime('%Y-%m-%dT%H:%M:%fZ','now'));
END;

CREATE TRIGGER post_assets_insert_invalidation
AFTER INSERT ON post_assets
WHEN (SELECT state FROM posts WHERE post_id = NEW.post_id)
 IN ('content_approved','schedule_approved','packaged','publishing')
BEGIN
  UPDATE posts SET state = 'needs_review' WHERE post_id = NEW.post_id;
  INSERT INTO approvals(post_id, approval_type, decision, approver, reason,
    snapshot_hash, created_at)
  SELECT NEW.post_id, 'content', 'invalidated', 'system',
    'approved asset set changed', content_hash,
    strftime('%Y-%m-%dT%H:%M:%fZ','now')
  FROM posts WHERE post_id = NEW.post_id;
END;

CREATE TRIGGER post_assets_delete_invalidation
AFTER DELETE ON post_assets
WHEN (SELECT state FROM posts WHERE post_id = OLD.post_id)
 IN ('content_approved','schedule_approved','packaged','publishing')
BEGIN
  UPDATE posts SET state = 'needs_review' WHERE post_id = OLD.post_id;
  INSERT INTO approvals(post_id, approval_type, decision, approver, reason,
    snapshot_hash, created_at)
  SELECT OLD.post_id, 'content', 'invalidated', 'system',
    'approved asset set changed', content_hash,
    strftime('%Y-%m-%dT%H:%M:%fZ','now')
  FROM posts WHERE post_id = OLD.post_id;
END;
