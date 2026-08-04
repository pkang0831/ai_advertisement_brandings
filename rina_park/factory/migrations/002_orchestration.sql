CREATE TABLE orchestrator_leases (
    lease_name TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE runtime_settings (
    setting_key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX posts_due
ON posts(platform, state, publish_at_utc);

CREATE TRIGGER posts_schedule_invalidation
AFTER UPDATE OF publish_at_local, publish_at_utc, timezone, calendar_version ON posts
WHEN OLD.state IN ('schedule_approved','packaged','publishing')
 AND (
   OLD.publish_at_local <> NEW.publish_at_local
   OR OLD.publish_at_utc <> NEW.publish_at_utc
   OR OLD.timezone <> NEW.timezone
   OR OLD.calendar_version <> NEW.calendar_version
 )
BEGIN
  INSERT INTO approvals(
    post_id, approval_type, decision, approver, reason,
    snapshot_hash, created_at
  ) VALUES (
    NEW.post_id, 'schedule', 'invalidated', 'system',
    'approved schedule changed', NEW.content_hash,
    strftime('%Y-%m-%dT%H:%M:%fZ','now')
  );
  UPDATE posts SET state = 'needs_review' WHERE post_id = NEW.post_id;
  INSERT INTO audit_events(entity_type, entity_id, event_type, actor, payload_json, created_at)
  VALUES(
    'post', NEW.post_id, 'schedule_invalidated', 'system',
    json_object(
      'old_publish_at_utc', OLD.publish_at_utc,
      'new_publish_at_utc', NEW.publish_at_utc,
      'calendar_version', NEW.calendar_version
    ),
    strftime('%Y-%m-%dT%H:%M:%fZ','now')
  );
END;
