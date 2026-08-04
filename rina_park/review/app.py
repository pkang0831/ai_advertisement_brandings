from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rina_park.review.capabilities import ReviewProfile, capabilities_for
from rina_park.review.store import ReviewStore


def _settings() -> tuple[ReviewProfile, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--profile",
        choices=[item.value for item in ReviewProfile],
        default=os.environ.get("RINA_REVIEW_PROFILE", ReviewProfile.PLATFORM.value),
    )
    parser.add_argument("--db", default=os.environ.get("RINA_REVIEW_DB"))
    args, _ = parser.parse_known_args()
    profile = ReviewProfile(args.profile)
    if not args.db:
        raise RuntimeError("Set RINA_REVIEW_DB or pass --db with a dedicated absolute DB path")
    db_path = Path(args.db)
    if not db_path.is_absolute():
        raise RuntimeError("Review DB path must be absolute")
    return profile, db_path


def render() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Streamlit is optional; install it in the project virtualenv") from exc

    profile, db_path = _settings()
    capabilities = capabilities_for(profile)
    store = ReviewStore(db_path, profile)

    title = "Rina Platform Review" if profile is ReviewProfile.PLATFORM else "Rina Mature Local Review"
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.caption(f"Profile: {profile.value} · DB: {db_path}")

    posts = store.list_posts()
    if not posts:
        st.info("No review candidates are queued.")
        return
    post_id = st.selectbox("Post", [row["post_id"] for row in posts])
    post = store.get_post(post_id)
    st.write(f"Status: `{post['status']}` · Channel: `{post['platform']}`")

    candidates = store.candidates(post_id)
    if candidates:
        columns = st.columns(min(4, len(candidates)))
        for index, candidate in enumerate(candidates):
            with columns[index % len(columns)]:
                path = Path(candidate["path"])
                if path.exists():
                    st.image(str(path), caption=candidate["asset_id"], use_container_width=True)
                else:
                    st.error(f"Missing: {path}")
                st.code(candidate["sha256"], language=None)
                reason = st.text_input(
                    "Reject reason",
                    key=f"reason-{candidate['asset_id']}",
                    placeholder="Required when rejecting",
                )
                approve_col, reject_col = st.columns(2)
                if approve_col.button("Approve", key=f"approve-{candidate['asset_id']}"):
                    store.decide_candidate(post_id, candidate["asset_id"], "approved")
                    st.rerun()
                if reject_col.button("Reject", key=f"reject-{candidate['asset_id']}"):
                    try:
                        store.decide_candidate(
                            post_id, candidate["asset_id"], "rejected", reason
                        )
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

    st.subheader("Content")
    caption = st.text_area("Caption", value=post["caption"])
    location = st.text_input("Location label", value=post["location_label"])
    tiers = st.multiselect(
        "Audience tiers",
        ["A", "B", "C"],
        default=json.loads(post["audience_tiers_json"]),
    )
    if st.button("Save content edits"):
        store.update_content(post_id, caption, location, tiers)
        st.success("Saved. Existing approvals are now invalid if the hash changed.")

    regeneration_reason = st.text_input("Regeneration request")
    if st.button("Request regeneration"):
        try:
            store.request_regeneration(post_id, regeneration_reason)
            st.success("Regeneration requested.")
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("Immutable review snapshot")
    try:
        snapshot_hash = store.immutable_hash(post_id)
        st.code(snapshot_hash, language=None)
    except (FileNotFoundError, KeyError) as exc:
        st.error(f"Snapshot cannot be approved: {exc}")
        return

    reviewer = st.text_input("Reviewer", value=os.environ.get("USER", "local-reviewer"))
    content_valid = store.approval_valid(post_id, "content")
    st.write(f"Content approval valid: `{content_valid}`")
    if capabilities.content_approval and st.button("Approve content"):
        store.approve(post_id, "content", reviewer)
        st.rerun()

    if capabilities.schedule_approval:
        schedule_valid = store.approval_valid(post_id, "schedule")
        st.write(f"Schedule approval valid: `{schedule_valid}`")
        if st.button("Approve schedule", disabled=not content_valid):
            store.approve(post_id, "schedule", reviewer)
            st.rerun()
    else:
        st.info("Local mature profile: scheduling, export, package, and publishing are disabled.")


if __name__ == "__main__":
    render()
