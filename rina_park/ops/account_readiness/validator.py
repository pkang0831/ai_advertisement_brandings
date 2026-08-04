"""Offline account-readiness validation with fail-closed secret handling."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

TIMEZONE = "America/Toronto"
REQUIRED_PERMISSIONS = {
    "instagram_business_basic",
    "instagram_business_content_publish",
}
PLACEHOLDERS = {"", "FILL_ME", "<FILL>", "TODO", None}

UNSAFE_KEY_PARTS = {
    "password",
    "passcode",
    "secret",
    "api_key",
    "private_key",
    "client_secret",
    "access_token",
    "refresh_token",
    "token_value",
    "token_string",
    "cookie",
    "session_id",
    "legal_name",
    "owner_name",
    "full_name",
    "first_name",
    "last_name",
    "dob",
    "date_of_birth",
    "birth_date",
    "government_id",
    "id_number",
    "passport",
    "driver_license",
    "sin",
    "ssn",
    "tax_id",
    "bank",
    "account_number",
    "routing_number",
    "card_number",
    "cvv",
    "payment",
    "payment_details",
}

SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bIGQVJ[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bEAA[A-Za-z0-9]{16,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    re.compile(r"\b\d{3}[ -]\d{3}[ -]\d{3}\b"),
)

ROOT_KEYS = {
    "schema_version",
    "timezone",
    "launch",
    "public_ai_disclosure",
    "instagram",
    "patreon",
    "link_routing",
    "graph_api_optional",
}
LAUNCH_KEYS = {
    "policy",
    "quality_and_readiness_gates_passed",
    "gates_passed_on",
    "release_candidate_finalized",
    "resolved_launch_date",
}
DISCLOSURE_KEYS = {"status", "instagram", "patreon"}
PLATFORM_DISCLOSURE_KEYS = {"present", "wording"}
INSTAGRAM_KEYS = {
    "public_handle",
    "public_profile_url",
    "account_type",
    "professional_status_confirmed",
    "is_public",
    "two_factor_enabled",
}
PATREON_KEYS = {
    "public_handle",
    "public_page_url",
    "safe_for_all_audiences",
    "two_factor_enabled",
    "legal_identity_completed_with_real_owner_info",
    "payout_setup_complete",
    "tax_setup_complete",
    "currency",
    "tiers",
}
TIER_KEYS = {
    "code",
    "id",
    "name",
    "monthly_price_cad",
    "audience_includes_tier_ids",
}
ROUTING_KEYS = {
    "instagram_to_patreon_url",
    "patreon_to_instagram_url",
    "canonical_landing_url",
}
GRAPH_KEYS = {
    "intended",
    "meta_app_created",
    "instagram_login_product_configured",
    "account_added_as_app_tester",
    "tester_invitation_accepted",
    "permissions",
    "token_available_in_external_secret_store",
    "token_expires_on",
    "public_https_media_transport_ready",
    "publishing_limit_probe_passed",
    "mocked_reconciliation_tests_passed",
    "dedicated_test_account_rehearsal_passed",
}


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _is_placeholder(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and (
            value.strip() in PLACEHOLDERS
            or value.strip().startswith("FILL_")
        )
    )


def _is_unsafe_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return any(
        normalized == part
        or normalized.startswith(f"{part}_")
        or normalized.endswith(f"_{part}")
        for part in UNSAFE_KEY_PARTS
    )


def looks_sensitive(value: str) -> bool:
    """Return True for common credentials, private keys, cards, and SIN-like values."""
    return any(pattern.search(value) for pattern in SENSITIVE_VALUE_PATTERNS)


def redact_sensitive(value: Any) -> Any:
    """Recursively redact unsafe fields and sensitive-looking string values."""
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if _is_unsafe_key(key)
            else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str) and looks_sensitive(value):
        return "[REDACTED]"
    return value


def _scan_safety(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if _is_unsafe_key(key):
                errors.append(f"{child}: prohibited sensitive field")
            errors.extend(_scan_safety(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_scan_safety(item, f"{path}[{index}]"))
    elif isinstance(value, str) and looks_sensitive(value):
        errors.append(f"{path}: sensitive-looking value is prohibited")
    return errors


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML template without third-party dependencies."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "readiness YAML must preserve the template's JSON-compatible YAML format"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("readiness YAML root must be a mapping")
    return value


def _require_keys(
    value: Any,
    allowed: set[str],
    path: str,
    errors: list[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: must be a mapping")
        return {}
    missing = allowed - set(value)
    unknown = set(value) - allowed
    for key in sorted(missing):
        errors.append(f"{path}.{key}: required field is missing")
    for key in sorted(unknown):
        errors.append(f"{path}.{key}: field is not allowed")
    return value


def _required_text(value: Any, path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or _is_placeholder(value):
        errors.append(f"{path}: must be filled with public, non-secret text")
        return ""
    return value.strip()


def _required_true(value: Any, path: str, errors: list[str]) -> None:
    if value is not True:
        errors.append(f"{path}: must be true")


def _https_url(
    value: Any,
    path: str,
    errors: list[str],
    allowed_hosts: set[str] | None = None,
    optional: bool = False,
) -> None:
    if optional and _is_placeholder(value):
        return
    text = _required_text(value, path, errors)
    if not text:
        return
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        errors.append(f"{path}: must be a public HTTPS URL without credentials")
        return
    host = (parsed.hostname or "").lower()
    if allowed_hosts and host not in allowed_hosts:
        errors.append(f"{path}: host is not an approved public platform host")


def next_monday_after(value: date) -> date:
    """Return the Monday strictly after a local calendar date."""
    days = (7 - value.weekday()) % 7
    return value + timedelta(days=days or 7)


def _optional_iso_date(value: Any, path: str, errors: list[str]) -> date | None:
    if _is_placeholder(value):
        return None
    if not isinstance(value, str):
        errors.append(f"{path}: must be an ISO date (YYYY-MM-DD) or null")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        errors.append(f"{path}: must be an ISO date (YYYY-MM-DD)")
        return None


def _validate_launch(root: Mapping[str, Any], errors: list[str]) -> None:
    launch = _require_keys(root.get("launch"), LAUNCH_KEYS, "$.launch", errors)
    if launch.get("policy") != "next_monday_after_all_gates_pass":
        errors.append(
            "$.launch.policy: must equal next_monday_after_all_gates_pass"
        )
    gates_passed = launch.get("quality_and_readiness_gates_passed")
    finalized = launch.get("release_candidate_finalized")
    if not isinstance(gates_passed, bool):
        errors.append("$.launch.quality_and_readiness_gates_passed: must be a boolean")
    elif not gates_passed:
        errors.append(
            "$.launch.quality_and_readiness_gates_passed: all gates must pass before launch"
        )
    if not isinstance(finalized, bool):
        errors.append("$.launch.release_candidate_finalized: must be a boolean")
    elif not finalized:
        errors.append(
            "$.launch.release_candidate_finalized: release candidate must be finalized"
        )
    passed_on = _optional_iso_date(
        launch.get("gates_passed_on"), "$.launch.gates_passed_on", errors
    )
    resolved = _optional_iso_date(
        launch.get("resolved_launch_date"), "$.launch.resolved_launch_date", errors
    )
    if gates_passed is True and passed_on is None:
        errors.append("$.launch.gates_passed_on: required after all gates pass")
    if gates_passed is False and passed_on is not None:
        errors.append("$.launch.gates_passed_on: must be null until all gates pass")
    if finalized is True and gates_passed is not True:
        errors.append(
            "$.launch.release_candidate_finalized: requires all gates to have passed"
        )
    if finalized is not True and resolved is not None:
        errors.append(
            "$.launch.resolved_launch_date: must be null before release candidate finalization"
        )
    if finalized is True:
        if resolved is None:
            errors.append(
                "$.launch.resolved_launch_date: required for a finalized release candidate"
            )
        elif passed_on is not None and resolved != next_monday_after(passed_on):
            errors.append(
                "$.launch.resolved_launch_date: must be the Monday strictly after gates_passed_on"
            )


def _validate_disclosure(root: Mapping[str, Any], errors: list[str]) -> None:
    disclosure = _require_keys(
        root.get("public_ai_disclosure"),
        DISCLOSURE_KEYS,
        "$.public_ai_disclosure",
        errors,
    )
    status = disclosure.get("status")
    if status not in {"undecided", "resolved"}:
        errors.append("$.public_ai_disclosure.status: must be undecided or resolved")
    if status == "undecided":
        errors.append(
            "$.public_ai_disclosure.status: user decision is required before public launch"
        )
    for platform in ("instagram", "patreon"):
        path = f"$.public_ai_disclosure.{platform}"
        item = _require_keys(
            disclosure.get(platform), PLATFORM_DISCLOSURE_KEYS, path, errors
        )
        present = item.get("present")
        wording = item.get("wording")
        if status == "undecided":
            if present is not None or wording is not None:
                errors.append(f"{path}: must remain null while decision is undecided")
            continue
        if not isinstance(present, bool):
            errors.append(f"{path}.present: must be true or false after resolution")
        elif present:
            _required_text(wording, f"{path}.wording", errors)
        elif wording is not None:
            errors.append(f"{path}.wording: must be null when disclosure is absent")


def _validate_manual(data: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    root = _require_keys(data, ROOT_KEYS, "$", errors)
    if root.get("schema_version") != 2:
        errors.append("$.schema_version: must equal 2")
    if root.get("timezone") != TIMEZONE:
        errors.append(f"$.timezone: must equal {TIMEZONE}")
    _validate_launch(root, errors)
    _validate_disclosure(root, errors)

    instagram = _require_keys(
        root.get("instagram"), INSTAGRAM_KEYS, "$.instagram", errors
    )
    _required_text(instagram.get("public_handle"), "$.instagram.public_handle", errors)
    _https_url(
        instagram.get("public_profile_url"),
        "$.instagram.public_profile_url",
        errors,
        {"instagram.com", "www.instagram.com"},
    )
    if instagram.get("account_type") != "creator":
        errors.append("$.instagram.account_type: must equal creator")
    for key in (
        "professional_status_confirmed",
        "is_public",
        "two_factor_enabled",
    ):
        _required_true(instagram.get(key), f"$.instagram.{key}", errors)

    patreon = _require_keys(root.get("patreon"), PATREON_KEYS, "$.patreon", errors)
    _required_text(patreon.get("public_handle"), "$.patreon.public_handle", errors)
    _https_url(
        patreon.get("public_page_url"),
        "$.patreon.public_page_url",
        errors,
        {"patreon.com", "www.patreon.com"},
    )
    for key in (
        "safe_for_all_audiences",
        "two_factor_enabled",
        "legal_identity_completed_with_real_owner_info",
        "payout_setup_complete",
        "tax_setup_complete",
    ):
        _required_true(patreon.get(key), f"$.patreon.{key}", errors)
    if patreon.get("currency") != "CAD":
        errors.append("$.patreon.currency: must equal CAD")
    _validate_tiers(patreon.get("tiers"), errors)

    routing = _require_keys(
        root.get("link_routing"), ROUTING_KEYS, "$.link_routing", errors
    )
    _https_url(
        routing.get("instagram_to_patreon_url"),
        "$.link_routing.instagram_to_patreon_url",
        errors,
        {"patreon.com", "www.patreon.com"},
    )
    _https_url(
        routing.get("patreon_to_instagram_url"),
        "$.link_routing.patreon_to_instagram_url",
        errors,
        {"instagram.com", "www.instagram.com"},
    )
    _https_url(
        routing.get("canonical_landing_url"),
        "$.link_routing.canonical_landing_url",
        errors,
        optional=True,
    )
    return errors


def _validate_tiers(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 3:
        errors.append("$.patreon.tiers: must contain exactly A, B, and C")
        return
    tiers: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        tiers.append(_require_keys(item, TIER_KEYS, f"$.patreon.tiers[{index}]", errors))
    ids: list[str] = []
    for index, item in enumerate(tiers):
        raw_id = item.get("id")
        _required_text(raw_id, f"$.patreon.tiers[{index}].id", errors)
        ids.append(raw_id.strip() if isinstance(raw_id, str) else "")
    codes = [item.get("code") for item in tiers]
    if codes != ["A", "B", "C"]:
        errors.append("$.patreon.tiers: tier codes must be exactly A, B, C")
    if len(set(ids)) != len(ids):
        errors.append("$.patreon.tiers: tier IDs must be unique")
    known_ids = set(ids)
    prices: list[float] = []
    for index, item in enumerate(tiers):
        path = f"$.patreon.tiers[{index}]"
        _required_text(item.get("name"), f"{path}.name", errors)
        price = item.get("monthly_price_cad")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
            errors.append(f"{path}.monthly_price_cad: must be a positive number")
        else:
            prices.append(float(price))
        audience = item.get("audience_includes_tier_ids")
        if not isinstance(audience, list) or not audience:
            errors.append(f"{path}.audience_includes_tier_ids: must be a non-empty list")
        elif not all(isinstance(tier_id, str) for tier_id in audience):
            errors.append(f"{path}.audience_includes_tier_ids: values must be tier IDs")
        else:
            unknown = set(audience) - known_ids
            if unknown:
                errors.append(f"{path}.audience_includes_tier_ids: contains unknown tier IDs")
            if item.get("id") not in audience:
                errors.append(f"{path}.audience_includes_tier_ids: must include its own tier ID")
            if audience != ids[index:]:
                errors.append(
                    f"{path}.audience_includes_tier_ids: must equal this tier and all higher tiers"
                )
    if len(prices) == len(tiers) and prices != sorted(prices):
        errors.append("$.patreon.tiers: prices must increase in listed order")
    if prices and prices != [3.0, 8.0, 15.0]:
        errors.append("$.patreon.tiers: CAD prices must be exactly 3, 8, and 15")


def _validate_graph(data: Mapping[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    graph = _require_keys(
        data.get("graph_api_optional"), GRAPH_KEYS, "$.graph_api_optional", errors
    )
    intended = graph.get("intended")
    if not isinstance(intended, bool):
        errors.append("$.graph_api_optional.intended: must be a boolean")
        return False, errors
    if not intended:
        return False, errors
    for key in (
        "meta_app_created",
        "instagram_login_product_configured",
        "account_added_as_app_tester",
        "tester_invitation_accepted",
        "token_available_in_external_secret_store",
        "public_https_media_transport_ready",
        "publishing_limit_probe_passed",
        "mocked_reconciliation_tests_passed",
        "dedicated_test_account_rehearsal_passed",
    ):
        _required_true(graph.get(key), f"$.graph_api_optional.{key}", errors)
    permissions = graph.get("permissions")
    if (
        not isinstance(permissions, list)
        or not all(isinstance(item, str) for item in permissions)
        or set(permissions) != REQUIRED_PERMISSIONS
    ):
        errors.append(
            "$.graph_api_optional.permissions: must contain exactly the two approved permissions"
        )
    expires = graph.get("token_expires_on")
    if not _is_placeholder(expires):
        if not isinstance(expires, str):
            errors.append("$.graph_api_optional.token_expires_on: must be an ISO date or null")
        else:
            try:
                date.fromisoformat(expires)
            except ValueError:
                errors.append("$.graph_api_optional.token_expires_on: must be an ISO date")
    return True, errors


def validate(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return separate manual-launch and optional Graph API gates."""
    safety_errors = _scan_safety(data)
    manual_errors = _validate_manual(data)
    graph_intended, graph_errors = _validate_graph(data)
    return {
        "safe_to_store": not safety_errors,
        "safety_errors": safety_errors,
        "manual_launch": {
            "ready": not safety_errors and not manual_errors,
            "errors": manual_errors,
        },
        "graph_api_optional": {
            "intended": graph_intended,
            "ready": graph_intended and not safety_errors and not graph_errors,
            "errors": graph_errors,
            "blocks_manual_launch": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate account readiness offline")
    parser.add_argument("yaml_path", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate(load_yaml(args.yaml_path))
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(redact_sensitive(report), ensure_ascii=False, indent=2))
    if not report["safe_to_store"]:
        return 2
    return 0 if report["manual_launch"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
