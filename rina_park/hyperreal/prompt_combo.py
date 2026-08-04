"""Multi-axis prompt combinations: pose × expression × outfit × background × lora.

CLIP budget: assembled pose_short + scene_glue should stay tiny so LOOK_POS
survives the ~77-token truncate. Prefer short phrase fragments from YAML banks.

Axes (explicit):
  poses | expressions | outfit_types | outfit_colors | backgrounds | lora_presets

Modes:
  random_seeded — hash picks one value per axis for each (pose, seed)
  cartesian     — full product; exactly 1 gen per combo; seed = base_seed + job_index

Public banks: ops/prompt_bank/. Private NSFW: private/prompt_bank/.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import yaml

RINA = Path(__file__).resolve().parents[1]
PUBLIC_BANKS = RINA / "ops" / "prompt_bank"
PRIVATE_BANKS = RINA / "private" / "prompt_bank"

ComboMode = Literal["random_seeded", "cartesian"]

# Compact framing for known poses (CLIP-safe). Fallback: compress catalog text.
POSE_COMP_SHORT: dict[str, str] = {
    "hand_pockets_3q": "hands in pockets hiding fingers, standing three-quarter",
    "hand_cropped_out": "hands and feet out of frame, upper-body three-quarter",
    "hand_resting_lap_seated": "hands resting on lap, seated three-quarter indoor",
    "hand_holding_tote_side": "one hand on tote strap, other pocket, street pause",
    "hand_holding_cup_soft": "one hand holding thick cup, other soft, waist-up",
    "fitness_mat_soft_seated": "seated on yoga mat, hands on thighs soft",
    "foot_standing_flat_sneakers": "standing flat sneakers, full-body soft stance",
    "foot_seated_ankle_soft": "seated, sneakers soft ankle view, three-quarter",
    "foot_mid_stride_side": "mid-stride walk sneakers, full-body candid",
    # NSFW (ids only referenced when private overlay present)
    "nsfw_reclined_knee_up_partial": "reclined bed, one knee up, soft side light",
    "nsfw_standing_mirror_hip_angle": "standing mirror, slight hip angle, hands on thighs",
    "nsfw_side_lying_tucked": "side-lying sheets, thighs together, soft lamp",
    "nsfw_seated_bed_edge_soft": "seated bed edge, knees soft, hands on mattress",
    "nsfw_prone_pillow_rest": "prone pillow rest, soft sheets, side-back",
    "nsfw_kneeling_bed_soft": "kneeling on bed, hands on thighs soft",
    "nsfw_standing_doorway_soft": "standing doorway, soft hip lean",
    "nsfw_bathtub_edge_seated": "bathtub edge seated, knees together, soft steam",
    "nsfw_towel_hip_wrap_standing": "towel at hips, standing bathroom soft",
    "nsfw_back_view_over_shoulder": "back view, over-shoulder glance soft",
    "nsfw_window_light_standing": "standing window light, sheer curtains",
    "nsfw_sheet_drape_reclined": "reclined, sheet drape hips soft",
    "nsfw_armchair_reclined_soft": "armchair reclined, soft leg bend",
    "nsfw_floor_sitting_knees_soft": "floor sit, knees soft, elbows on knees",
    "nsfw_stretch_overhead_standing": "standing overhead stretch, dawn soft",
    "nsfw_leaning_counter_soft": "leaning bathroom counter, hands flat",
    "nsfw_morning_bed_stretch": "morning bed stretch, sheets at waist",
    "nsfw_robe_slip_shoulder": "robe slip one shoulder, soft hand on robe",
}


@dataclass(frozen=True)
class LoraPreset:
    name: str
    weight: float


@dataclass(frozen=True)
class PhraseBanks:
    expressions: list[str]
    outfit_types: list[str]
    outfit_colors: list[str]
    backgrounds: list[str]
    lora_presets: list[LoraPreset]
    track: str
    public_loaded: bool
    private_loaded: bool
    paths: dict[str, str]

    def counts(self) -> dict[str, int]:
        return {
            "expressions": len(self.expressions),
            "outfit_types": len(self.outfit_types),
            "outfit_colors": len(self.outfit_colors),
            "backgrounds": len(self.backgrounds),
            "lora_presets": len(self.lora_presets),
        }

    # Legacy aliases (older notebooks / meta)
    @property
    def outfits(self) -> list[str]:
        return [assemble_outfit(c, t) for c in self.outfit_colors for t in self.outfit_types]

    @property
    def concepts(self) -> list[str]:
        return self.backgrounds


@dataclass(frozen=True)
class ComboJob:
    pose_id: str
    seed: int
    pose_prompt: str
    scene_glue: str
    expression: str
    outfit_type: str
    outfit_color: str
    outfit: str
    background: str
    lora_name: str
    lora_weight: float
    pose_comp: str
    mode: str
    job_index: int = 0

    @property
    def concept(self) -> str:
        return self.background

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["concept"] = self.background  # legacy key
        return d


def _load_items(path: Path) -> list[str]:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    items = data.get("items") or []
    out: list[str] = []
    for x in items:
        if isinstance(x, dict):
            continue  # structured items handled elsewhere
        s = " ".join(str(x).replace("\n", " ").split()).strip()
        if s:
            out.append(s)
    return out


def _load_lora_presets(path: Path) -> list[LoraPreset]:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    items = data.get("items") or []
    out: list[LoraPreset] = []
    for i, x in enumerate(items):
        if isinstance(x, (int, float)):
            w = float(x)
            out.append(LoraPreset(name=f"w{w:.2f}".replace(".", ""), weight=w))
        elif isinstance(x, dict):
            w = float(x.get("weight", x.get("lora", 0.80)))
            name = str(x.get("name") or f"w{w:.2f}").strip() or f"preset_{i}"
            out.append(LoraPreset(name=name, weight=w))
        elif isinstance(x, str) and x.strip():
            try:
                w = float(x.strip())
                out.append(LoraPreset(name=f"w{w:.2f}".replace(".", ""), weight=w))
            except ValueError:
                continue
    # Cap at 3 for MPS reload cost
    return out[:3]


def _is_nsfw_track(track: str) -> bool:
    t = track.lower().strip()
    return t in {"nsfw", "nsfw_private", "private", "adult"}


def assemble_outfit(color: str, outfit_type: str) -> str:
    c = (color or "").strip()
    t = (outfit_type or "").strip()
    if c and t:
        return f"{c} {t}"
    return t or c


def load_phrase_banks(
    track: str,
    *,
    banks_dir: Path | None = None,
    private_banks_dir: Path | None = None,
) -> PhraseBanks:
    """Load multi-axis banks; merge private NSFW overlays when track is NSFW."""
    pub = Path(banks_dir) if banks_dir else PUBLIC_BANKS
    priv = Path(private_banks_dir) if private_banks_dir else PRIVATE_BANKS
    nsfw = _is_nsfw_track(track)

    def _axis_pair(stem_new: str, stem_legacy: str | None = None) -> tuple[list[str], Path]:
        """Prefer new axis file; fall back to legacy stem if present."""
        if nsfw:
            p_new = priv / f"{stem_new}_nsfw.yml"
            items = _load_items(p_new)
            if items:
                return items, p_new
            if stem_legacy:
                p_leg = priv / f"{stem_legacy}_nsfw.yml"
                items = _load_items(p_leg)
                if items:
                    return items, p_leg
            # Soft fallback to public
            p_pub = pub / f"{stem_new}_sfw.yml"
            items = _load_items(p_pub)
            if items:
                return items, p_pub
            if stem_legacy:
                p_pub = pub / f"{stem_legacy}_sfw.yml"
                return _load_items(p_pub), p_pub
            return [], p_new
        p_new = pub / f"{stem_new}_sfw.yml"
        items = _load_items(p_new)
        if items:
            return items, p_new
        if stem_legacy:
            p_leg = pub / f"{stem_legacy}_sfw.yml"
            return _load_items(p_leg), p_leg
        return [], p_new

    expressions, p_expr = _axis_pair("expressions")
    outfit_types, p_types = _axis_pair("outfit_types", "outfits")
    outfit_colors, p_colors = _axis_pair("outfit_colors")
    backgrounds, p_bg = _axis_pair("backgrounds", "concepts")

    if nsfw:
        p_lora = priv / "lora_presets_nsfw.yml"
        lora_presets = _load_lora_presets(p_lora)
        if not lora_presets:
            lora_presets = _load_lora_presets(pub / "lora_presets_sfw.yml")
            p_lora = pub / "lora_presets_sfw.yml"
        private_loaded = bool(expressions or outfit_types or backgrounds)
        public_loaded = (pub / "outfit_types_sfw.yml").is_file() or (pub / "outfits_sfw.yml").is_file()
        resolved_track = "nsfw_private"
    else:
        p_lora = pub / "lora_presets_sfw.yml"
        lora_presets = _load_lora_presets(p_lora)
        private_loaded = False
        public_loaded = bool(expressions or outfit_types or backgrounds)
        resolved_track = "sfw"

    # If colors missing but types came from legacy full outfits, use empty color (= type only)
    if not outfit_colors and outfit_types:
        outfit_colors = [""]

    if not lora_presets:
        lora_presets = [LoraPreset(name="char_080", weight=0.80)]

    paths = {
        "expressions": str(p_expr),
        "outfit_types": str(p_types),
        "outfit_colors": str(p_colors),
        "backgrounds": str(p_bg),
        "lora_presets": str(p_lora),
    }

    if not expressions or not outfit_types or not backgrounds:
        raise FileNotFoundError(
            f"phrase banks incomplete for track={resolved_track}: "
            f"expressions={len(expressions)} outfit_types={len(outfit_types)} "
            f"outfit_colors={len(outfit_colors)} backgrounds={len(backgrounds)} "
            f"lora_presets={len(lora_presets)} paths={paths}"
        )
    return PhraseBanks(
        expressions=expressions,
        outfit_types=outfit_types,
        outfit_colors=outfit_colors,
        backgrounds=backgrounds,
        lora_presets=lora_presets,
        track=resolved_track,
        public_loaded=public_loaded,
        private_loaded=private_loaded,
        paths=paths,
    )


def seeded_index(n: int, seed: int, pose_id: str, axis: str) -> int:
    if n <= 0:
        raise ValueError("empty bank")
    digest = hashlib.sha256(f"{seed}|{pose_id}|{axis}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def seeded_pick(items: Sequence[str], seed: int, pose_id: str, axis: str) -> str:
    return items[seeded_index(len(items), seed, pose_id, axis)]


def compress_phrase(text: str, max_chars: int = 72) -> str:
    flat = " ".join(text.replace("\n", " ").split())
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars]
    if "," in cut:
        cut = cut.rsplit(",", 1)[0]
    return cut.strip()


def pose_comp_short(pose_id: str, composition: str = "", positive_extra: str = "") -> str:
    if pose_id in POSE_COMP_SHORT:
        return POSE_COMP_SHORT[pose_id]
    for candidate in (positive_extra, composition):
        c = compress_phrase(candidate, max_chars=64)
        if c:
            return c
    return "three-quarter lifestyle soft"


def assemble_pose_prompt(pose_comp: str, outfit: str, expression: str) -> str:
    bits = [pose_comp.strip(), outfit.strip(), expression.strip()]
    return ", ".join(b for b in bits if b)


def count_axes(track: str, poses: Sequence[str] | None = None) -> dict[str, Any]:
    """Return per-category choice counts + total cartesian size."""
    banks = load_phrase_banks(track)
    n_poses = len(poses) if poses is not None else 0
    c = banks.counts()
    bank_product = (
        c["expressions"]
        * c["outfit_types"]
        * c["outfit_colors"]
        * c["backgrounds"]
        * c["lora_presets"]
    )
    total = n_poses * bank_product if n_poses else bank_product
    return {
        "track": banks.track,
        "pose": n_poses,
        "expression": c["expressions"],
        "outfit_type": c["outfit_types"],
        "outfit_color": c["outfit_colors"],
        "background": c["backgrounds"],
        "lora": c["lora_presets"],
        "bank_product": bank_product,
        "total_cartesian": total,
        "paths": banks.paths,
        "private_loaded": banks.private_loaded,
    }


def estimate_combo_space(banks: PhraseBanks, n_poses: int) -> int:
    c = banks.counts()
    return (
        max(0, n_poses)
        * c["expressions"]
        * c["outfit_types"]
        * c["outfit_colors"]
        * c["backgrounds"]
        * c["lora_presets"]
    )


def _make_job(
    *,
    pose_id: str,
    seed: int,
    pose_comp: str,
    expression: str,
    outfit_type: str,
    outfit_color: str,
    background: str,
    lora: LoraPreset,
    mode: str,
    job_index: int,
) -> ComboJob:
    outfit = assemble_outfit(outfit_color, outfit_type)
    pose_prompt = assemble_pose_prompt(pose_comp, outfit, expression)
    return ComboJob(
        pose_id=pose_id,
        seed=seed,
        pose_prompt=pose_prompt,
        scene_glue=background,
        expression=expression,
        outfit_type=outfit_type,
        outfit_color=outfit_color,
        outfit=outfit,
        background=background,
        lora_name=lora.name,
        lora_weight=lora.weight,
        pose_comp=pose_comp,
        mode=mode,
        job_index=job_index,
    )


def build_combo_job(
    pose_id: str,
    seed: int,
    banks: PhraseBanks,
    *,
    composition: str = "",
    positive_extra: str = "",
    mode: str = "random_seeded",
    job_index: int = 0,
    expression: str | None = None,
    outfit_type: str | None = None,
    outfit_color: str | None = None,
    background: str | None = None,
    lora: LoraPreset | None = None,
) -> ComboJob:
    """Build one combo. Prefer explicit axis values; else hash-pick (random_seeded)."""
    pose_comp = pose_comp_short(pose_id, composition, positive_extra)
    if mode == "cartesian" and all(
        v is not None for v in (expression, outfit_type, outfit_color, background, lora)
    ):
        return _make_job(
            pose_id=pose_id,
            seed=seed,
            pose_comp=pose_comp,
            expression=expression or "",
            outfit_type=outfit_type or "",
            outfit_color=outfit_color if outfit_color is not None else "",
            background=background or "",
            lora=lora or banks.lora_presets[0],
            mode=mode,
            job_index=job_index,
        )

    # random_seeded (or cartesian without explicit axes — legacy stride fallback)
    if mode == "cartesian" and expression is None:
        # Legacy mixed-radix stride over bank product (kept for old callers)
        oi, ci, ei = _cartesian_indices(
            job_index,
            len(banks.outfit_types),
            len(banks.backgrounds),
            len(banks.expressions),
        )
        col_i = job_index % max(1, len(banks.outfit_colors))
        lor_i = (job_index // max(1, len(banks.outfit_colors))) % max(1, len(banks.lora_presets))
        return _make_job(
            pose_id=pose_id,
            seed=seed,
            pose_comp=pose_comp,
            expression=banks.expressions[ei],
            outfit_type=banks.outfit_types[oi],
            outfit_color=banks.outfit_colors[col_i],
            background=banks.backgrounds[ci],
            lora=banks.lora_presets[lor_i],
            mode=mode,
            job_index=job_index,
        )

    expr = expression or seeded_pick(banks.expressions, seed, pose_id, "expression")
    otype = outfit_type or seeded_pick(banks.outfit_types, seed, pose_id, "outfit_type")
    ocol = (
        outfit_color
        if outfit_color is not None
        else seeded_pick(banks.outfit_colors, seed, pose_id, "outfit_color")
    )
    bg = background or seeded_pick(banks.backgrounds, seed, pose_id, "background")
    if lora is None:
        li = seeded_index(len(banks.lora_presets), seed, pose_id, "lora")
        lora = banks.lora_presets[li]
    return _make_job(
        pose_id=pose_id,
        seed=seed,
        pose_comp=pose_comp,
        expression=expr,
        outfit_type=otype,
        outfit_color=ocol,
        background=bg,
        lora=lora,
        mode="random_seeded",
        job_index=job_index,
    )


def _cartesian_indices(
    job_i: int,
    n_outfit: int,
    n_concept: int,
    n_expr: int,
) -> tuple[int, int, int]:
    """Legacy mixed-radix stride (compat for old cartesian job_index callers)."""
    period = max(1, n_outfit * n_concept * n_expr)
    k = job_i % period
    oi = (k * 7) % n_outfit
    ci = (k * 11) % n_concept
    ei = (k * 13) % n_expr
    return oi, ci, ei


def iter_cartesian(
    poses: Sequence[str],
    banks: PhraseBanks,
    *,
    base_seed: int = 0,
    pose_meta: dict[str, dict[str, str]] | None = None,
) -> Iterator[ComboJob]:
    """Yield one ComboJob per full cartesian combo. seed = base_seed + job_index."""
    meta = pose_meta or {}
    axes = (
        banks.expressions,
        banks.outfit_types,
        banks.outfit_colors,
        banks.backgrounds,
        banks.lora_presets,
    )
    job_index = 0
    for pose_id in poses:
        pm = meta.get(pose_id, {})
        pose_comp = pose_comp_short(
            pose_id, pm.get("composition", ""), pm.get("positive_extra", "")
        )
        for expression, otype, ocolor, background, lora in itertools.product(*axes):
            seed = base_seed + job_index
            yield _make_job(
                pose_id=pose_id,
                seed=seed,
                pose_comp=pose_comp,
                expression=expression,
                outfit_type=otype,
                outfit_color=ocolor,
                background=background,
                lora=lora,
                mode="cartesian",
                job_index=job_index,
            )
            job_index += 1


def build_jobs_cartesian(
    track: str,
    poses: Sequence[str],
    base_seed: int,
    *,
    max_jobs: int | None = 0,
    allow_huge: bool = False,
    banks_dir: Path | None = None,
    private_banks_dir: Path | None = None,
    pose_meta: dict[str, dict[str, str]] | None = None,
) -> list[ComboJob]:
    """Build full cartesian jobs (1 image per combo).

    max_jobs:
      None or 0 → no hard cap (prints warning when total is large)
      >0 → if total > max_jobs and not allow_huge: raise
           if allow_huge: take first max_jobs with warning
    """
    banks = load_phrase_banks(track, banks_dir=banks_dir, private_banks_dir=private_banks_dir)
    total = estimate_combo_space(banks, len(poses))
    cap = None if max_jobs in (None, 0) else int(max_jobs)

    if cap is None:
        if total > 1000:
            # Rough MPS still timing (~60s/img); warn only — do not refuse.
            est_h = total * 60 / 3600
            print(
                f"WARN cartesian: total={total} with max_jobs=0 (no limit). "
                f"Rough ETA ≈ {est_h:.0f}h at ~60s/img. Smoke with max_jobs=64."
            )
    elif total > cap:
        msg = (
            f"cartesian total={total} exceeds max_jobs={cap}. "
            f"axes={count_axes(track, poses)}. "
            "Set max_jobs=0 for full run, raise the cap, shrink banks/poses, "
            f"or set allow_huge=True (will take first {cap})."
        )
        if not allow_huge:
            raise ValueError(msg)
        print("WARN", msg)

    jobs: list[ComboJob] = []
    for job in iter_cartesian(poses, banks, base_seed=base_seed, pose_meta=pose_meta):
        jobs.append(job)
        if cap is not None and len(jobs) >= cap:
            break
    return jobs


def interleave_longest(a: Sequence[ComboJob], b: Sequence[ComboJob]) -> list[ComboJob]:
    """Zip-longest interleave: a0,b0,a1,b1,... then remainder of the longer list."""
    out: list[ComboJob] = []
    n = max(len(a), len(b))
    for i in range(n):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def build_interleaved_both(
    sfw_poses: Sequence[str],
    nsfw_poses: Sequence[str],
    base_seed: int,
    *,
    mode: ComboMode = "cartesian",
    max_jobs: int | None = 0,
    allow_huge: bool = False,
    seeds_per_pose: int = 1,
    banks_dir: Path | None = None,
    private_banks_dir: Path | None = None,
    pose_meta: dict[str, dict[str, str]] | None = None,
) -> list[ComboJob]:
    """Build SFW + NSFW combo lists, then interleave SFW,NSFW,SFW,NSFW,...

    Each track keeps its own cartesian/random_seeded seed = base_seed + track index
    (1 image per combo when mode=cartesian). max_jobs applies per track.
    """
    if not sfw_poses:
        raise ValueError("build_interleaved_both: sfw_poses empty")
    if not nsfw_poses:
        raise ValueError("build_interleaved_both: nsfw_poses empty")

    sfw_jobs = build_combo_overrides(
        "sfw",
        sfw_poses,
        base_seed,
        seeds_per_pose,
        mode=mode,
        max_jobs=max_jobs,
        allow_huge=allow_huge,
        banks_dir=banks_dir,
        private_banks_dir=private_banks_dir,
        pose_meta=pose_meta,
    )
    nsfw_jobs = build_combo_overrides(
        "nsfw",
        nsfw_poses,
        base_seed,
        seeds_per_pose,
        mode=mode,
        max_jobs=max_jobs,
        allow_huge=allow_huge,
        banks_dir=banks_dir,
        private_banks_dir=private_banks_dir,
        pose_meta=pose_meta,
    )
    merged = interleave_longest(sfw_jobs, nsfw_jobs)
    print(
        f"interleaved both: sfw={len(sfw_jobs)} nsfw={len(nsfw_jobs)} "
        f"merged={len(merged)} order=s0,n0,s1,n1,... "
        f"(max_jobs/track={'unlimited' if max_jobs in (None, 0) else max_jobs})"
    )
    return merged


def build_combo_overrides(
    track: str,
    poses: Sequence[str],
    base_seed: int,
    seeds_per_pose: int,
    *,
    mode: ComboMode = "random_seeded",
    max_jobs: int | None = None,
    allow_huge: bool = False,
    banks_dir: Path | None = None,
    private_banks_dir: Path | None = None,
    pose_meta: dict[str, dict[str, str]] | None = None,
) -> list[ComboJob]:
    """Build combo jobs for the selected mode.

    random_seeded: seeds_per_pose images per pose (hash picks axes).
    cartesian: ignores seeds_per_pose as multiplier; 1 image per combo;
               seed = base_seed + job_index.
    """
    if mode == "cartesian":
        return build_jobs_cartesian(
            track,
            poses,
            base_seed,
            max_jobs=max_jobs,
            allow_huge=allow_huge,
            banks_dir=banks_dir,
            private_banks_dir=private_banks_dir,
            pose_meta=pose_meta,
        )

    banks = load_phrase_banks(track, banks_dir=banks_dir, private_banks_dir=private_banks_dir)
    meta = pose_meta or {}
    jobs: list[ComboJob] = []
    job_i = 0
    for pi, pose_id in enumerate(poses):
        pm = meta.get(pose_id, {})
        for si in range(seeds_per_pose):
            seed = base_seed + pi * 1000 + si * 97
            jobs.append(
                build_combo_job(
                    pose_id,
                    seed,
                    banks,
                    composition=pm.get("composition", ""),
                    positive_extra=pm.get("positive_extra", ""),
                    mode="random_seeded",
                    job_index=job_i,
                )
            )
            job_i += 1
            if max_jobs not in (None, 0) and len(jobs) >= int(max_jobs):
                return jobs
    return jobs


def bank_summary(track: str, *, redact_private_text: bool = True, n_poses: int = 0) -> dict[str, Any]:
    """Counts + sample paths for preflight. NSFW item text redacted by default."""
    banks = load_phrase_banks(track)
    pose_stubs = [f"pose_{i}" for i in range(n_poses)] if n_poses else None
    axes = count_axes(track, pose_stubs)

    if _is_nsfw_track(track) and redact_private_text:
        samples = "redacted"
        sample_types: list[str] | int = len(banks.outfit_types)
        sample_colors: list[str] | int = len(banks.outfit_colors)
        sample_bg: list[str] | int = len(banks.backgrounds)
        sample_expr: list[str] | int = len(banks.expressions)
    else:
        samples = "preview"
        sample_types = banks.outfit_types[:3]
        sample_colors = banks.outfit_colors[:3]
        sample_bg = banks.backgrounds[:3]
        sample_expr = banks.expressions[:3]

    return {
        "track": banks.track,
        "counts": banks.counts(),
        "axes": {
            "expression": len(banks.expressions),
            "outfit_type": len(banks.outfit_types),
            "outfit_color": len(banks.outfit_colors),
            "background": len(banks.backgrounds),
            "lora": len(banks.lora_presets),
        },
        "combo_space_per_pose": estimate_combo_space(banks, 1),
        "total_cartesian": axes.get("total_cartesian") if n_poses else None,
        "lora_presets": [{"name": p.name, "weight": p.weight} for p in banks.lora_presets],
        "private_loaded": banks.private_loaded,
        "public_loaded": banks.public_loaded,
        "paths": banks.paths,
        "samples": samples,
        "sample_outfit_types": sample_types,
        "sample_outfit_colors": sample_colors,
        "sample_backgrounds": sample_bg,
        "sample_expressions": sample_expr,
        # legacy keys
        "sample_outfits": sample_types,
        "sample_concepts": sample_bg,
    }


def sample_assembled_prompts(
    track: str,
    poses: Sequence[str],
    base_seed: int,
    *,
    n: int = 3,
    seeds_per_pose: int = 1,
    mode: ComboMode = "cartesian",
    redact_nsfw: bool = True,
) -> list[dict[str, Any]]:
    """Build a few combo jobs for preflight / dry-run display (no max_jobs refuse)."""
    if mode == "cartesian":
        banks = load_phrase_banks(track)
        jobs = list(
            itertools.islice(
                iter_cartesian(list(poses), banks, base_seed=base_seed),
                max(0, n),
            )
        )
    else:
        jobs = build_combo_overrides(
            track,
            list(poses),
            base_seed,
            seeds_per_pose,
            mode=mode,
            max_jobs=n,
            allow_huge=True,
        )
    out: list[dict[str, Any]] = []
    for j in jobs:
        if _is_nsfw_track(track) and redact_nsfw:
            out.append(
                {
                    "pose_id": j.pose_id,
                    "seed": j.seed,
                    "job_index": j.job_index,
                    "pose_prompt": "(private redacted)",
                    "scene_glue": "(private redacted)",
                    "outfit": "(private)",
                    "outfit_type": "(private)",
                    "outfit_color": "(private)",
                    "background": "(private)",
                    "expression": "(private)",
                    "lora_weight": j.lora_weight,
                }
            )
        else:
            out.append(
                {
                    "pose_id": j.pose_id,
                    "seed": j.seed,
                    "job_index": j.job_index,
                    "pose_prompt": j.pose_prompt,
                    "scene_glue": j.scene_glue,
                    "outfit": j.outfit,
                    "outfit_type": j.outfit_type,
                    "outfit_color": j.outfit_color,
                    "background": j.background,
                    "expression": j.expression,
                    "lora_weight": j.lora_weight,
                }
            )
    return out


def iter_unique_keys(jobs: Iterable[ComboJob]) -> set[tuple[str, str, str, str, str]]:
    return {
        (j.pose_id, j.outfit_type, j.outfit_color, j.background, j.expression) for j in jobs
    }
