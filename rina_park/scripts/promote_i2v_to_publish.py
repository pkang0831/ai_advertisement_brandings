#!/usr/bin/env python3
"""Promote I2V hero stills to publishing queue.

This script takes auto-passed stills from i2v_heroes and adds them to the 
publishing queue for Instagram/Patreon.

Usage:
  # Promote auto-passed stills to publishing queue
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/promote_i2v_to_publish.py

  # Promote specific stills
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/promote_i2v_to_publish.py \\
    --image rina_park/out/i2v_heroes/20260730T211045Z/hand_pockets_3q/s00_seed30072026.jpg

  # Promote with custom track and tier
  PYTHONPATH=rina_park .venv/bin/python rina_park/scripts/promote_i2v_to_publish.py \\
    --image <path> --track ig --tier a
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RINA = Path(__file__).resolve().parents[1]
QUEUE_ROOT = RINA / "queues"
QUEUE_FILE = QUEUE_ROOT / "rina_ig_queue.csv"
CURRENT_HEROES = RINA / "out" / "i2v_heroes" / "current"
HEROES_RUNS = RINA / "out" / "i2v_heroes"

# Default IG post schedule (from strategy.md)
DEFAULT_IG_SCHEDULE = {
    "monday": "19:30",
    "wednesday": "12:15",
    "friday": "20:30",
    "sunday": "10:30",
}


def _get_next_available_slot(queued_items: list[dict], track: str) -> tuple[str, str]:
    """Find next available IG slot based on schedule."""
    import calendar
    
    today = datetime.now(timezone.utc)
    
    # Find next available day/time from schedule
    days_order = ["monday", "wednesday", "friday", "sunday"]
    
    for day_name in days_order:
        day_num = list(calendar.day_name).index(day_name.capitalize())
        days_ahead = day_num - today.weekday()
        if days_ahead < 0:
            days_ahead += 7
        
        next_day = today + timedelta(days=days_ahead)
        time_str = DEFAULT_IG_SCHEDULE[day_name]
        schedule_dt = datetime.strptime(f"{next_day.strftime('%Y-%m-%d')} {time_str}", 
                                       "%Y-%m-%d %H:%M")
        schedule_dt = schedule_dt.replace(tzinfo=timezone.utc)
        
        # Check if this slot is taken
        is_taken = any(
            item.get("production_id", "").startswith(f"ig_{next_day.strftime('%Y%m%d')}")
            for item in queued_items
        )
        
        if not is_taken:
            return day_name, schedule_dt.isoformat()
    
    # Fallback: next Monday
    days_ahead = 7 - today.weekday()
    if days_ahead <= 0:
        days_ahead = 7
    next_day = today + timedelta(days=days_ahead)
    return "monday", f"{next_day.strftime('%Y-%m-%d')} 19:30"


def _generate_asset_id(production_id: str, idx: int) -> str:
    """Generate asset ID for queue entry."""
    return f"{production_id}_img{idx:02d}"


def _load_existing_queue() -> list[dict]:
    """Load existing queue from CSV."""
    if not QUEUE_FILE.exists():
        return []
    
    import csv
    items = []
    with QUEUE_FILE.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items


def _save_queue(items: list[dict]) -> None:
    """Save queue to CSV."""
    import csv
    
    fieldnames = [
        'asset_id', 'track', 'tier', 'production_id', 'slide_role', 'location', 
        'outfit', 'shot', 'phone_real_notes', 'seed', 'width', 'height', 
        'steps', 'cfg', 'prompt', 'negative_prompt', 'output_subdir', 
        'image_filename', 'status'
    ]
    
    with QUEUE_FILE.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(item)


def _load_sidecar(image_path: Path) -> dict | None:
    """Load sidecar JSON for image."""
    sidecar_path = image_path.with_suffix('.jpg.json')
    if sidecar_path.exists():
        return json.loads(sidecar_path.read_text(encoding='utf-8'))
    return None


def _load_scorecard(image_path: Path) -> dict | None:
    """Load scorecard JSON for image."""
    scorecard_path = image_path.parent / f"{image_path.stem}_scorecard.json"
    if scorecard_path.exists():
        return json.loads(scorecard_path.read_text(encoding='utf-8'))
    return None


def _build_ig_prompt(sidecar: dict, scorecard: dict) -> tuple[str, str]:
    """Build IG prompt from sidecar data."""
    prompt_parts = []
    
    # Extract key elements from sidecar
    if 'prompt' in sidecar:
        prompt_parts.append(sidecar['prompt'])
    
    # Add composition elements from scorecard if available
    if scorecard and 'auto_qc' in scorecard:
        auto_qc = scorecard['auto_qc']
        if auto_qc.get('identity', {}).get('pass'):
            prompt_parts.append("soft glam Korean-Canadian woman 27")
        if auto_qc.get('hands', {}).get('pass'):
            prompt_parts.append("hands naturally positioned")
    
    prompt = ", ".join(prompt_parts) if prompt_parts else "Rina Park lifestyle shot"
    
    # Build negative prompt
    negative_parts = [
        "nude", "explicit", "nipples", "genitals", 
        "painterly", "plastic", "beauty filter", "CGI",
        "text", "watermark", "logo", "deformed",
        "hands near face", "splayed fingers", "bare feet"
    ]
    
    return prompt, ", ".join(negative_parts)


def _get_location_from_sidecar(sidecar: dict) -> str:
    """Extract location from sidecar or use default."""
    if sidecar and 'prompt' in sidecar:
        prompt = sidecar['prompt'].lower()
        if 'trench' in prompt or 'sidewalk' in prompt:
            return "soft morning sidewalk"
        elif 'apartment' in prompt or 'loungewear' in prompt:
            return "sunlit apartment"
        elif 'sofa' in prompt or 'home' in prompt:
            return "home living room"
        elif 'yoga' in prompt or 'studio' in prompt:
            return "home yoga studio"
    
    return "Toronto park, soft golden hour lighting"


def _get_outfit_from_sidecar(sidecar: dict) -> str:
    """Extract outfit from sidecar or use default."""
    if sidecar and 'prompt' in sidecar:
        prompt = sidecar['prompt'].lower()
        if 'trench' in prompt:
            return "beige trench coat"
        elif 'loungewear' in prompt or 'apartment' in prompt:
            return "soft loungewear"
        elif 'yoga' in prompt or 'mat' in prompt:
            return "athletic yoga wear"
    
    return "casual athleisure"


def main() -> int:
    import csv
    from datetime import timedelta
    
    ap = argparse.ArgumentParser(description="Promote I2V hero stills to publishing queue")
    ap.add_argument("--image", type=Path, help="Specific image to promote")
    ap.add_argument("--run-id", type=str, help="Run ID to process (default: most recent)")
    ap.add_argument("--track", type=str, default="ig", choices=["ig", "patreon"], help="Track to add to")
    ap.add_argument("--tier", type=str, default="a", choices=["a", "b", "c"], help="Patreon tier")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be queued without writing")
    
    args = ap.parse_args()
    
    # Find image(s) to promote
    images_to_process = []
    
    if args.image:
        # Single image specified
        if not args.image.exists():
            print(f"ERROR: Image not found: {args.image}")
            return 1
        images_to_process.append(args.image)
    else:
        # Find auto-passed images from most recent run
        if args.run_id:
            run_dir = HEROES_RUNS / args.run_id
        else:
            # Find most recent run
            runs = sorted(HEROES_RUNS.glob("20*"))
            if not runs:
                print("ERROR: No i2v_heroes runs found")
                return 1
            run_dir = runs[-1]
        
        if not run_dir.exists():
            print(f"ERROR: Run directory not found: {run_dir}")
            return 1
        
        # Find auto-passed images
        for pose_dir in run_dir.glob("*"):
            if pose_dir.is_dir():
                for scorecard_file in pose_dir.glob("*_scorecard.json"):
                    card = json.loads(scorecard_file.read_text(encoding='utf-8'))
                    if card.get('auto_pass'):
                        img_path = Path(card['image'])
                        if img_path.exists():
                            images_to_process.append(img_path)
    
    if not images_to_process:
        print("ERROR: No images to process")
        return 1
    
    # Load existing queue
    existing_queue = _load_existing_queue()
    
    # Process each image
    new_items = []
    
    for img_path in images_to_process:
        print(f"\nProcessing: {img_path.name}")
        
        # Load sidecar and scorecard
        sidecar = _load_sidecar(img_path)
        scorecard = _load_scorecard(img_path)
        
        if not sidecar:
            print(f"  WARNING: No sidecar found for {img_path.name}, using defaults")
            sidecar = {}
        
        # Extract pose_id from path
        pose_id = img_path.parent.name
        
        # Determine production ID
        production_id = f"ig_{datetime.now(timezone.utc).strftime('%Y%m%d')}_001"
        
        # Find next available slot
        day_name, schedule_dt = _get_next_available_slot(existing_queue + new_items, args.track)
        
        # Build prompt
        prompt, negative = _build_ig_prompt(sidecar, scorecard)
        
        # Extract location and outfit
        location = _get_location_from_sidecar(sidecar)
        outfit = _get_outfit_from_sidecar(sidecar)
        
        # Determine shot type
        shot = "three-quarter body"
        if pose_id == "hand_pockets_3q":
            shot = "three-quarter lifestyle"
        elif pose_id == "hand_cropped_out":
            shot = "upper-body portrait"
        
        # Phone real notes
        phone_notes = f" candid phone photo, soft natural light"
        if "golden" in location.lower():
            phone_notes = " candid phone photo, soft golden hour light"
        elif "apartment" in location.lower():
            phone_notes = " candid phone photo, soft indoor light"
        
        # Create queue entry
        asset_id = _generate_asset_id(production_id, len(new_items) + 1)
        width = sidecar.get('i2v_res', [1080, 1920])[0] if 'i2v_res' in sidecar else 1080
        height = sidecar.get('i2v_res', [1080, 1920])[1] if 'i2v_res' in sidecar else 1920
        
        queue_item = {
            'asset_id': asset_id,
            'track': args.track,
            'tier': args.tier if args.track == 'patreon' else '',
            'production_id': production_id,
            'slide_role': 'main',
            'location': location,
            'outfit': outfit,
            'shot': shot,
            'phone_real_notes': phone_notes,
            'seed': scorecard.get('seed', 0) if scorecard else 0,
            'width': width,
            'height': height,
            'steps': sidecar.get('steps', 36),
            'cfg': sidecar.get('cfg', 4.2),
            'prompt': prompt,
            'negative_prompt': negative,
            'output_subdir': f"i2v_heroes/{img_path.parent.name}",
            'image_filename': img_path.name,
            'status': 'queued',
        }
        
        new_items.append(queue_item)
        print(f"  Created queue entry: {asset_id}")
        print(f"  Production ID: {production_id}")
        print(f"  Scheduled for: {day_name} {schedule_dt}")
    
    # Combine and save queue
    final_queue = existing_queue + new_items
    
    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Would add {len(new_items)} items to queue")
        for item in new_items:
            print(f"  - {item['asset_id']}: {item['production_id']}")
        return 0
    
    # Save queue
    _save_queue(final_queue)
    
    print(f"\n=== SUCCESS ===")
    print(f"Added {len(new_items)} items to queue")
    print(f"Total queue items: {len(final_queue)}")
    print(f"Queue file: {QUEUE_FILE}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())