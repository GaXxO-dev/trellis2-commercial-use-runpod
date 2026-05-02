#!/usr/bin/env python3
"""
Pre-download auxiliary models to network volume with file locking.
Prevents race conditions when multiple workers start simultaneously.

Models cached:
- microsoft/TRELLIS.2-4B (main model - handled by RunPod Model Caching)
- facebook/dinov3-vitl16-pretrain-lvd1689m (DINOv3 - gated, requires HF_TOKEN)
- ZhengPeng7/BiRefNet (BiRefNet background removal)

Usage:
    python preload_models.py [--force]

Environment variables:
    HF_HOME - Cache directory (default: /runpod-volume/huggingface-cache)
    HF_TOKEN - HuggingFace token (required for gated models)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from filelock import FileLock

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

HF_CACHE = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
# Don't set HF_HUB_CACHE - let HF create hub/ subdirectory automatically
LOCK_DIR = "/tmp/model_locks"
LOCK_TIMEOUT = 600  # 10 minutes

AUXILIARY_MODELS = [
    {
        "repo_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "description": "DINOv3 feature extractor (gated)",
        "gated": True,
        "required": True,
    },
    {
        "repo_id": "ZhengPeng7/BiRefNet",
        "description": "BiRefNet background removal",
        "gated": False,
        "required": True,
    },
]


def get_cache_path(repo_id: str) -> Path:
    """Get the expected cache path for a model."""
    cache_name = f"models--{repo_id.replace('/', '--')}"
    return Path(HF_CACHE) / "hub" / cache_name


def is_model_cached(repo_id: str) -> bool:
    """Check if a model is already in the cache."""
    cache_path = get_cache_path(repo_id)
    if not cache_path.exists():
        return False

    # Check for refs/main which indicates a complete download
    refs_main = cache_path / "refs" / "main"
    if refs_main.exists():
        return True

    # Check for snapshots directory with content
    snapshots_dir = cache_path / "snapshots"
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            # Check that snapshot has actual files
            snapshot_path = snapshots[0]
            if list(snapshot_path.iterdir()):
                return True

    return False


def download_model(repo_id: str, description: str, gated: bool = False) -> bool:
    """Download a model with retries and proper error handling."""
    from huggingface_hub import snapshot_download, HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    print(f"[Preload] Downloading {description}...")
    print(f"[Preload]   Repository: {repo_id}")

    # Check if gated and token
    if gated:
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token or hf_token.startswith("{{"):
            print(f"[Preload] ERROR: HF_TOKEN not set for gated model: {repo_id}")
            print("[Preload] Set HF_TOKEN environment variable with access to the model.")
            return False

    max_retries = 3
    for attempt in range(max_retries):
        try:
            path = snapshot_download(
                repo_id,
                local_files_only=False,
                etag_timeout=120,
                resume_download=True,
                local_dir_use_symlinks=False,
            )
            print(f"[Preload] SUCCESS: {description}")
            print(f"[Preload]   Cached at: {path}")
            return True

        except GatedRepoError as e:
            print(f"[Preload] ERROR: Gated repo access denied: {repo_id}")
            print(f"[Preload] Accept the license at: https://huggingface.co/{repo_id}")
            print(f"[Preload] Then set HF_TOKEN with 'read' permissions.")
            return False

        except RepositoryNotFoundError as e:
            print(f"[Preload] ERROR: Repository not found: {repo_id}")
            return False

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[Preload] ERROR: Failed after {max_retries} attempts: {e}")
                return False
            print(f"[Preload]   Attempt {attempt + 1}/{max_retries} failed: {e}")
            print("[Preload]   Retrying in 5 seconds...")
            time.sleep(5)

    return False


def download_with_lock(model_info: dict, force: bool = False) -> bool:
    """Download a model with file locking to prevent concurrent downloads."""
    repo_id = model_info["repo_id"]
    description = model_info["description"]
    gated = model_info.get("gated", False)
    required = model_info.get("required", True)

    # Create lock directory
    os.makedirs(LOCK_DIR, exist_ok=True)

    lock_file = os.path.join(LOCK_DIR, f"{repo_id.replace('/', '--')}.lock")
    lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)

    try:
        with lock:
            # Double-check if already cached (another worker may have downloaded)
            if not force and is_model_cached(repo_id):
                print(f"[Preload] Already cached: {description}")
                return True

            # Download
            success = download_model(repo_id, description, gated)

            if not success and required:
                print(f"[Preload] FAILED: Required model {description} not available")
                return False

            return success

    except Timeout:
        print(f"[Preload] TIMEOUT: Could not acquire lock for {description}")
        print("[Preload] Another worker may be downloading. Waiting...")
        # Wait and check if it's cached after timeout
        time.sleep(60)
        if is_model_cached(repo_id):
            print(f"[Preload] {description} was downloaded by another worker.")
            return True
        return False


def verify_cached_models() -> dict:
    """Verify which models are cached and return status."""
    all_models = [
        {"repo_id": "microsoft/TRELLIS.2-4B", "description": "TRELLIS.2 main model"},
    ] + AUXILIARY_MODELS

    status = {}
    for model in all_models:
        repo_id = model["repo_id"]
        cached = is_model_cached(repo_id)
        status[repo_id] = {
            "description": model["description"],
            "cached": cached,
            "path": str(get_cache_path(repo_id)),
        }

    return status


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download auxiliary models to network volume"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models are cached",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify cached models, don't download",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TRELLIS.2 Model Pre-loader")
    print("=" * 60)

    # Check network volume
    if not os.path.exists("/runpod-volume"):
        print("[Preload] ERROR: /runpod-volume not found!")
        print("[Preload] Network volume must be attached.")
        sys.exit(1)

    # Setup cache directory
    os.makedirs(f"{HF_CACHE}/hub", exist_ok=True)
    print(f"[Preload] Cache directory: {HF_CACHE}")

    # Check HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token and not hf_token.startswith("{{"):
        print(f"[Preload] HF_TOKEN: configured ({len(hf_token)} chars)")
    else:
        print("[Preload] WARNING: HF_TOKEN not configured!")
        print("[Preload] Gated models (DINOv3) will fail without valid token.")

    # Verify status
    print("\n[Preload] Checking cache status...")
    status = verify_cached_models()

    print("\n[Preload] Cache status:")
    for repo_id, info in status.items():
        state = "CACHED" if info["cached"] else "NOT CACHED"
        print(f"  [{state:12}] {info['description']}")
        print(f"               {repo_id}")

    if args.verify_only:
        print("\n[Preload] Verify-only mode complete.")
        all_cached = all(s["cached"] for s in status.values())
        sys.exit(0 if all_cached else 1)

    # Download auxiliary models
    print("\n[Preload] Processing auxiliary models...")
    print("-" * 60)

    all_success = True
    for model_info in AUXILIARY_MODELS:
        success = download_with_lock(model_info, force=args.force)
        if not success and model_info.get("required", True):
            all_success = False
        print()

    # Final verification
    print("-" * 60)
    print("\n[Preload] Final cache status:")
    status = verify_cached_models()

    for repo_id, info in status.items():
        state = "CACHED" if info["cached"] else "MISSING"
        print(f"  [{state:7}] {info['description']}")

    if all_success:
        print("\n[Preload] SUCCESS: All required models are cached.")
        print("[Preload] Workers can now use offline mode (HF_HUB_OFFLINE=1)")
        sys.exit(0)
    else:
        print("\n[Preload] FAILED: Some required models are missing.")
        print("[Preload] Check HF_TOKEN and network volume permissions.")
        sys.exit(1)


if __name__ == "__main__":
    main()