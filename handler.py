"""
TRELLIS.2 RunPod Serverless Handler

Usage:
    POST /run with JSON body:
    {
        "input": {
            "image": "base64...",
            "resolution": 512 | 1024 | 1536,
            "texture_size": 1024 | 2048 | 4096,
            "output_format": "glb" | "obj" | "ply",
            "seed": 42,
            "sparse_structure_steps": 20,
            "slat_sampler_steps": 20,
            "max_num_tokens": 49152,
            "extension_webp": true
        }
    }

Required env vars:
    HF_TOKEN: HuggingFace token (required for gated models: facebook/dinov3-*)
    R2_ENDPOINT_URL: Cloudflare R2 endpoint
    R2_ACCESS_KEY_ID: R2 access key
    R2_SECRET_ACCESS_KEY: R2 secret key
    R2_BUCKET_NAME: R2 bucket name
    R2_PUBLIC_URL: (optional) Public base URL for R2 bucket

RunPod setup:
    - Attach a network volume for model caching (100GB+ recommended)
    - Models are cached at /runpod-volume/huggingface-cache/
"""

import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

HF_CACHE_DIR = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
if os.path.exists("/runpod-volume"):
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    print(f"[Startup] HuggingFace cache directory: {HF_CACHE_DIR}")
else:
    print("[Startup] WARNING: /runpod-volume not found - models will not be persisted across restarts")
    print("[Startup]         Attach a network volume for model caching to reduce cold start times")

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN and not HF_TOKEN.startswith("{{"):
    print(f"[Startup] HF_TOKEN configured (length: {len(HF_TOKEN)} chars)")
elif not HF_TOKEN:
    print("[Startup] WARNING: HF_TOKEN not set - gated models (facebook/dinov3-*) may fail")
    print("[Startup]         Set HF_TOKEN as a RunPod secret with access to facebook/dinov3 models")
elif HF_TOKEN.startswith("{{"):
    raise RuntimeError(
        "HF_TOKEN not resolved! The secret {{ RUNPOD_SECRET_HF_TOKEN }} was not injected. "
        "Verify the secret exists in RunPod Console → Settings → Secrets"
    )

import runpod
import sys
import base64
import tempfile
import time
import uuid
import torch
import requests
import boto3
from botocore.config import Config
from io import BytesIO
from PIL import Image

# Add TRELLIS.2 to path
TRELLIS_PATH = os.environ.get("TRELLIS_PATH", "/app/TRELLIS.2")
sys.path.insert(0, TRELLIS_PATH)

HF_MODEL_ID = "microsoft/TRELLIS.2-4B"

# Global pipeline - initialized once at worker startup
pipeline = None


def upload_to_r2(filepath: str, filename: str) -> str:
    client = boto3.client(
        's3',
        endpoint_url=os.environ['R2_ENDPOINT_URL'],
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4'),
        region_name='auto',
    )
    bucket = os.environ['R2_BUCKET_NAME']
    content_types = {
        'glb': 'model/gltf-binary',
        'obj': 'application/object',
        'ply': 'application/ply',
    }
    ext = filename.rsplit('.', 1)[-1].lower()
    client.upload_file(
        filepath, bucket, filename,
        ExtraArgs={'ContentType': content_types.get(ext, 'application/octet-stream')},
    )
    public_url = os.environ.get('R2_PUBLIC_URL', '').rstrip('/')
    if public_url:
        return f"{public_url}/{filename}"
    return client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': filename},
        ExpiresIn=86400 * 7,
    )


def load_model():
    """Load TRELLIS.2 model into GPU memory.
    
    Uses HuggingFace cache at HF_HOME for model persistence.
    Requires HF_TOKEN for gated models (facebook/dinov3-*).
    """
    global pipeline

    if pipeline is not None:
        return pipeline

    print("=" * 50)
    print("Loading TRELLIS.2 model...")
    print("=" * 50)
    start_time = time.time()

    print(f"[Model] Model ID: {HF_MODEL_ID}")
    print(f"[Model] HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    print(f"[Model] HF_TOKEN: {'configured' if os.environ.get('HF_TOKEN') and not os.environ.get('HF_TOKEN', '').startswith('{{') else 'MISSING'}")

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(HF_MODEL_ID)
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg or "403" in error_msg:
            raise RuntimeError(
                f"HuggingFace authentication failed. Ensure HF_TOKEN is set correctly "
                f"and you have accepted the license for facebook/dinov3-vitl16-pretrain-lvd1689m. "
                f"Original error: {error_msg}"
            ) from e
        if "Repository Not Found" in error_msg or "not found" in error_msg.lower():
            raise RuntimeError(
                f"Model repository not found. Check: 1) HF_TOKEN is valid, "
                f"2) Network volume is attached (for caching), "
                f"3) Internet connectivity. Original error: {error_msg}"
            ) from e
        raise

    pipeline.low_vram = False
    pipeline.cuda()

    elapsed = time.time() - start_time
    print(f"[Model] Model loaded successfully in {elapsed:.2f}s")
    print("=" * 50)

    return pipeline


def decode_base64_image(data: str) -> Image.Image:
    """Decode base64 image string."""
    # Remove data URL prefix if present
    if "," in data:
        data = data.split(",", 1)[1]
    image_data = base64.b64decode(data)
    return Image.open(BytesIO(image_data)).convert("RGBA")


def validate_image(image: Image.Image) -> None:
    MAX_PIXELS = 4096 * 4096
    pixels = image.width * image.height
    if pixels > MAX_PIXELS:
        raise ValueError(
            f"Image too large: {image.width}x{image.height} ({pixels:,} px). "
            f"Maximum: {MAX_PIXELS:,} px (~4096x4096)"
        )


def download_image(url: str) -> Image.Image:
    """Download image from URL with retry."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGBA")
            validate_image(img)
            return img
        except (requests.RequestException, OSError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def handler(job):
    """
    RunPod serverless handler for TRELLIS.2 image-to-3D.
    API compatible with mockupWebsite frontend.

    Input parameters:
    - image: str - Base64 encoded image or URL (required)
    - resolution: int - 512, 1024, or 1536 (default: 1024)
    - texture_size: int - 1024, 2048, or 4096 (default: 2048)
    - output_format: str - "glb", "obj", or "ply" (default: "glb")
    - seed: int - Random seed (default: random)
    - sparse_structure_steps: int - Steps for sparse structure (default: 20)
    - slat_sampler_steps: int - Steps for SLAT sampler (default: 20)
    - max_num_tokens: int - Max tokens for cascade upsampling (default: 49152)
    - extension_webp: bool - Use WebP extension for GLB textures (default: true)

    Returns:
    - model_url: str - URL to download the generated 3D model
    - metadata: object - Generation metadata
    """
    job_input = job["input"]
    req_id = job.get("id", str(uuid.uuid4())[:8])
    start_time = time.time()

    # Get image - support both 'image' (website) and 'input_image' (direct API)
    image_data = job_input.get("image") or job_input.get("input_image")
    if not image_data:
        return {"error": "image is required"}

    # Load model (cached after first call)
    runpod.serverless.progress_update(job, "Loading model...")
    pipe = load_model()

    # Load image
    runpod.serverless.progress_update(job, "Processing image...")
    try:
        if image_data.startswith(("http://", "https://")):
            image = download_image(image_data)
        else:
            image = decode_base64_image(image_data)
            validate_image(image)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    print(f"[{req_id}] Image size: {image.size}")

    # Extract parameters with defaults
    resolution = job_input.get("resolution", 1024)
    texture_size = job_input.get("texture_size", 2048)
    output_format = job_input.get("output_format", "glb")
    seed = job_input.get("seed")
    sparse_structure_steps = job_input.get("sparse_structure_steps", 20)
    slat_sampler_steps = job_input.get("slat_sampler_steps", 20)
    max_num_tokens = job_input.get("max_num_tokens", 49152)
    extension_webp = job_input.get("extension_webp", True)

    if seed is None:
        import random
        seed = random.randint(0, 2**32 - 1)

    # Simplify target always 16777216 (nvdiffrast limit, matches official example.py)
    simplify_target = 16777216

    # Map resolution to decimation target
    decimation_targets = {512: 100000, 1024: 500000, 1536: 1000000}
    decimation_target = decimation_targets.get(resolution, 500000)

    # Map resolution to pipeline_type (TRELLIS.2 API requirement)
    pipeline_types = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}
    pipeline_type = pipeline_types.get(resolution, "1024_cascade")

    # Run inference
    runpod.serverless.progress_update(job, "Generating 3D model...")
    print(f"[{req_id}] Running inference: resolution={resolution}, pipeline_type={pipeline_type}, seed={seed}")

    try:
        mesh = pipe.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params={"steps": sparse_structure_steps},
            shape_slat_sampler_params={"steps": slat_sampler_steps},
            tex_slat_sampler_params={"steps": slat_sampler_steps},
            max_num_tokens=max_num_tokens,
        )[0]
        mesh.simplify(simplify_target)
        torch.cuda.empty_cache()
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    inference_time = time.time() - start_time
    print(f"[{req_id}] Inference completed in {inference_time:.2f}s")

    # Export mesh
    runpod.serverless.progress_update(job, "Exporting 3D model...")
    try:
        import o_voxel

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )

        download_name = f"{uuid.uuid4()}.{output_format}"
        r2_key = f"trellis2/{download_name}"
        suffix = f".{output_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            model_path = f.name

        glb.export(model_path, extension_webp=(output_format == "glb" and extension_webp))
        torch.cuda.empty_cache()

        model_url = upload_to_r2(model_path, r2_key)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        os.unlink(model_path)

    except Exception as e:
        return {"error": f"Export failed: {str(e)}"}

    total_time = time.time() - start_time
    print(f"[{req_id}] Export completed. Model size: {model_size_mb:.2f}MB, Total time: {total_time:.2f}s")

    return {
        "model_url": model_url,
        "metadata": {
            "format": output_format,
            "download_name": download_name,
            "resolution": resolution,
            "pipeline_type": pipeline_type,
            "triangle_target": decimation_target,
            "texture_size": texture_size,
            "seed": seed,
            "generation_time_ms": int(total_time * 1000),
            "model_size_mb": round(model_size_mb, 2),
        },
    }


# Initialize model at worker startup (RunPod best practice)
print("=" * 60)
print("TRELLIS.2 RunPod Worker Initializing...")
print("=" * 60)

print(f"[Startup] Working directory: {os.getcwd()}")
print(f"[Startup] Python version: {sys.version}")
print(f"[Startup] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Startup] CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"[Startup] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"[Startup] HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
print(f"[Startup] HF_TOKEN: {'configured' if os.environ.get('HF_TOKEN') and not os.environ.get('HF_TOKEN', '').startswith('{{') else 'MISSING OR UNRESOLVED'}")

if not os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "").startswith("{{"):
    print("[Startup] ERROR: HF_TOKEN is not properly configured!")
    print("[Startup] This will cause model loading to fail for gated models.")
    print("[Startup] Fix: Add HF_TOKEN as a RunPod secret and reference it as:")
    print("[Startup]        HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}")

print("[Startup] Pre-loading model (this may take several minutes on first run)...")
try:
    load_model()
    print("[Startup] Model loaded successfully!")
except Exception as e:
    print(f"[Startup] WARNING: Model pre-load failed: {e}")
    print("[Startup] Model will be loaded on first request")
    import traceback
    traceback.print_exc()

# Start RunPod serverless worker
print("=" * 60)
print("Starting RunPod serverless handler...")
print("=" * 60)
runpod.serverless.start({"handler": handler})
