# TRELLIS.2 RunPod Serverless

RunPod serverless worker for [TRELLIS.2](https://github.com/GaXxO-dev/TRELLIS.2-commercial-use) — Microsoft's state-of-the-art 4B-parameter image-to-3D generative model. Converts a single image into a high-fidelity, PBR-textured 3D mesh (GLB/OBJ/PLY) with full material channels (base color, roughness, metallic, opacity).

Uses the [commercial-use fork](https://github.com/GaXxO-dev/TRELLIS.2-commercial-use) with MIT-licensed DRTK renderer (replaces the original NVidia-licensed nvdiffrast).

## Requirements

- **GPU**: NVIDIA A100 (80GB), H100 (80GB), or equivalent — minimum 48GB VRAM
- **RunPod Account**: Serverless endpoint with network volume (~$2.10/mo for 30GB)
- **Cloudflare R2**: S3-compatible storage for generated 3D models
- **Hugging Face Token**: Required for gated `facebook/dinov3-vitl16-pretrain-lvd1689m` model

## Prerequisites

### 1. HuggingFace Token Setup

TRELLIS.2 uses `facebook/dinov3-vitl16-pretrain-lvd1689m` for image feature extraction, which is a **gated model** requiring license acceptance.

1. Create a HuggingFace token: https://huggingface.co/settings/tokens
   - Select **Read** permissions
2. Accept the license: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
   - Requires Meta account sign-in
   - License acceptance may take a few minutes to propagate

### 2. Cloudflare R2 Setup

1. Create an R2 bucket in Cloudflare dashboard
2. Generate R2 API tokens (Access Key ID + Secret Access Key)
3. Note your:
   - Endpoint URL: `https://<account-id>.r2.cloudflarestorage.com`
   - Bucket name
   - Access Key ID
   - Secret Access Key
4. (Optional) Configure public access for direct download URLs

## RunPod Deployment

### Step 1: Create Network Volume

**Network Volume** stores auxiliary models (~3GB total):
- `facebook/dinov3-vitl16-pretrain-lvd1689m` (~1.2 GB) - Image features (gated)
- `ZhengPeng7/BiRefNet` (~1.5 GB) - Background removal

**Note:** The main `microsoft/TRELLIS.2-4B` model (~18GB) is handled by **RunPod Model Caching** (Step 4), not the network volume.

**Recommended network volume: 10 GB** (~$0.70/mo) — RunPod minimum

1. Go to **Storage → Network Volumes** in RunPod Console
2. Click **New Network Volume**
3. Configure:
   - **Name**: `trellis2-aux-models` (or your preference)
   - **Size**: `10` GB (RunPod minimum)
   - **Data Center**: Choose based on GPU availability
4. Note the volume ID (e.g., `nv-xxxxxx`)

### Step 2: Create RunPod Secrets

Go to **Settings → Secrets** in RunPod Console and create:

| Secret Name | Value |
|-------------|-------|
| `HF_TOKEN` | Your HuggingFace token |
| `R2_ACCESS_KEY_ID` | Cloudflare R2 access key ID |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 secret access key |

### Step 3: Build and Push Docker Image

```bash
# Clone the repository
git clone https://github.com/<your-username>/trellis2-commercial-use-runpod.git
cd trellis2-commercial-use-runpod

# Build the image (25-30 min on first build)
docker build -t trellis2-runpod:latest .

# Push to your registry
docker tag trellis2-runpod:latest <your-registry>/trellis2-runpod:latest
docker push <your-registry>/trellis2-runpod:latest
```

**Build Notes:**
- Requires PyTorch 2.6.0, flash-attn 2.7.3, DRTK (prebuilt wheels)
- Compiles CUDA extensions: o-voxel, CuMesh, FlexGEMM
- Base image: `nvidia/cuda:12.4.1-devel-ubuntu22.04`

### Step 4: Create Serverless Endpoint

Using **RunPod Console**:

1. Go to **Serverless → New Endpoint**
2. Configure:
   - **Name**: `trellis2-image-to-3d`
   - **Image**: `<your-registry>/trellis2-runpod:latest`
   - **GPU Types**: 
     - Primary: `NVIDIA H100 80GB HBM3` or `NVIDIA H100 PCIe`
     - Fallback: `NVIDIA A100-SXM4-80GB`, `NVIDIA A100 80GB PCIe`, `NVIDIA RTX 6000 Ada`
   - **Network Volume**: Select the volume created in Step 1
   - **Workers**: Min `0`, Max `3` (adjust based on expected load)
   - **Idle Timeout**: `5` seconds (default, keeps workers warm briefly)
   - **FlashBoot**: Enabled (default, faster cold starts)

3. **CRITICAL: Enable Model Caching** (reduces cold starts from minutes to seconds)
   
   In the endpoint configuration, scroll to **Model** field and enter:
   ```
   microsoft/TRELLIS.2-4B
   ```
   
   This tells RunPod to pre-download the main TRELLIS.2 model to hosts before workers start, eliminating the largest download from cold start time.

4. Set **Environment Variables**:
   
   | Variable | Value |
   |----------|-------|
   | `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
   | `R2_ACCESS_KEY_ID` | `{{ RUNPOD_SECRET_R2_ACCESS_KEY_ID }}` |
   | `R2_SECRET_ACCESS_KEY` | `{{ RUNPOD_SECRET_R2_SECRET_ACCESS_KEY }}` |
   | `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
   | `R2_BUCKET_NAME` | `<your-bucket-name>` |
   | `R2_PUBLIC_URL` | *(optional)* `https://pub-xxx.r2.dev` |

5. Click **Deploy Endpoint**

Using **RunPod API**:

```bash
curl -X POST "https://rest.runpod.io/v1/endpoints" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "trellis2-image-to-3d",
    "imageName": "<your-registry>/trellis2-runpod:latest",
    "gpuTypeIds": [
      "NVIDIA H100 80GB HBM3",
      "NVIDIA H100 PCIe", 
      "NVIDIA A100-SXM4-80GB",
      "NVIDIA A100 80GB PCIe"
    ],
    "networkVolumeId": "nv-xxxxxx",
    "workersMin": 0,
    "workersMax": 3,
    "idleTimeout": 5,
    "flashboot": true,
    "modelName": "microsoft/TRELLIS.2-4B",
    "env": {
      "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
      "R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_R2_ACCESS_KEY_ID }}",
      "R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_R2_SECRET_ACCESS_KEY }}",
      "R2_ENDPOINT_URL": "https://<account-id>.r2.cloudflarestorage.com",
      "R2_BUCKET_NAME": "<your-bucket-name>"
    }
  }'
```

**Note:** The `modelName` field enables RunPod Model Caching for the main TRELLIS.2 model, which significantly reduces cold start time.

### Step 5: First Request (Populates Cache)

On first cold start, the worker logs should show:

```
============================================================
TRELLIS.2 RunPod Worker Initializing...
============================================================
[Startup] Network volume detected: /runpod-volume
[Startup] HF cache directory: /runpod-volume/huggingface-cache
[Startup] Cached models status:
[Startup]   TRELLIS.2: NOT cached
[Startup]   DINOv3: NOT cached
[Startup]   BiRefNet: NOT cached
[Startup] Missing models will be downloaded with file locking...
[Startup] HF_TOKEN: configured
[Startup] Pre-loading model (this may take several minutes on first run)...
```

**First cold start** (8-15 minutes):
- ✅ `microsoft/TRELLIS.2-4B`: Pre-downloaded by RunPod Model Caching (fast)
- ⬇️ `facebook/dinov3-vitl16-pretrain-lvd1689m`: Downloads with file locking
- ⬇️ `ZhengPeng7/BiRefNet`: Downloads with file locking

**Subsequent cold starts** (10-30 seconds):
- All models load from network volume cache
- `HF_HUB_OFFLINE=1` prevents network checks

**Warm worker** (within idle timeout): < 1 second

### How Cold Start Optimization Works

```
┌─ Worker Start ──────────────────────────────────────────────┐
│ 1. Detect /runpod-volume                                     │
│ 2. Check if ALL models cached:                              │
│    ├─ microsoft/TRELLIS.2-4B (RunPod Model Caching)        │
│    ├─ facebook/dinov3-vitl16-pretrain-lvd1689m              │
│    └─ ZhengPeng7/BiRefNet                                    │
│                                                              │
│ IF all cached → Set HF_HUB_OFFLINE=1 (10-30 sec load)       │
│ IF any missing → Download with file locking                  │
│                 (first worker only, others wait)            │
│                 → Set HF_HUB_OFFLINE=1 after success        │
└──────────────────────────────────────────────────────────────┘
```

**Key optimizations:**
- **RunPod Model Caching**: Pre-downloads main TRELLIS.2 model to host
- **Network Volume**: Persists auxiliary models across worker restarts
- **Offline Mode**: Skips network checks when models are cached
- **File Locking**: Prevents concurrent downloads from multiple workers

## API Usage

### Endpoint URL

```
POST https://api.runpod.ai/v2/<endpoint-id>/runsync
Authorization: Bearer <api-key>
```

### Request Body

```json
{
  "input": {
    "image": "https://example.com/image.jpg",
    "resolution": 1024,
    "texture_size": 2048,
    "output_format": "glb",
    "seed": 42
  }
}
```

For base64-encoded images:

```json
{
  "input": {
    "image": "data:image/png;base64,iVBORw0KGgoAAAANS..."
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | **required** | Base64-encoded image or HTTPS URL |
| `resolution` | int | `1024` | Voxel resolution: `512`, `1024`, or `1536` |
| `texture_size` | int | `2048` | Texture resolution: `1024`, `2048`, or `4096` |
| `output_format` | string | `"glb"` | Output format: `"glb"`, `"obj"`, `"ply"` |
| `seed` | int | *random* | Random seed for reproducibility |
| `sparse_structure_steps` | int | `20` | Denoising steps for sparse structure |
| `slat_sampler_steps` | int | `20` | Denoising steps for shape/texture SLAT |
| `max_num_tokens` | int | `49152` | Max tokens for cascade upsampling |
| `extension_webp` | bool | `true` | Use WebP texture encoding (smaller files) |

### Response

```json
{
  "model_url": "https://pub-xxx.r2.dev/trellis2/abc123.glb",
  "metadata": {
    "format": "glb",
    "download_name": "abc123.glb",
    "resolution": 1024,
    "pipeline_type": "1024_cascade",
    "triangle_target": 500000,
    "texture_size": 2048,
    "seed": 42,
    "generation_time_ms": 17234,
    "model_size_mb": 18.42
  }
}
```

### Example: cURL

```bash
curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg",
      "resolution": 1024
    }
  }'
```

## Performance

| Resolution | Inference Time | Total (incl. export) |
|------------|----------------|----------------------|
| 512³ | ~3s | ~5s |
| 1024³ | ~17s | ~20s |
| 1536³ | ~60s | ~65s |

Benchmarks from NVIDIA H100 80GB. Times include model loading, inference, export, and R2 upload.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Request Flow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request ──┬──► RunPod Worker                                   │
│             │       │                                            │
│             │       ├── RunPod Model Cache (HOST):              │
│             │       │   └── microsoft/TRELLIS.2-4B (~18GB)     │
│             │       │                                            │
│             │       ├── Network Volume (/runpod-volume/):      │
│             │       │   └── facebook/dinov3-vitl16 (~1.2GB)    │
│             │       │   └── ZhengPeng7/BiRefNet (~1.5GB)       │
│             │       │                                            │
│             │       ├── TRELLIS.2 Pipeline (GPU)                 │
│             │       │   └── Image → 3D Voxel → Mesh              │
│             │       │                                            │
│             │       ├── o-voxel/DRTK Export                      │
│             │       │   └── Mesh → GLB/OBJ/PLY                   │
│             │       │                                            │
│             │       └── Cloudflare R2                           │
│             │           └── Upload → Return URL                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **RunPod Model Caching**: Pre-downloads `microsoft/TRELLIS.2-4B` to host (~18GB) — eliminates main model download from cold start
- **Network Volume**: Persists auxiliary models (DINOv3 + BiRefNet, ~3GB) across worker restarts — small 5GB volume is sufficient
- **Offline Mode**: After first download, workers use `HF_HUB_OFFLINE=1` for instant cache loads (10-30 sec)
- **File Locking**: First worker downloads auxiliary models, others wait and use cached copies
- **GPU memory**: Model stays loaded (`low_vram=False`) for sub-60s inference
- **Output storage**: Cloudflare R2 bypasses RunPod's 20MB response limit

## Troubleshooting

### "HF_TOKEN not resolved" or "{{ RUNPOD_SECRET_* }} in logs"

RunPod secret template syntax wasn't replaced. Fix:

1. Go to **RunPod Console → Settings → Secrets**
2. Create secrets with exact names: `HF_TOKEN`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
3. Ensure template references use `{{ RUNPOD_SECRET_<NAME> }}` syntax
4. Redeploy the endpoint

### "401 Unauthorized" or "Repository Not Found"

HuggingFace authentication failed. Fix:

1. Verify `HF_TOKEN` is set in RunPod secrets
2. Accept the license at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
3. Wait 5-10 minutes for license acceptance to propagate

### "Offline mode failed - models not found in cache"

Models are missing from cache. Fix:

1. Ensure network volume is attached in endpoint config
2. If first deployment, wait for first request to populate cache (8-15 min)
3. Check logs for download errors (network issues, HF token permissions)

### Models Re-downloaded Every Cold Start

Network volume not attached or offline mode not enabled. Fix:

1. Go to endpoint settings in RunPod Console
2. Ensure **Network Volume** is selected
3. Verify logs show `/runpod-volume` detected
4. Check for `[Startup] All models found in cache - enabling offline mode`

### Cold Start Takes 10+ Minutes Every Time

This should only happen ONCE (first worker). Subsequent starts should be 10-30 seconds.

If every cold start is slow:
1. **Network volume not attached**: Check endpoint config
2. **Cache permissions**: Ensure worker can write to `/runpod-volume`
3. **HF_HUB_OFFLINE not set**: Check logs for `HF_HUB_OFFLINE: 1`
4. **Model caching not enabled**: Add `microsoft/TRELLIS.2-4B` in endpoint's Model field

### First Cold Start Behavior

| Request | Expected Time | What Happens |
|---------|---------------|--------------|
| First ever | 8-15 min | Downloads DINOv3 + BiRefNet with file locking |
| Second+ | 10-30 sec | Loads all models from cache (offline mode) |
| Warm worker | < 1 sec | Model already in GPU memory |

### Optional: Pre-populate Cache Manually

To avoid the first-request download delay for auxiliary models (~3GB), you can pre-populate the network volume:

1. Deploy a **temporary Pod** with your network volume attached
2. SSH into the pod and run:
   ```bash
   cd /app
   python preload_models.py
   ```
3. Verify cache:
   ```bash
   ls -la /runpod-volume/huggingface-cache/hub/
   # Should show: models--facebook--dinov3-*, models--ZhengPeng7--BiRefNet
   ```
4. Stop the pod

This pre-downloads DINOv3 and BiRefNet before your first serverless request.

**Note:** The main `microsoft/TRELLIS.2-4B` model is handled by RunPod Model Caching (configured in Step 4) and doesn't need manual pre-population.

### R2 Upload Fails

1. Verify R2 credentials in RunPod secrets
2. Check R2 bucket exists in Cloudflare dashboard
3. Ensure bucket permissions allow write access

## Local Development

```bash
# Build and run locally (requires NVIDIA GPU with 48GB+ VRAM)
docker build -t trellis2-runpod .
docker run --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e R2_ENDPOINT_URL=$R2_ENDPOINT_URL \
  -e R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \
  -e R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
  -e R2_BUCKET_NAME=$R2_BUCKET_NAME \
  -v $HOME/.cache/huggingface:/runpod-volume/huggingface-cache \
  trellis2-runpod
```

## Tech Stack

- **Python**: 3.10 (required for prebuilt wheels)
- **PyTorch**: 2.6.0 with CUDA 12.4
- **flash-attn**: 2.7.3 (prebuilt wheel)
- **DRTK**: Custom build from GaXxO-dev/TRELLIS.2-commercial-use
- **Base Image**: `nvidia/cuda:12.4.1-devel-ubuntu22.04`

## License

MIT — matches TRELLIS.2's license.

## Acknowledgments

- [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) — Original model
- [GaXxO-dev/TRELLIS.2-commercial-use](https://github.com/GaXxO-dev/TRELLIS.2-commercial-use) — Commercial-use fork with MIT-licensed DRTK