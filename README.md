# TRELLIS.2 RunPod Serverless

RunPod serverless worker for [TRELLIS.2](https://github.com/GaXxO-dev/TRELLIS.2-commercial-use) — Microsoft's state-of-the-art 4B-parameter image-to-3D generative model. Converts a single image into a high-fidelity, PBR-textured 3D mesh (GLB/OBJ/PLY) with full material channels (base color, roughness, metallic, opacity).

Uses the [commercial-use fork](https://github.com/GaXxO-dev/TRELLIS.2-commercial-use) with MIT-licensed DRTK renderer (replaces the original NVidia-licensed nvdiffrast).

## Requirements

- **GPU**: NVIDIA A100 (80GB), H100 (80GB), or equivalent — minimum 48GB VRAM
- **RunPod Account**: Serverless endpoint with network volume for model caching
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

TRELLIS.2 requires ~16GB of model files. A network volume persists downloads across cold starts.

1. Go to **Storage → Network Volumes** in RunPod Console
2. Click **New Network Volume**
3. Configure:
   - **Name**: `trellis2-models` (or your preference)
   - **Size**: `100` GB (minimum)
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
     - Fallback: `NVIDIA A100 80GB PCIe`, `NVIDIA A100-SXM4-80GB`, `NVIDIA RTX 6000 Ada`
   - **Network Volume**: Select the volume created in Step 1
   - **Workers**: Min `0`, Max `3` (adjust based on expected load)
   - **Idle Timeout**: `300` seconds (5 min)
   - **FlashBoot**: Enabled (faster cold starts)

3. Set **Environment Variables**:
   
   | Variable | Value |
   |----------|-------|
   | `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
   | `R2_ACCESS_KEY_ID` | `{{ RUNPOD_SECRET_R2_ACCESS_KEY_ID }}` |
   | `R2_SECRET_ACCESS_KEY` | `{{ RUNPOD_SECRET_R2_SECRET_ACCESS_KEY }}` |
   | `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
   | `R2_BUCKET_NAME` | `<your-bucket-name>` |
   | `R2_PUBLIC_URL` | *(optional)* `https://pub-xxx.r2.dev` |

4. Click **Deploy Endpoint**

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
    "idleTimeout": 300,
    "flashboot": true,
    "env": {
      "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
      "R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_R2_ACCESS_KEY_ID }}",
      "R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_R2_SECRET_ACCESS_KEY }}",
      "R2_ENDPOINT_URL": "https://<account-id>.r2.cloudflarestorage.com",
      "R2_BUCKET_NAME": "<your-bucket-name>"
    }
  }'
```

### Step 5: Verify Deployment

On first cold start, the worker logs should show:

```
============================================================
TRELLIS.2 RunPod Worker Initializing...
============================================================
[Startup] Working directory: /app
[Startup] Python version: 3.10.x
[Startup] CUDA available: True
[Startup] CUDA device: NVIDIA H100 80GB HBM3
[Startup] CUDA memory: 80.x GB
[Startup] HF_HOME: /runpod-volume/huggingface-cache
[Startup] HF_TOKEN: configured
[Startup] Pre-loading model (this may take several minutes on first run)...
```

**First cold start**: Downloads ~16GB of models (TRELLIS.2-4B + DINOv3 + BiRefNet) — expect 5-10 minutes depending on network.

**Subsequent cold starts**: Models load from network volume cache — expect 30-60 seconds.

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
│             │       ├── Model Cache: /runpod-volume/             │
│             │       │   └── microsoft/TRELLIS.2-4B              │
│             │       │   └── facebook/dinov3-vitl16 (gated)      │
│             │       │   └── ZhengPeng7/BiRefNet                 │
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
- **Model caching**: Network volume at `/runpod-volume/huggingface-cache/` persists ~16GB of models
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

### Models Re-downloaded Every Cold Start

Network volume not attached. Fix:

1. Go to endpoint settings in RunPod Console
2. Ensure **Network Volume** is selected
3. Verify `/runpod-volume` exists in worker logs

### Cold Start Takes 10+ Minutes

First cold start downloads models. Expected behavior. Subsequent starts should be 30-60s.

If subsequent cold starts are slow:
- Check network volume is attached (see above)
- Check `/runpod-volume/huggingface-cache/` exists in worker

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