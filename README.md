# TRELLIS.2 RunPod Serverless

RunPod serverless worker for [TRELLIS.2](https://github.com/GaXxO-dev/TRELLIS.2-commercial-use) — Microsoft's state-of-the-art 4B-parameter image-to-3D generative model. Converts a single image into a high-fidelity, PBR-textured 3D mesh (GLB/OBJ/PLY) with full material channels (base color, roughness, metallic, opacity).

Uses the [commercial-use fork](https://github.com/GaXxO-dev/TRELLIS.2-commercial-use) with MIT-licensed DRTK renderer (replaces the original NVidia-licensed nvdiffrast).

## Requirements

- **GPU**: NVIDIA A100 (80GB) or H100 (80GB) — minimum 48GB VRAM
- **RunPod**: Serverless endpoint with model caching enabled for `microsoft/TRELLIS.2-4B`
- **Cloudflare R2**: Bucket for storing generated 3D models (workaround for RunPod's 10/20MB response payload limit)
- **Hugging Face Token**: Required for the gated `dinov3-vitl16` model (accept terms at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | Hugging Face access token (read-only) — needed for gated dinov3 model |
| `R2_ENDPOINT_URL` | Yes | Cloudflare R2 S3-compatible endpoint (`https://<account>.r2.cloudflarestorage.com`) |
| `R2_ACCESS_KEY_ID` | Yes | R2 access key |
| `R2_SECRET_ACCESS_KEY` | Yes | R2 secret key |
| `R2_BUCKET_NAME` | Yes | R2 bucket name |
| `R2_PUBLIC_URL` | No | Public base URL for direct downloads (faster than presigned URLs) |
| `TRELLIS_PATH` | No | Path to TRELLIS.2 repo (default: `/app/TRELLIS.2`) |

## Quick Start

### 1. Build the Docker image

```bash
docker build -t trellis2-commercial-use-runpod .
```

The build installs PyTorch 2.6.0, flash-attn 2.7.3 (prebuilt wheel), DRTK (prebuilt wheel), pip dependencies from `requirements-inference.txt`, clones the commercial-use fork, and compiles CUDA extensions (o-voxel, CuMesh, FlexGEMM). Expect 25-30 minutes on first build.

### 2. Push to a registry

```bash
docker tag trellis2-commercial-use-runpod <your-registry>/trellis2-runpod:latest
docker push <your-registry>/trellis2-runpod:latest
```

### 3. Create a RunPod serverless endpoint

1. Set the **Container Image** to your pushed image
2. Under **Model**, enter `microsoft/TRELLIS.2-4B` to enable caching
3. Set **GPU Type** to A100-80GB or H100-80GB
4. Configure the R2 environment variables above
5. Deploy

## API Reference

### Request

```
POST https://api.runpod.ai/v2/<endpoint-id>/runsync
```

```json
{
  "input": {
    "image": "base64-encoded-png-or-url",
    "resolution": 1024,
    "texture_size": 2048,
    "output_format": "glb",
    "seed": 42,
    "sparse_structure_steps": 20,
    "slat_sampler_steps": 20,
    "max_num_tokens": 49152,
    "extension_webp": true
  }
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | string | **required** | Base64-encoded image (PNG/JPG) or HTTPS URL |
| `resolution` | int | `1024` | Voxel resolution: `512`, `1024`, or `1536` |
| `texture_size` | int | `2048` | Output texture resolution: `1024`, `2048`, or `4096` |
| `output_format` | string | `"glb"` | Output format: `"glb"`, `"obj"`, or `"ply"` |
| `seed` | int | random | Random seed for reproducibility |
| `sparse_structure_steps` | int | `20` | Denoising steps for sparse structure stage |
| `slat_sampler_steps` | int | `20` | Denoising steps for shape and texture SLAT stages |
| `max_num_tokens` | int | `49152` | Max sparse tokens during cascade upsampling |
| `extension_webp` | bool | `true` | Use WebP texture encoding in GLB (smaller files) |

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

### Performance Benchmarks (H100)

| Resolution | Total Time | Breakdown |
|---|---|---|
| 512³ | ~3s | 2s shape + 1s texture |
| 1024³ | ~17s | 10s shape + 7s texture |
| 1536³ | ~60s | 35s shape + 25s texture |

Times from [TRELLIS.2 README](https://github.com/microsoft/TRELLIS.2), plus ~2-5s for export + R2 upload.

## Architecture

```
Request → RunPod Worker → TRELLIS.2 (GPU) → o-voxel/DTRK → Temp File → R2 → URL
```

- Model loaded once at worker startup, kept in GPU memory (`low_vram=False`)
- GPU memory freed with `torch.cuda.empty_cache()` after each inference and export
- Output uploaded to R2, download URL returned — stays within RunPod payload limits

## License

MIT — matches TRELLIS.2's license.
