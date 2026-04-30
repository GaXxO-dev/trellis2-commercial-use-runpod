# AGENTS.md — TRELLIS.2 RunPod Serverless

## Architecture

Single-file RunPod serverless worker (`handler.py`) wrapping [GaXxO-dev/TRELLIS.2-commercial-use](https://github.com/GaXxO-dev/TRELLIS.2-commercial-use) (MIT-licensed fork with DRTK renderer replacing nvidia's nvdiffrast).

Flow: `Request → handler.py → Trellis2ImageTo3DPipeline (GPU) → o-voxel/DRTK → temp file → Cloudflare R2 → URL`

## Local build

```bash
docker build -t trellis2-commercial-use-runpod .
docker push gaxxo/trellis2-commercial-use-runpod:latest
```

CI auto-builds on push to `main` (`.github/workflows/build.yml`). Requires GitHub secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.

## Tech stack constraints (easy to miss)

- **Python 3.10** only — the fork depends on prebuilt wheels (`cp310`) for flash-attn and DRTK. Do NOT upgrade to 3.11+.
- **Base image**: `nvidia/cuda:12.4.1-devel-ubuntu22.04` (need `nvcc` for CuMesh/FlexGEMM/o-voxel compilation).

## Model loading (critical gotcha)

Always pass the HF repo ID string to `Trellis2ImageTo3DPipeline.from_pretrained()`:

```python
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
```

Do **NOT** resolve the local cache path and pass that. TRELLIS.2's internal `models.from_pretrained()` splits the path as `org/repo/subpath` — passing `/runpod-volume/...` breaks with a `HFValidationError`.

RunPod's model caching + `huggingface_hub` auto-resolve from `$HF_HOME/hub/` when given the repo ID. No manual cache-path logic needed.

## Required environment variables

| Variable | Type | Why |
|---|---|---|
| `HF_TOKEN` | Secret | Gated `facebook/dinov3-vitl16-pretrain-lvd1689m` used internally |
| `R2_ENDPOINT_URL` | Env | Cloudflare R2 S3 endpoint |
| `R2_ACCESS_KEY_ID` | Secret | R2 auth |
| `R2_SECRET_ACCESS_KEY` | Secret | R2 auth |
| `R2_BUCKET_NAME` | Env | R2 bucket name |

## RunPod endpoint setup

- GPU priority: H100 (80GB) first, then A100 (80GB)
- Enable model caching for `microsoft/TRELLIS.2-4B` in endpoint config (Model field)
- R2 needed because RunPod `/runsync` payload limit is 20MB — generated GLBs are larger

## Prebuilt wheels (do not build from source)

flash-attn 2.7.3 and DRTK are installed as prebuilt wheels from release URLs. Building from source is slower and error-prone. Wheel URLs are in `ENVIRONMENT.md` on the fork repo and hardcoded in the Dockerfile.

## Easy-to-miss runtime dependencies

`kornia`, `timm`, and `psutil` are NOT in `requirements-inference.txt` but are required at runtime by TRELLIS.2's BiRefNet background removal model (`transformers.AutoModelForImageSegmentation` loads them dynamically). They are installed in a dedicated Dockerfile RUN step.
