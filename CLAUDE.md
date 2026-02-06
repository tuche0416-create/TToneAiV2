# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

T-Tone AI V2 — a dental whiteness diagnosis web service. Users upload a tooth photo, and the system calculates a **WID (Whiteness Index for Dentistry)** score, percentile ranking, and estimated tooth age via AI-based tooth segmentation.

## Tech Stack

- **Frontend**: Next.js 16 (App Router) + React 19 + TypeScript + Tailwind CSS v4 + Shadcn UI
- **AI Server**: Lightning.ai (FastAPI + PyTorch) — **ALL computation here**
- **Client-side**: MediaPipe Face Mesh (mouth landmarks), browser-image-compression
- **Deploy**: Vercel (frontend static hosting only), Lightning.ai (AI server)
- **Design Theme**: 2026 Pantone Cloud Dancer

## Architecture Principle

> **Lightning.ai handles ALL computation. Vercel serves frontend only. No Vercel API Routes.**
> Client communicates directly with Lightning.ai via async polling.

This completely bypasses the Vercel 10-second timeout limitation.

## Commands

```bash
npm run dev          # Start dev server
npm run build        # Production build
npm run lint         # ESLint
```

Lightning.ai server (Python):
```bash
cd lightning-server
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Data Flow

```
Client: Image + UserInfo + MouthInfo (MediaPipe)
  → POST Lightning.ai/analyze → {job_id} (instant, <200ms)
  → GET Lightning.ai/status/{job_id} (poll every 2s)
  → Lightning.ai (background): EXIF → CLAHE → FPN inference → Mask → RGB → Lab → WID → Stats → Visualization
  → Response: {status: "completed", result: AnalysisResult}
```

## Key Directories

- `app/page.tsx` — Main orchestrator (5-step flow state machine)
- `components/` — UI components per step + Shadcn `ui/` primitives
- `lib/ai-client.ts` — Lightning.ai API client (submit, poll, health)
- `lib/use-analysis.ts` — React hook for analysis flow (polling + cold start)
- `lightning-server/app/` — Python FastAPI server (inference, color science, statistics)

**No `app/api/` directory** — Vercel has zero API routes.

## CRITICAL: Protected Algorithms (NEVER modify logic)

These algorithms are implemented in Python on Lightning.ai (`lightning-server/app/`):
- `color_science.py` — sRGB→Linear→XYZ→CIELab (D65: {x:95.047, y:100.0, z:108.883})
- `statistics.py` — WID formula, percentile (normal CDF), tooth age estimation
- `data.csv` — Research dataset (Lab means/SDs per gender/age group)

**WID**: `0.511 × L* + (-2.324) × a* + (-1.100) × b*`
**WID Stats (Error Propagation)**: `WID_sd = sqrt((0.511×L_sd)² + (2.324×a_sd)² + (1.100×b_sd)²)`

## Lightning.ai API Contract

- `GET /health` — Server status + pre-warm
- `POST /analyze` — Submit job (image + gender + age + mouthInfo) → `{job_id}`
- `GET /status/{job_id}` — Poll result → `{status, progress?, result?, error?}`

## Environment Variables

- `NEXT_PUBLIC_AI_SERVER_URL` — Lightning.ai endpoint (client-side, embedded in bundle)

## Cold Start Handling (3-Layer)

1. **Pre-warm**: Hero/InfoForm mount fires `/health` (fire-and-forget)
2. **Submit timeout**: 15s timeout → "AI server starting..." UI
3. **Auto-retry**: `/health` check every 3s → auto-resubmit when warm

## Design Conventions

- Korean language primary
- Tooth region visualization: **green overlay** on detected teeth (critical UX)
- Result gauge: semicircle gradient (yellow → white)
- iOS Safari compatibility (empty MIME type handling, HEIC rejection)
- 2026 Pantone Cloud Dancer: Primary #F0EBE3, Text #2D2D2D, Success #86EFAC
