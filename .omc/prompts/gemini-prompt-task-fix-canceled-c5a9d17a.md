---
provider: "gemini"
agent_role: "designer"
model: "gemini-3-pro-preview"
files:
  - "lib/ai-client.ts"
  - "lib/use-analysis.ts"
  - "components/hero-section.tsx"
  - "lib/constants.ts"
  - "lib/types.ts"
timestamp: "2026-02-06T20:19:03.113Z"
---

--- File: lib/ai-client.ts ---
import { AI_SERVER_URL, SUBMIT_TIMEOUT_MS, HEALTH_CHECK_TIMEOUT_MS } from './constants';
import type { JobStatusResponse, MouthInfo } from './types';

export interface SubmitParams {
  image: File;
  gender: 'male' | 'female';
  age: number;
  mouthInfo?: MouthInfo;
}

export async function checkHealth(externalSignal?: AbortSignal): Promise<boolean> {
  try {
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), HEALTH_CHECK_TIMEOUT_MS);

    // Combine external signal (component lifecycle) with internal timeout
    let signal: AbortSignal = timeoutController.signal;
    if (externalSignal) {
      if (typeof AbortSignal.any === 'function') {
        signal = AbortSignal.any([externalSignal, timeoutController.signal]);
      } else {
        // Fallback: forward external abort to timeout controller
        if (externalSignal.aborted) {
          clearTimeout(timeoutId);
          return false;
        }
        externalSignal.addEventListener('abort', () => timeoutController.abort(), { once: true });
      }
    }

    const res = await fetch(`${AI_SERVER_URL}/health`, { signal });
    clearTimeout(timeoutId);
    return res.ok;
  } catch {
    return false;
  }
}

export async function submitAnalysis(params: SubmitParams): Promise<string> {
  const formData = new FormData();
  formData.append('image', params.image);
  formData.append('gender', params.gender);
  formData.append('age', String(params.age));
  if (params.mouthInfo) {
    formData.append('mouth_info', JSON.stringify(params.mouthInfo));
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), SUBMIT_TIMEOUT_MS);

  const res = await fetch(`${AI_SERVER_URL}/analyze`, {
    method: 'POST',
    body: formData,
    signal: controller.signal,
  });
  clearTimeout(timeoutId);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: 'Analysis submission failed' }));
    throw new Error(err.message || `HTTP ${res.status}`);
  }

  const { job_id } = await res.json();
  return job_id;
}

export async function pollStatus(jobId: string): Promise<JobStatusResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);

  const res = await fetch(`${AI_SERVER_URL}/status/${jobId}`, {
    signal: controller.signal,
  });
  clearTimeout(timeoutId);

  if (!res.ok) {
    if (res.status === 404) {
      throw new Error('Job not found or expired');
    }
    throw new Error('Status check failed');
  }
  return res.json();
}


--- File: lib/use-analysis.ts ---
'use client';

import { useState, useCallback, useRef } from 'react';
import { checkHealth, submitAnalysis, pollStatus, type SubmitParams } from './ai-client';
import { POLLING_INTERVAL_MS, HEALTH_RETRY_INTERVAL_MS, MAX_POLL_ATTEMPTS, MAX_HEALTH_RETRIES } from './constants';
import type { AnalysisPhase, AnalysisResult } from './types';

export function useAnalysis() {
  const [state, setState] = useState<AnalysisPhase>({ phase: 'idle' });
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState({ phase: 'idle' });
  }, []);

  const waitForWarmup = useCallback(async (signal?: AbortSignal): Promise<boolean> => {
    for (let i = 0; i < MAX_HEALTH_RETRIES; i++) {
      if (signal?.aborted) return false;
      const healthy = await checkHealth(signal);
      if (healthy) return true;
      if (signal?.aborted) return false;
      await new Promise(r => setTimeout(r, HEALTH_RETRY_INTERVAL_MS));
    }
    return false;
  }, []);

  const startPolling = useCallback(async (jobId: string): Promise<AnalysisResult> => {
    for (let i = 0; i < MAX_POLL_ATTEMPTS; i++) {
      await new Promise(r => setTimeout(r, i === 0 ? 1000 : POLLING_INTERVAL_MS));

      const status = await pollStatus(jobId);

      if (status.status === 'completed' && status.result) {
        return status.result;
      }

      if (status.status === 'failed') {
        throw new Error(status.message || status.error || 'Analysis failed');
      }

      setState({ phase: 'processing', progress: status.progress });
    }

    throw new Error('Analysis timed out');
  }, []);

  const analyze = useCallback(async (params: SubmitParams) => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    try {
      setState({ phase: 'submitting' });

      let jobId: string;
      try {
        jobId = await submitAnalysis(params);
      } catch (submitError) {
        // Likely cold start - try warming up
        setState({ phase: 'warming' });
        const warmedUp = await waitForWarmup(abortRef.current?.signal);
        if (!warmedUp) {
          setState({ phase: 'failed', error: '서비스에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.', canRetry: true });
          return;
        }
        // Retry submission after warmup
        setState({ phase: 'submitting' });
        jobId = await submitAnalysis(params);
      }

      setState({ phase: 'processing' });
      const result = await startPolling(jobId);
      setState({ phase: 'completed', result });
    } catch (error) {
      const message = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      setState({ phase: 'failed', error: message, canRetry: true });
    }
  }, [waitForWarmup, startPolling]);

  return { state, analyze, reset };
}


--- File: components/hero-section.tsx ---
"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { checkHealth } from "@/lib/ai-client";

interface HeroSectionProps {
  onStart: () => void;
}

export default function HeroSection({ onStart }: HeroSectionProps) {
  // Pre-warm Lightning.ai on mount with proper cleanup
  useEffect(() => {
    const controller = new AbortController();
    checkHealth(controller.signal).catch(() => {});
    return () => controller.abort();
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <div className="max-w-md w-full text-center space-y-8">
        {/* Logo & Brand */}
        <div className="space-y-3">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-white shadow-sm">
            <span className="text-3xl">🦷</span>
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-[var(--foreground)]">
            T-Tone AI
          </h1>
          <p className="text-lg text-[var(--muted-foreground)]">
            AI 기반 치아 미백 진단
          </p>
        </div>

        {/* Description */}
        <div className="space-y-4 text-[var(--muted-foreground)]">
          <p className="text-sm leading-relaxed">
            스마트폰 카메라로 치아 사진을 촬영하면
            <br />
            <span className="font-medium text-[var(--foreground)]">
              WID(치아 미백 지수)
            </span>
            를 계산하고
            <br />
            나이 대비 치아 상태를 진단합니다.
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-3 gap-3 text-center">
          {[
            { icon: "📸", label: "간편 촬영" },
            { icon: "🤖", label: "AI 분석" },
            { icon: "📊", label: "상세 리포트" },
          ].map((feat) => (
            <div
              key={feat.label}
              className="bg-white rounded-xl p-3 shadow-sm"
            >
              <div className="text-2xl mb-1">{feat.icon}</div>
              <div className="text-xs font-medium text-[var(--muted-foreground)]">
                {feat.label}
              </div>
            </div>
          ))}
        </div>

        {/* CTA Button */}
        <Button
          onClick={onStart}
          size="lg"
          className="w-full h-14 text-lg font-semibold rounded-xl bg-[var(--foreground)] text-[var(--background)] hover:opacity-90 transition-opacity"
        >
          진단 시작
        </Button>

        <p className="text-xs text-[var(--muted-foreground)]">
          소요 시간: 약 30초 · 무료 · 개인정보 저장 없음
        </p>
      </div>
    </div>
  );
}


--- File: lib/constants.ts ---
export const AI_SERVER_URL = process.env.NEXT_PUBLIC_AI_SERVER_URL || 'http://localhost:8000';

export const POLLING_INTERVAL_MS = 2000;
export const SUBMIT_TIMEOUT_MS = 15000;
export const HEALTH_CHECK_TIMEOUT_MS = 5000;
export const HEALTH_RETRY_INTERVAL_MS = 3000;
export const MAX_POLL_ATTEMPTS = 60; // 2 minutes max polling
export const MAX_HEALTH_RETRIES = 10; // 30 seconds max warmup

export const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', ''];
export const MAX_IMAGE_SIZE_MB = 5;

export const IMAGE_COMPRESSION_OPTIONS = {
  maxSizeMB: 1,
  maxWidthOrHeight: 1920,
  useWebWorker: true,
};


--- File: lib/types.ts ---
// User info collected in Step 2
export interface UserInfo {
  gender: 'male' | 'female';
  age: number;
}

// MediaPipe mouth landmarks from Step 3
export interface MouthInfo {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  upperY: number;
  lowerY: number;
  lipPoints: [number, number][];
}

// CIELab color values
export interface LabValues {
  l: number;
  a: number;
  b: number;
}

// Visualization data
export interface Visualization {
  image: string; // data:image/png;base64,...
}

// AI processing metadata
export interface AiMetadata {
  detectedTeethCount: number;
  processingTimeMs: number;
  centralIncisorsMaskPixels: number;
  toothNumbers: number[];
  confidenceScore: number;
}

// Complete analysis result from Lightning.ai
export interface AnalysisResult {
  wid: number;
  percentile: number;
  estimatedAge: number;
  labValues: LabValues;
  visualization: Visualization;
  aiMetadata: AiMetadata;
  qualityWarnings: string[];
}

// Job status response from Lightning.ai
export interface JobStatusResponse {
  status: 'processing' | 'completed' | 'failed';
  progress?: 'preprocessing' | 'inference' | 'postprocessing' | 'statistics';
  result?: AnalysisResult;
  error?: string;
  message?: string;
}

// Analysis flow state
export type AnalysisPhase =
  | { phase: 'idle' }
  | { phase: 'warming' }
  | { phase: 'submitting' }
  | { phase: 'processing'; progress?: string }
  | { phase: 'completed'; result: AnalysisResult }
  | { phase: 'failed'; error: string; canRetry: boolean };


IMPORTANT: Write your complete response to the file: /Users/tanpapa/Desktop/develop-b/TToneAiV2/.tmp-gemini-output.md

# Task: Fix (canceled) fetch requests in T-Tone AI V2

## Problem
All `/health` and `/analyze` fetch requests from browser to Lightning.ai server are showing as `(canceled)` in Chrome DevTools Network tab.

## Root Causes Identified

1. **`submitAnalysis()` and `pollStatus()` don't accept external AbortSignal** - They create internal AbortControllers for timeout only, but the lifecycle AbortController from `useAnalysis` hook is never threaded through.

2. **HeroSection pre-warm health check aborts on unmount** - The useEffect cleanup `controller.abort()` fires immediately when user clicks "start" (step 1->2), killing the pre-warm request.

3. **`analyze()` unconditionally aborts previous controller** - `abortRef.current?.abort()` at the start can abort in-flight requests during retry flows.

4. **`pollStatus()` creates new AbortController per call without lifecycle integration** - Each poll creates an isolated controller.

## Files to Fix

### 1. `lib/ai-client.ts` - Add external signal support to ALL fetch functions

Current code is in the file. Required changes:
- `submitAnalysis` must accept an optional `externalSignal?: AbortSignal` parameter
- `pollStatus` must accept an optional `externalSignal?: AbortSignal` parameter
- Both must combine the external signal with their internal timeout signal using the same `AbortSignal.any` pattern that `checkHealth` already uses (with fallback for browsers without `AbortSignal.any`)
- Helper function `combineSignals(timeoutController, timeoutId, externalSignal)` to DRY the signal combination logic

### 2. `lib/use-analysis.ts` - Thread abort signal through all calls

Required changes:
- `startPolling` must accept an `AbortSignal` and pass it to each `pollStatus` call. Also check `signal.aborted` before each poll iteration.
- `analyze` must pass `abortRef.current.signal` to `submitAnalysis` calls and to `startPolling`
- When catching errors, distinguish abort errors (ignore silently - don't set failed state) from real errors (show to user). Check `signal.aborted` on the abortRef.
- The `waitForWarmup` delay should also be cancellable via signal.

### 3. `components/hero-section.tsx` - Fire-and-forget pre-warm (no abort on unmount)

Required change:
- The pre-warm health check should be truly fire-and-forget. Don't abort on unmount because the purpose is to warm up the Lightning.ai server. Aborting defeats the purpose.
- Remove the AbortController and cleanup. Just call `checkHealth()` without signal.
- The useEffect should still have the empty dependency array `[]`.

## Output Format

Output the COMPLETE fixed file contents for each of the 3 files, clearly labeled with the file path. Do not omit any existing code - output the full file. Preserve all existing imports, types, and JSX.

Important constraints:
- Keep `AbortSignal.any` fallback for browser compatibility (Safari <17 doesn't support it)
- Don't change function export signatures in a breaking way (add optional params only)
- Don't modify constants, types, or any other files
- Don't add console.log or debugging code
- Keep the code minimal and clean - no over-engineering
