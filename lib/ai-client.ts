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
