import { AI_SERVER_URL, SUBMIT_TIMEOUT_MS, HEALTH_CHECK_TIMEOUT_MS } from './constants';
import type { JobStatusResponse, MouthInfo } from './types';

export interface SubmitParams {
  image: File;
  gender: 'male' | 'female';
  age: number;
  mouthInfo?: MouthInfo;
}

// Helper to combine external signal with timeout controller
function getCombinedSignal(controller: AbortController, externalSignal?: AbortSignal): AbortSignal {
  if (!externalSignal) return controller.signal;

  if (typeof AbortSignal.any === 'function') {
    return AbortSignal.any([externalSignal, controller.signal]);
  }

  // Fallback
  if (externalSignal.aborted) {
    controller.abort();
  } else {
    externalSignal.addEventListener('abort', () => controller.abort(), { once: true });
  }
  return controller.signal;
}

export async function checkHealth(externalSignal?: AbortSignal): Promise<boolean> {
  try {
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), HEALTH_CHECK_TIMEOUT_MS);

    const signal = getCombinedSignal(timeoutController, externalSignal);

    const res = await fetch(`${AI_SERVER_URL}/health`, { signal });
    clearTimeout(timeoutId);
    return res.ok;
  } catch {
    return false;
  }
}

export async function submitAnalysis(params: SubmitParams, externalSignal?: AbortSignal): Promise<string> {
  const formData = new FormData();
  formData.append('image', params.image);
  formData.append('gender', params.gender);
  formData.append('age', String(params.age));
  if (params.mouthInfo) {
    formData.append('mouth_info', JSON.stringify(params.mouthInfo));
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), SUBMIT_TIMEOUT_MS);

  const signal = getCombinedSignal(controller, externalSignal);

  try {
    const res = await fetch(`${AI_SERVER_URL}/analyze`, {
      method: 'POST',
      body: formData,
      signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: 'Analysis submission failed' }));
      throw new Error(err.message || `HTTP ${res.status}`);
    }

    const { job_id } = await res.json();
    return job_id;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

export async function pollStatus(jobId: string, externalSignal?: AbortSignal): Promise<JobStatusResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);

  const signal = getCombinedSignal(controller, externalSignal);

  try {
    const res = await fetch(`${AI_SERVER_URL}/status/${jobId}`, {
      signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      if (res.status === 404) {
        throw new Error('Job not found or expired');
      }
      throw new Error('Status check failed');
    }
    return res.json();
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}
