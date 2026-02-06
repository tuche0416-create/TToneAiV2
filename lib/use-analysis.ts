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

      await new Promise<void>(resolve => {
        const timer = setTimeout(resolve, HEALTH_RETRY_INTERVAL_MS);
        signal?.addEventListener('abort', () => {
          clearTimeout(timer);
          resolve();
        }, { once: true });
      });
    }
    return false;
  }, []);

  const startPolling = useCallback(async (jobId: string, signal?: AbortSignal): Promise<AnalysisResult> => {
    for (let i = 0; i < MAX_POLL_ATTEMPTS; i++) {
      if (signal?.aborted) throw new Error('Aborted');

      await new Promise<void>(resolve => {
        const timer = setTimeout(resolve, i === 0 ? 1000 : POLLING_INTERVAL_MS);
        signal?.addEventListener('abort', () => {
          clearTimeout(timer);
          resolve();
        }, { once: true });
      });

      if (signal?.aborted) throw new Error('Aborted');

      const status = await pollStatus(jobId, signal);

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
    const signal = abortRef.current.signal;

    try {
      setState({ phase: 'submitting' });

      let jobId: string;
      try {
        jobId = await submitAnalysis(params, signal);
      } catch (submitError) {
        if (signal.aborted) throw submitError;

        // Likely cold start - try warming up
        setState({ phase: 'warming' });
        const warmedUp = await waitForWarmup(signal);

        if (signal.aborted) return;

        if (!warmedUp) {
          setState({ phase: 'failed', error: '서비스에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.', canRetry: true });
          return;
        }
        // Retry submission after warmup
        setState({ phase: 'submitting' });
        jobId = await submitAnalysis(params, signal);
      }

      setState({ phase: 'processing' });
      const result = await startPolling(jobId, signal);
      if (signal.aborted) return;
      setState({ phase: 'completed', result });
    } catch (error) {
      if (signal.aborted || (error instanceof Error && error.message === 'Aborted')) {
        return;
      }

      const message = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      setState({ phase: 'failed', error: message, canRetry: true });
    }
  }, [waitForWarmup, startPolling]);

  return { state, analyze, reset };
}
