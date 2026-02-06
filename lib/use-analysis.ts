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

  const waitForWarmup = useCallback(async (): Promise<boolean> => {
    for (let i = 0; i < MAX_HEALTH_RETRIES; i++) {
      const healthy = await checkHealth();
      if (healthy) return true;
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
        const warmedUp = await waitForWarmup();
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
