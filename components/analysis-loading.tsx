"use client";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { AnalysisPhase } from "@/lib/types";

interface AnalysisLoadingProps {
  state: AnalysisPhase;
  onRetry: () => void;
}

const PROGRESS_MAP: Record<string, { label: string; value: number }> = {
  preprocessing: { label: "이미지 전처리 중...", value: 20 },
  inference: { label: "AI가 치아를 분석하고 있습니다...", value: 50 },
  postprocessing: { label: "결과를 정제하고 있습니다...", value: 75 },
  statistics: { label: "통계를 계산하고 있습니다...", value: 90 },
};

export default function AnalysisLoading({ state, onRetry }: AnalysisLoadingProps) {
  if (state.phase === "failed") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
        <div className="max-w-md w-full text-center space-y-6">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-red-50">
            <span className="text-4xl">⚠️</span>
          </div>
          <div className="space-y-2">
            <h2 className="text-xl font-bold">분석에 실패했습니다</h2>
            <p className="text-sm text-[var(--muted-foreground)]">
              {state.error}
            </p>
          </div>
          {state.canRetry && (
            <Button
              onClick={onRetry}
              className="h-12 px-8 rounded-xl bg-[var(--foreground)] text-[var(--background)] hover:opacity-90"
            >
              다시 시도하기
            </Button>
          )}
        </div>
      </div>
    );
  }

  const isWarming = state.phase === "warming";
  const isSubmitting = state.phase === "submitting";
  const isProcessing = state.phase === "processing";

  const progressInfo =
    isProcessing && state.progress
      ? PROGRESS_MAP[state.progress] || { label: "분석 중...", value: 40 }
      : null;

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <div className="max-w-md w-full text-center space-y-8">
        {/* Animated Icon */}
        <div className="relative inline-flex items-center justify-center">
          <div className="w-24 h-24 rounded-full border-4 border-[var(--border)] border-t-[var(--foreground)] animate-spin" />
          <span className="absolute text-3xl">🦷</span>
        </div>

        {/* Status Text */}
        <div className="space-y-3">
          {isWarming && (
            <>
              <h2 className="text-xl font-bold">AI 서버 시작 중...</h2>
              <p className="text-sm text-[var(--muted-foreground)]">
                첫 사용 시 서버 시작에 20-30초가 소요됩니다.
                <br />
                잠시만 기다려주세요.
              </p>
            </>
          )}
          {isSubmitting && (
            <>
              <h2 className="text-xl font-bold">이미지 업로드 중...</h2>
              <p className="text-sm text-[var(--muted-foreground)]">
                이미지를 AI 서버로 전송하고 있습니다.
              </p>
            </>
          )}
          {isProcessing && (
            <>
              <h2 className="text-xl font-bold">
                {progressInfo?.label || "분석 중..."}
              </h2>
              <p className="text-sm text-[var(--muted-foreground)]">
                AI가 치아 영역을 검출하고 색상을 분석합니다.
              </p>
            </>
          )}
        </div>

        {/* Progress Bar */}
        {isProcessing && progressInfo && (
          <div className="space-y-2">
            <Progress
              value={progressInfo.value}
              className="h-2 bg-[var(--secondary)]"
            />
            <p className="text-xs text-[var(--muted-foreground)]">
              {progressInfo.value}%
            </p>
          </div>
        )}

        {/* Warming Progress */}
        {isWarming && (
          <div className="space-y-2">
            <div className="flex justify-center gap-1">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="w-3 h-3 rounded-full bg-[var(--foreground)] animate-bounce"
                  style={{ animationDelay: `${i * 0.2}s` }}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
