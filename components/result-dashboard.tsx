"use client";

import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { AnalysisResult, UserInfo } from "@/lib/types";

interface ResultDashboardProps {
  result: AnalysisResult;
  userInfo: UserInfo;
  originalImage: File | null;
  onRetry: () => void;
}

export default function ResultDashboard({
  result,
  userInfo,
  originalImage,
  onRetry,
}: ResultDashboardProps) {
  const ageDiff = result.estimatedAge - userInfo.age;
  const ageComparison = useMemo(() => {
    if (ageDiff <= -3) return { label: "더 젊음", color: "text-emerald-600", bg: "bg-emerald-50", emoji: "✨" };
    if (ageDiff >= 3) return { label: "더 늙음", color: "text-amber-600", bg: "bg-amber-50", emoji: "⏳" };
    return { label: "나이 대비 적절", color: "text-blue-600", bg: "bg-blue-50", emoji: "👍" };
  }, [ageDiff]);

  const originalImageUrl = useMemo(() => {
    if (!originalImage) return null;
    return URL.createObjectURL(originalImage);
  }, [originalImage]);

  return (
    <div className="min-h-screen px-6 py-8">
      <div className="max-w-md w-full mx-auto space-y-5">
        {/* Header */}
        <div className="text-center space-y-1">
          <h2 className="text-2xl font-bold">진단 결과</h2>
          <p className="text-sm text-[var(--muted-foreground)]">
            AI가 분석한 치아 상태입니다
          </p>
        </div>

        {/* Tooth Age - Hero Card */}
        <Card className="shadow-sm border-0 bg-white overflow-hidden">
          <CardContent className="p-6 text-center space-y-3">
            <p className="text-sm text-[var(--muted-foreground)]">추정 치아 나이</p>
            <div className="flex items-baseline justify-center gap-1">
              <span className="text-6xl font-bold tracking-tight">
                {result.estimatedAge}
              </span>
              <span className="text-2xl text-[var(--muted-foreground)]">세</span>
            </div>
            <div
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${ageComparison.bg} ${ageComparison.color}`}
            >
              <span>{ageComparison.emoji}</span>
              <span>실제 나이({userInfo.age}세) 대비 {ageComparison.label}</span>
            </div>
          </CardContent>
        </Card>

        {/* WID Gauge */}
        <Card className="shadow-sm border-0 bg-white">
          <CardContent className="p-6 space-y-4">
            <div className="text-center">
              <p className="text-sm text-[var(--muted-foreground)] mb-4">
                WID (치아 미백 지수)
              </p>
              <WIDGauge wid={result.wid} />
              <p className="text-3xl font-bold mt-2">{result.wid.toFixed(1)}</p>
            </div>
            {/* Percentile */}
            <div className="flex items-center justify-center gap-2 pt-2">
              <div className="text-center px-4 py-2 rounded-xl bg-[var(--secondary)]">
                <p className="text-xs text-[var(--muted-foreground)]">백분위</p>
                <p className="text-lg font-bold">
                  상위 {result.percentile.toFixed(0)}%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Lab Values */}
        <Card className="shadow-sm border-0 bg-white">
          <CardContent className="p-4">
            <p className="text-sm text-[var(--muted-foreground)] mb-3">
              CIELab 색상 값
            </p>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="rounded-lg bg-[var(--secondary)] p-3">
                <p className="text-xs text-[var(--muted-foreground)]">L* (밝기)</p>
                <p className="text-lg font-semibold">
                  {result.labValues.l.toFixed(1)}
                </p>
              </div>
              <div className="rounded-lg bg-[var(--secondary)] p-3">
                <p className="text-xs text-[var(--muted-foreground)]">a* (적-녹)</p>
                <p className="text-lg font-semibold">
                  {result.labValues.a.toFixed(1)}
                </p>
              </div>
              <div className="rounded-lg bg-[var(--secondary)] p-3">
                <p className="text-xs text-[var(--muted-foreground)]">b* (황-청)</p>
                <p className="text-lg font-semibold">
                  {result.labValues.b.toFixed(1)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Visualization */}
        <Card className="shadow-sm border-0 bg-white overflow-hidden">
          <CardContent className="p-4 space-y-3">
            <p className="text-sm text-[var(--muted-foreground)]">
              AI 치아 영역 검출
            </p>
            <div className="grid grid-cols-2 gap-2">
              {originalImageUrl && (
                <div className="space-y-1">
                  <img
                    src={originalImageUrl}
                    alt="원본 이미지"
                    className="w-full aspect-square object-cover rounded-lg"
                  />
                  <p className="text-xs text-center text-[var(--muted-foreground)]">
                    원본
                  </p>
                </div>
              )}
              {result.visualization?.image && (
                <div className="space-y-1">
                  <img
                    src={result.visualization.image}
                    alt="AI 분석 결과"
                    className="w-full aspect-square object-cover rounded-lg"
                  />
                  <p className="text-xs text-center text-[var(--muted-foreground)]">
                    AI 검출 영역
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Quality Warnings */}
        {result.qualityWarnings.length > 0 && (
          <div className="bg-amber-50 rounded-xl p-3 text-center">
            <p className="text-sm text-amber-700">
              ⚠️{" "}
              {result.qualityWarnings
                .map((w) => {
                  if (w === "low_brightness") return "이미지가 다소 어둡습니다";
                  if (w === "blur_detected") return "이미지가 다소 흐릿합니다";
                  return w;
                })
                .join(", ")}
            </p>
          </div>
        )}

        {/* AI Metadata */}
        <div className="text-center text-xs text-[var(--muted-foreground)] space-y-1">
          <p>
            검출 치아: {result.aiMetadata.detectedTeethCount}개 · 처리 시간:{" "}
            {(result.aiMetadata.processingTimeMs / 1000).toFixed(1)}초
          </p>
          <p>신뢰도: {(result.aiMetadata.confidenceScore * 100).toFixed(0)}%</p>
        </div>

        {/* Retry Button */}
        <Button
          onClick={onRetry}
          variant="outline"
          className="w-full h-12 rounded-xl"
        >
          다시 진단하기
        </Button>

        <p className="text-xs text-center text-[var(--muted-foreground)] pb-4">
          본 서비스는 참고용이며 의료 진단을 대체하지 않습니다.
        </p>
      </div>
    </div>
  );
}

/** Semicircular WID gauge component (yellow → white gradient) */
function WIDGauge({ wid }: { wid: number }) {
  // WID typically ranges from ~0 (very yellow) to ~40 (very white)
  // Clamp to 0-40 for gauge display
  const normalized = Math.max(0, Math.min(1, wid / 40));
  const angle = -90 + normalized * 180; // -90 (left) to 90 (right)

  return (
    <div className="relative w-48 h-24 mx-auto">
      <svg viewBox="0 0 200 100" className="w-full h-full">
        {/* Background arc */}
        <defs>
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--gauge-yellow)" />
            <stop offset="100%" stopColor="var(--gauge-white)" />
          </linearGradient>
        </defs>
        <path
          d="M 10 95 A 90 90 0 0 1 190 95"
          fill="none"
          stroke="var(--border)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Colored arc */}
        <path
          d="M 10 95 A 90 90 0 0 1 190 95"
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${normalized * 283} 283`}
        />
        {/* Needle */}
        <line
          x1="100"
          y1="95"
          x2={100 + 70 * Math.cos((angle * Math.PI) / 180)}
          y2={95 + 70 * Math.sin((angle * Math.PI) / 180)}
          stroke="var(--foreground)"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
        <circle cx="100" cy="95" r="5" fill="var(--foreground)" />
        {/* Labels */}
        <text x="10" y="85" fontSize="10" fill="var(--muted-foreground)" textAnchor="middle">
          0
        </text>
        <text x="190" y="85" fontSize="10" fill="var(--muted-foreground)" textAnchor="middle">
          40
        </text>
      </svg>
    </div>
  );
}
