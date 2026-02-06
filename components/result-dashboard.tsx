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
    if (ageDiff <= -3) return { label: "더 젊어 보여요", color: "text-emerald-700", bg: "bg-emerald-50", emoji: "✨" };
    if (ageDiff >= 3) return { label: "관리가 필요해요", color: "text-amber-700", bg: "bg-amber-50", emoji: "💪" };
    return { label: "나이와 비슷해요", color: "text-slate-700", bg: "bg-slate-100", emoji: "👍" };
  }, [ageDiff]);

  return (
    <div className="min-h-screen bg-[#F0EBE3] px-6 py-12 flex items-center justify-center">
      <div className="max-w-md w-full mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2 mb-8">
          <h2 className="text-3xl font-bold tracking-tight text-slate-900">진단 리포트</h2>
          <p className="text-sm text-slate-500 font-medium">
            AI가 분석한 당신의 치아 색상입니다
          </p>
        </div>

        {/* Tooth Age - Hero Card */}
        <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm rounded-3xl overflow-hidden ring-1 ring-slate-900/5">
          <CardContent className="p-8 text-center space-y-6">
            <div className="space-y-2">
              <p className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Estimated Tooth Age</p>
              <div className="flex items-baseline justify-center gap-1.5 status-text-animation">
                <span className="text-7xl font-bold tracking-tighter text-slate-900">
                  {result.estimatedAge}
                </span>
                <span className="text-2xl text-slate-400 font-medium">세</span>
              </div>
            </div>

            <div
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold transition-colors ${ageComparison.bg} ${ageComparison.color}`}
            >
              <span className="text-lg">{ageComparison.emoji}</span>
              <span>실제 나이 대비 {ageComparison.label}</span>
            </div>
          </CardContent>
        </Card>

        {/* AI Visualization (Only Result) */}
        {result.visualization?.image && (
          <div className="flex justify-center relative z-10">
            <div className="relative w-full aspect-[4/3] max-w-[280px] rounded-2xl overflow-hidden shadow-lg border-4 border-white/50">
              <img
                src={result.visualization.image}
                alt="AI Analysis"
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-2 right-2 bg-black/50 text-white text-[10px] px-2 py-1 rounded-full backdrop-blur-sm">
                AI 분석 결과
              </div>
            </div>
          </div>
        )}

        {/* Analysis Details Grid */}
        <div className="grid gap-4">
          {/* WID & Percentile */}
          <Card className="shadow-md border-0 bg-white rounded-2xl p-6">
            <div className="text-center space-y-4">
              <div className="space-y-1">
                <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                  WID Analysis
                </p>
                <p className="text-slate-900 font-semibold">치아 색상 분석 지수</p>
              </div>

              <div className="relative py-2">
                <WIDGauge wid={result.wid} />
                <div className="absolute inset-x-0 bottom-0 text-center translate-y-2">
                  <p className="text-4xl font-bold text-slate-900 tracking-tight">{result.wid.toFixed(1)}</p>
                </div>
              </div>

              <div className="pt-4">
                <div className="bg-slate-50 rounded-xl py-3 px-4 inline-block w-full">
                  <p className="text-xs text-slate-500 mb-1">Total Whiteness Rank</p>
                  <p className="text-xl font-bold text-slate-900">
                    상위 {result.percentile.toFixed(0)}%
                  </p>
                  <p className="text-[10px] text-slate-400 mt-1">
                    (0%에 가까울수록 가장 밝은 치아입니다)
                  </p>
                </div>
              </div>
            </div>
          </Card>

          {/* Lab Values */}
          <Card className="shadow-md border-0 bg-white rounded-2xl overflow-hidden">
            <div className="p-5 bg-slate-50 border-b border-slate-100">
              <p className="text-sm font-semibold text-slate-600">CIELab 상세 분석</p>
            </div>
            <div className="grid grid-cols-3 divide-x divide-slate-100">
              <div className="p-4 text-center">
                <p className="text-xs text-slate-400 font-medium mb-1">L* (명도)</p>
                <p className="text-lg font-bold text-slate-900">{result.labValues.l.toFixed(1)}</p>
              </div>
              <div className="p-4 text-center">
                <p className="text-xs text-slate-400 font-medium mb-1">a* (적색도)</p>
                <p className="text-lg font-bold text-slate-900">{result.labValues.a.toFixed(1)}</p>
              </div>
              <div className="p-4 text-center">
                <p className="text-xs text-slate-400 font-medium mb-1">b* (황색도)</p>
                <p className="text-lg font-bold text-slate-900">{result.labValues.b.toFixed(1)}</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Footer Info */}
        <div className="pt-4 text-center space-y-2">
          <div className="flex justify-center gap-4 text-[10px] text-slate-400 uppercase tracking-widest font-semibold">
            <span>AI Confidence {(result.aiMetadata.confidenceScore * 100).toFixed(0)}%</span>
            <span>•</span>
            <span>Process Time {(result.aiMetadata.processingTimeMs / 1000).toFixed(1)}s</span>
          </div>

          {/* Quality Warnings */}
          {result.qualityWarnings.length > 0 && (
            <div className="inline-block bg-amber-50 text-amber-600 text-xs px-3 py-1 rounded-full">
              ⚠️ {result.qualityWarnings.join(", ")}
            </div>
          )}

          <Button
            onClick={onRetry}
            size="lg"
            className="w-full h-14 text-lg rounded-2xl bg-slate-900 hover:bg-slate-800 text-white shadow-xl shadow-slate-900/10 mt-4 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            다시 진단하기
          </Button>

          <p className="text-xs text-slate-400 pt-4 pb-8">
            의료적 진단이 아닌 AI 분석 결과입니다.
          </p>
        </div>
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
