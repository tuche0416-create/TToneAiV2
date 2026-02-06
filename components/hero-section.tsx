"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { checkHealth } from "@/lib/ai-client";

interface HeroSectionProps {
  onStart: () => void;
}

export default function HeroSection({ onStart }: HeroSectionProps) {
  // Pre-warm Lightning.ai on mount (fire-and-forget)
  useEffect(() => {
    checkHealth().catch(() => {});
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
