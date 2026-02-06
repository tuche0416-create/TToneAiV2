"use client";

import { useState, useCallback, useEffect } from "react";
import { useAnalysis } from "@/lib/use-analysis";
import type { UserInfo, MouthInfo } from "@/lib/types";
import type { SubmitParams } from "@/lib/ai-client";
import HeroSection from "@/components/hero-section";
import InfoForm from "@/components/info-form";
import CameraUpload from "@/components/camera-upload";
import AnalysisLoading from "@/components/analysis-loading";
import ResultDashboard from "@/components/result-dashboard";

type Step = 1 | 2 | 3 | 4 | 5;

export default function Home() {
  const [step, setStep] = useState<Step>(1);
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [capturedImage, setCapturedImage] = useState<File | null>(null);
  const { state: analysisState, analyze, reset: resetAnalysis } = useAnalysis();

  const handleStart = useCallback(() => {
    setStep(2);
  }, []);

  const handleInfoSubmit = useCallback((info: UserInfo) => {
    setUserInfo(info);
    setStep(3);
  }, []);

  const handleImageCapture = useCallback(
    async (image: File, mouthInfo?: MouthInfo) => {
      if (!userInfo) return;
      setCapturedImage(image);
      setStep(4);

      const params: SubmitParams = {
        image,
        gender: userInfo.gender,
        age: userInfo.age,
        mouthInfo,
      };

      await analyze(params);
    },
    [userInfo, analyze]
  );

  // Auto-transition to results when analysis completes
  useEffect(() => {
    if (step === 4 && analysisState.phase === "completed") {
      setStep(5);
    }
  }, [step, analysisState.phase]);

  const handleRetry = useCallback(() => {
    resetAnalysis();
    setUserInfo(null);
    setCapturedImage(null);
    setStep(1);
  }, [resetAnalysis]);

  const handleBack = useCallback(() => {
    if (step === 2) setStep(1);
    else if (step === 3) setStep(2);
  }, [step]);

  return (
    <main className="min-h-screen">
      {step === 1 && <HeroSection onStart={handleStart} />}
      {step === 2 && (
        <InfoForm onSubmit={handleInfoSubmit} onBack={handleBack} />
      )}
      {step === 3 && (
        <CameraUpload onCapture={handleImageCapture} onBack={handleBack} />
      )}
      {step === 4 && (
        <AnalysisLoading state={analysisState} onRetry={handleRetry} />
      )}
      {step === 5 && analysisState.phase === "completed" && (
        <ResultDashboard
          result={analysisState.result}
          userInfo={userInfo!}
          originalImage={capturedImage}
          onRetry={handleRetry}
        />
      )}
    </main>
  );
}
