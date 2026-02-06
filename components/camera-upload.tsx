"use client";

import { useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ACCEPTED_IMAGE_TYPES, IMAGE_COMPRESSION_OPTIONS } from "@/lib/constants";
import type { MouthInfo } from "@/lib/types";

interface CameraUploadProps {
  onCapture: (image: File, mouthInfo?: MouthInfo) => void;
  onBack: () => void;
}

export default function CameraUpload({ onCapture, onBack }: CameraUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    // iOS Safari can send empty MIME type
    if (file.type && !ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      // Check extension as fallback
      const ext = file.name.toLowerCase().split(".").pop();
      if (!["jpg", "jpeg", "png", "webp"].includes(ext || "")) {
        setError("JPEG, PNG, WebP 형식의 이미지만 지원합니다.");
        return false;
      }
    }
    // Reject HEIC
    const ext = file.name.toLowerCase().split(".").pop();
    if (ext === "heic" || ext === "heif") {
      setError("HEIC 형식은 지원하지 않습니다. JPEG로 변환 후 업로드해주세요.");
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("10MB 이하의 이미지를 업로드해주세요.");
      return false;
    }
    return true;
  };

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setError(null);
      if (!validateFile(file)) return;

      // Show preview
      const url = URL.createObjectURL(file);
      setPreview(url);
      setSelectedFile(file);
    },
    []
  );

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setError(null);

    try {
      // Compress image
      let compressedFile = selectedFile;
      try {
        const imageCompression = (await import("browser-image-compression")).default;
        const compressed = await imageCompression(selectedFile, IMAGE_COMPRESSION_OPTIONS);
        compressedFile = new File([compressed], selectedFile.name, {
          type: compressed.type || "image/jpeg",
        });
      } catch {
        // Compression failed, use original
      }

      // Try MediaPipe face mesh for mouth landmarks
      let mouthInfo: MouthInfo | undefined;
      try {
        mouthInfo = await detectMouthLandmarks(compressedFile);
      } catch {
        // MediaPipe failed, continue without mouthInfo (graceful degradation)
      }

      onCapture(compressedFile, mouthInfo);
    } catch {
      setError("이미지 처리 중 오류가 발생했습니다. 다시 시도해주세요.");
      setIsProcessing(false);
    }
  }, [selectedFile, onCapture]);

  const handleRemove = () => {
    setPreview(null);
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <div className="max-w-md w-full space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <p className="text-sm text-[var(--muted-foreground)]">STEP 2 / 3</p>
          <h2 className="text-2xl font-bold">치아 사진 촬영</h2>
          <p className="text-sm text-[var(--muted-foreground)]">
            밝은 곳에서 입을 벌리고 치아가 잘 보이도록 촬영하세요
          </p>
        </div>

        {/* Guide */}
        <Card className="shadow-sm border-0 bg-white">
          <CardContent className="p-4 space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center text-xs text-[var(--muted-foreground)]">
              <div className="space-y-1">
                <div className="w-full aspect-square rounded-lg bg-[var(--secondary)] flex items-center justify-center text-2xl">
                  💡
                </div>
                <p>밝은 곳에서</p>
              </div>
              <div className="space-y-1">
                <div className="w-full aspect-square rounded-lg bg-[var(--secondary)] flex items-center justify-center text-2xl">
                  😁
                </div>
                <p>입을 벌리고</p>
              </div>
              <div className="space-y-1">
                <div className="w-full aspect-square rounded-lg bg-[var(--secondary)] flex items-center justify-center text-2xl">
                  📸
                </div>
                <p>정면에서 촬영</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Upload Area */}
        <Card className="shadow-sm border-0 bg-white overflow-hidden">
          <CardContent className="p-0">
            {preview ? (
              <div className="relative">
                <img
                  src={preview}
                  alt="촬영된 치아 사진"
                  className="w-full aspect-[4/3] object-cover"
                />
                <button
                  onClick={handleRemove}
                  className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/50 text-white flex items-center justify-center text-sm"
                >
                  ✕
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center w-full aspect-[4/3] bg-[var(--secondary)] gap-3 p-4">
                <div className="text-center space-y-3">
                  <div className="text-5xl">📷</div>
                  <div>
                    <p className="text-sm font-medium text-[var(--foreground)]">
                      사진 촬영 또는 파일 선택
                    </p>
                    <p className="text-xs text-[var(--muted-foreground)] mt-1">
                      JPEG, PNG, WebP (최대 10MB)
                    </p>
                  </div>
                </div>
                <div className="flex gap-3 w-full">
                  <label className="flex-1">
                    <input
                      ref={cameraInputRef}
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      capture="environment"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                    <div className="flex flex-col items-center justify-center h-24 rounded-lg bg-white hover:bg-gray-50 cursor-pointer transition-colors border border-gray-200">
                      <div className="text-3xl mb-1">📸</div>
                      <p className="text-xs font-medium">카메라로 촬영</p>
                    </div>
                  </label>
                  <label className="flex-1">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                    <div className="flex flex-col items-center justify-center h-24 rounded-lg bg-white hover:bg-gray-50 cursor-pointer transition-colors border border-gray-200">
                      <div className="text-3xl mb-1">🖼️</div>
                      <p className="text-xs font-medium">갤러리에서 선택</p>
                    </div>
                  </label>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Error */}
        {error && (
          <p className="text-sm text-center text-[var(--destructive)]">{error}</p>
        )}

        {/* Buttons */}
        <div className="flex gap-3">
          <Button
            type="button"
            variant="outline"
            onClick={onBack}
            className="flex-1 h-12 rounded-xl"
            disabled={isProcessing}
          >
            이전
          </Button>
          <Button
            onClick={handleAnalyze}
            disabled={!selectedFile || isProcessing}
            className="flex-[2] h-12 rounded-xl bg-[var(--foreground)] text-[var(--background)] hover:opacity-90 disabled:opacity-50"
          >
            {isProcessing ? "처리 중..." : "치아 분석 시작"}
          </Button>
        </div>
      </div>
    </div>
  );
}

/**
 * Detect mouth landmarks using MediaPipe Face Mesh.
 * Returns MouthInfo or throws if detection fails.
 */
async function detectMouthLandmarks(file: File): Promise<MouthInfo> {
  const { FaceMesh } = await import("@mediapipe/face_mesh");
  const faceMesh = new FaceMesh({
    locateFile: (f: string) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
  });

  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("MediaPipe timeout"));
    }, 10000);

    const img = new Image();
    img.onload = async () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);

      faceMesh.onResults((results) => {
        clearTimeout(timeout);

        if (!results.multiFaceLandmarks?.[0]) {
          reject(new Error("No face detected"));
          return;
        }

        const landmarks = results.multiFaceLandmarks[0];
        // Inner lip landmarks (indices for upper and lower inner lip)
        const lipIndices = [
          78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
          308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        ];

        const lipPoints: [number, number][] = lipIndices.map((i) => [
          landmarks[i].x * img.width,
          landmarks[i].y * img.height,
        ]);

        const xs = lipPoints.map((p) => p[0]);
        const ys = lipPoints.map((p) => p[1]);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

        resolve({
          centerX: (minX + maxX) / 2,
          centerY: (minY + maxY) / 2,
          width: maxX - minX,
          height: maxY - minY,
          upperY: minY,
          lowerY: maxY,
          lipPoints,
        });

        faceMesh.close();
      });

      await faceMesh.send({ image: canvas });
    };
    img.onerror = () => {
      clearTimeout(timeout);
      reject(new Error("Failed to load image"));
    };
    img.src = URL.createObjectURL(file);
  });
}
