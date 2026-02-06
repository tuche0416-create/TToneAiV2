// User info collected in Step 2
export interface UserInfo {
  gender: 'male' | 'female';
  age: number;
}

// MediaPipe mouth landmarks from Step 3
export interface MouthInfo {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  upperY: number;
  lowerY: number;
  lipPoints: [number, number][];
}

// CIELab color values
export interface LabValues {
  l: number;
  a: number;
  b: number;
}

// Visualization data
export interface Visualization {
  image: string; // data:image/png;base64,...
}

// AI processing metadata
export interface AiMetadata {
  detectedTeethCount: number;
  processingTimeMs: number;
  centralIncisorsMaskPixels: number;
  toothNumbers: number[];
  confidenceScore: number;
}

// Complete analysis result from Lightning.ai
export interface AnalysisResult {
  wid: number;
  percentile: number;
  estimatedAge: number;
  labValues: LabValues;
  visualization: Visualization;
  aiMetadata: AiMetadata;
  qualityWarnings: string[];
}

// Job status response from Lightning.ai
export interface JobStatusResponse {
  status: 'processing' | 'completed' | 'failed';
  progress?: 'preprocessing' | 'inference' | 'postprocessing' | 'statistics';
  result?: AnalysisResult;
  error?: string;
  message?: string;
}

// Analysis flow state
export type AnalysisPhase =
  | { phase: 'idle' }
  | { phase: 'warming' }
  | { phase: 'submitting' }
  | { phase: 'processing'; progress?: string }
  | { phase: 'completed'; result: AnalysisResult }
  | { phase: 'failed'; error: string; canRetry: boolean };
