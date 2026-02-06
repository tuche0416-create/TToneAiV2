export const AI_SERVER_URL = process.env.NEXT_PUBLIC_AI_SERVER_URL || 'http://localhost:8000';

export const POLLING_INTERVAL_MS = 2000;
export const SUBMIT_TIMEOUT_MS = 60000;  // 60초 - Lightning.ai Cold Start 대응
export const HEALTH_CHECK_TIMEOUT_MS = 30000;  // 30초 - Cold Start 대응
export const HEALTH_RETRY_INTERVAL_MS = 5000;  // 5초 간격으로 재시도
export const MAX_POLL_ATTEMPTS = 90; // 3 minutes max polling
export const MAX_HEALTH_RETRIES = 12; // 60초 max warmup (5초 x 12회)

export const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', ''];
export const MAX_IMAGE_SIZE_MB = 5;

export const IMAGE_COMPRESSION_OPTIONS = {
  maxSizeMB: 1,
  maxWidthOrHeight: 1920,
  useWebWorker: true,
};
