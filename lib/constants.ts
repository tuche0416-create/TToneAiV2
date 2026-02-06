export const AI_SERVER_URL = process.env.NEXT_PUBLIC_AI_SERVER_URL || 'http://localhost:8000';

export const POLLING_INTERVAL_MS = 2000;
export const SUBMIT_TIMEOUT_MS = 15000;
export const HEALTH_CHECK_TIMEOUT_MS = 5000;
export const HEALTH_RETRY_INTERVAL_MS = 3000;
export const MAX_POLL_ATTEMPTS = 60; // 2 minutes max polling
export const MAX_HEALTH_RETRIES = 10; // 30 seconds max warmup

export const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', ''];
export const MAX_IMAGE_SIZE_MB = 5;

export const IMAGE_COMPRESSION_OPTIONS = {
  maxSizeMB: 1,
  maxWidthOrHeight: 1920,
  useWebWorker: true,
};
