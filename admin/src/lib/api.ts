import type { AdminConfig, ApplyResult, LocalProviderResult, TestResult, ValidateResult } from './config';

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

export function loadConfig(): Promise<AdminConfig> {
  return api<AdminConfig>('/admin/api/config');
}

export function validateConfig(values: Record<string, string>): Promise<ValidateResult> {
  return api<ValidateResult>('/admin/api/config/validate', {
    method: 'POST',
    body: JSON.stringify({ values }),
  });
}

export function applyConfig(values: Record<string, string>): Promise<ApplyResult> {
  return api<ApplyResult>('/admin/api/config/apply', {
    method: 'POST',
    body: JSON.stringify({ values }),
  });
}

export function loadLocalStatus(): Promise<{ providers: LocalProviderResult[] }> {
  return api<{ providers: LocalProviderResult[] }>('/admin/api/providers/local-status');
}

export function testProvider(providerId: string): Promise<TestResult> {
  return api<TestResult>(`/admin/api/providers/${providerId}/test`, {
    method: 'POST',
    body: '{}',
  });
}

export function loadStatus(): Promise<Record<string, unknown>> {
  return api<Record<string, unknown>>('/admin/api/status');
}
