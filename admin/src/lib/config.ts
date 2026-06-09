export const MASKED_SECRET = '********';

export interface Field {
  key: string;
  label: string;
  description: string;
  section: string;
  type: string;
  value: string;
  source: string;
  locked: boolean;
  advanced: boolean;
  secret: boolean;
  configured: boolean;
  options?: string[];
}

export interface Section {
  id: string;
  label: string;
  description: string;
}

export interface ProviderStatus {
  provider_id: string;
  status: string;
  label: string;
  kind: string;
  base_url: string;
  credential_env: string;
}

export interface AdminConfig {
  fields: Field[];
  sections: Section[];
  provider_status: ProviderStatus[];
  paths: { managed: string };
}

export interface LocalProviderResult {
  provider_id: string;
  status: string;
  label: string;
  base_url: string;
  status_code?: number;
  error_type?: string;
}

export interface TestResult {
  provider_id: string;
  ok: boolean;
  models?: string[];
  error_type?: string;
}

export interface ValidateResult {
  valid: boolean;
  errors?: string[];
}

export interface ApplyResult {
  applied: boolean;
  errors?: string[];
  pending_fields?: string[];
  restart?: {
    required: boolean;
    automatic: boolean;
    admin_url?: string;
    fields?: string[];
  };
}

export type LocalStatus = 'ok' | 'warn' | 'error' | 'neutral';

export function statusClass(status: string): LocalStatus {
  if (['configured', 'reachable', 'running'].includes(status)) return 'ok';
  if (['missing_key', 'missing_url', 'unknown'].includes(status)) return 'warn';
  if (['offline', 'error'].includes(status)) return 'error';
  return 'neutral';
}

export function providerName(providerId: string): string {
  const names: Record<string, string> = {
    nvidia_nim: 'NVIDIA NIM',
    open_router: 'OpenRouter',
    mistral_codestral: 'Mistral Codestral',
    deepseek: 'DeepSeek',
    lmstudio: 'LM Studio',
    llamacpp: 'llama.cpp',
    ollama: 'Ollama',
    kimi: 'Kimi',
    wafer: 'Wafer',
    opencode: 'OpenCode Zen',
    opencode_go: 'OpenCode Go',
    zai: 'Z.ai',
  };
  if (names[providerId]) return names[providerId];
  return providerId
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

export function sourceLabel(source: string): string {
  const labels: Record<string, string> = {
    default: 'default',
    template: 'template',
    repo_env: 'repo .env',
    managed_env: '',
    explicit_env_file: 'FCC_ENV_FILE',
    process: 'process env',
  };
  return Object.prototype.hasOwnProperty.call(labels, source) ? labels[source] : source;
}

export function sourceText(field: Field): string {
  const parts: string[] = [];
  const label = sourceLabel(field.source);
  if (label) parts.push(label);
  if (field.locked) parts.push('locked');
  return parts.join(' ');
}

export const VIEW_GROUPS = [
  { id: 'providers', label: 'Providers', title: 'Providers', sections: ['providers', 'runtime'], containerId: 'providersSections' },
  { id: 'model_config', label: 'Model Config', title: 'Model Config', sections: ['models', 'thinking', 'web_tools'], containerId: 'modelConfigSections' },
  { id: 'messaging', label: 'Channels', title: 'Channels', sections: ['messaging', 'voice'], containerId: 'messagingSections' },
] as const;

export type ViewId = typeof VIEW_GROUPS[number]['id'];
