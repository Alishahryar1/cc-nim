import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

const API_KEY_ENV = "FCC_PI_API_KEY";
const BASE_URL_ENV = "FCC_PI_BASE_URL";
const CATALOG_TIMEOUT_MS = 3000;
const DEFAULT_CONTEXT_WINDOW = 128000;
const DEFAULT_MAX_TOKENS = 16384;

function requireEnvironment(name: string): string {
	const value = process.env[name]?.trim();
	if (!value) {
		throw new Error(`Missing required ${name} environment variable.`);
	}
	return value;
}

function normalizeBaseUrl(value: string): string {
	let url: URL;
	try {
		url = new URL(value);
	} catch {
		throw new Error(`${BASE_URL_ENV} is not a valid URL.`);
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") {
		throw new Error(`${BASE_URL_ENV} must use http or https.`);
	}
	url.search = "";
	url.hash = "";
	return url.toString().replace(/\/+$/, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function modelDefinition(id: string, providerModel: string): ProviderModelConfig {
	return {
		id,
		name: providerModel,
		reasoning: id === providerModel,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: DEFAULT_CONTEXT_WINDOW,
		maxTokens: DEFAULT_MAX_TOKENS,
	};
}

export function projectFccModels(payload: unknown): ProviderModelConfig[] {
	if (!isRecord(payload) || payload.object !== "list" || !Array.isArray(payload.data)) {
		throw new Error("FCC model catalog returned an invalid response shape.");
	}
	const models: ProviderModelConfig[] = [];
	const seen = new Set<string>();
	for (const entry of payload.data) {
		if (!isRecord(entry) || typeof entry.id !== "string" || typeof entry.provider_model_ref !== "string") continue;
		const id = entry.id.trim();
		const providerModel = entry.provider_model_ref.trim();
		if (!id || !providerModel.includes("/") || seen.has(id)) continue;
		seen.add(id);
		models.push(modelDefinition(id, providerModel));
	}

	if (models.length === 0) {
		throw new Error("FCC model catalog contains no routable provider models.");
	}
	return models;
}

function requestIdSuffix(response: Response): string {
	const requestId = response.headers.get("request-id") ?? response.headers.get("x-request-id");
	return requestId ? ` (request ${requestId})` : "";
}

async function fetchFccModels(baseUrl: string, apiKey: string): Promise<ProviderModelConfig[]> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), CATALOG_TIMEOUT_MS);
	try {
		let response: Response;
		try {
			response = await fetch(`${baseUrl}/v1/models?view=messages`, {
				headers: { Authorization: `Bearer ${apiKey}` },
				signal: controller.signal,
			});
		} catch (error) {
			if (error instanceof Error && error.name === "AbortError") {
				throw new Error(`FCC model catalog timed out after ${CATALOG_TIMEOUT_MS}ms.`);
			}
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(`Could not reach the FCC model catalog: ${message}`);
		}

		if (!response.ok) {
			throw new Error(`FCC model catalog returned HTTP ${response.status}${requestIdSuffix(response)}.`);
		}

		let payload: unknown;
		try {
			payload = await response.json();
		} catch (error) {
			if (error instanceof Error && error.name === "AbortError") {
				throw new Error(`FCC model catalog timed out after ${CATALOG_TIMEOUT_MS}ms.`);
			}
			throw new Error(`FCC model catalog returned invalid JSON${requestIdSuffix(response)}.`);
		}
		return projectFccModels(payload);
	} finally {
		clearTimeout(timeout);
	}
}

export default async function freeClaudeCode(pi: ExtensionAPI): Promise<void> {
	const baseUrl = normalizeBaseUrl(requireEnvironment(BASE_URL_ENV));
	const apiKey = requireEnvironment(API_KEY_ENV);
	const models = await fetchFccModels(baseUrl, apiKey);

	pi.registerProvider("free-claude-code", {
		name: "Free Claude Code",
		baseUrl,
		apiKey: `$${API_KEY_ENV}`,
		authHeader: true,
		api: "anthropic-messages",
		models,
	});
}
