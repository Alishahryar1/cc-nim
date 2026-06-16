import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ProviderMetrics:
    success_count: int = 0
    failure_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=20))
    ttfts: deque = field(default_factory=lambda: deque(maxlen=20))
    throughputs: deque = field(default_factory=lambda: deque(maxlen=20))
    last_status: int = 0

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def avg_ttft(self) -> float:
        if not self.ttfts:
            return 0.0
        return sum(self.ttfts) / len(self.ttfts)

    @property
    def avg_throughput(self) -> float:
        if not self.throughputs:
            return 0.0
        return sum(self.throughputs) / len(self.throughputs)

    @property
    def error_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.failure_count / total

class PerformanceTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerformanceTracker, cls).__new__(cls)
            cls._instance.metrics: Dict[str, ProviderMetrics] = {}
            # Track best models per category: {category: {model_ref: ProviderMetrics}}
            cls._instance.category_metrics: Dict[str, Dict[str, ProviderMetrics]] = {}
        return cls._instance

    def record_category_performance(self, category: str, model_ref: str, latency: float, status_code: int):
        if category not in self.category_metrics:
            self.category_metrics[category] = {}
        if model_ref not in self.category_metrics[category]:
            self.category_metrics[category][model_ref] = ProviderMetrics()

        m = self.category_metrics[category][model_ref]
        m.latencies.append(latency)
        if 200 <= status_code < 400: m.success_count += 1
        else: m.failure_count += 1

    def get_best_model_for_category(self, category: str, fallback: str) -> str:
        metrics_dict = self.category_metrics.get(category, {})
        if not metrics_dict:
            return fallback

        def _score(m_ref: str) -> float:
            m = metrics_dict[m_ref]
            if m.success_count == 0: return 0.0
            # Incorporate Quality Score if available (Shadow Feedback)
            quality = getattr(m, "quality_score", 1.0)
            return (quality * (1.0 - m.error_rate)) / max(m.avg_latency, 0.001)

        sorted_models = sorted(metrics_dict.keys(), key=_score, reverse=True)
        return sorted_models[0]

    def record_quality_score(self, model_ref: str, score: float):
        for cat in self.category_metrics.values():
            if model_ref in cat:
                m = cat[model_ref]
                if not hasattr(m, "quality_score"): m.quality_score = 1.0
                m.quality_score = (m.quality_score * 0.9) + (score * 0.1)

    def record_request(
        self,
        provider_id: str,
        latency: float,
        status_code: int,
        ttft: float | None = None,
        throughput: float | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        if provider_id not in self.metrics:
            self.metrics[provider_id] = ProviderMetrics()

        m = self.metrics[provider_id]
        m.latencies.append(latency)
        if ttft is not None:
            m.ttfts.append(ttft)
        if throughput is not None:
            m.throughputs.append(throughput)

        m.total_input_tokens += input_tokens
        m.total_output_tokens += output_tokens

        m.last_status = status_code
        if 200 <= status_code < 400:
            m.success_count += 1
        else:
            m.failure_count += 1

    def get_metrics(self, provider_id: str) -> ProviderMetrics:
        return self.metrics.get(provider_id, ProviderMetrics())

performance_tracker = PerformanceTracker()
