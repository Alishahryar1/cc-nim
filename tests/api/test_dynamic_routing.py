from api.model_router import ModelRouter
from api.performance import performance_tracker
from config.settings import Settings


def test_dynamic_routing_ranking():
    settings = Settings.model_construct(model="slow/m1,fast/m1")
    router = ModelRouter(settings)

    # 1. Initially, they might be in order or equal
    candidates1 = router.resolve_candidates("claude-3-sonnet")
    assert candidates1[0].provider_id == "slow"

    # 2. Record metrics: 'fast' is fast, 'slow' is slow
    performance_tracker.record_request("fast", 0.1, 200)
    performance_tracker.record_request("slow", 2.0, 200)

    # 3. Resolve again: 'fast' should now be first
    candidates2 = router.resolve_candidates("claude-3-sonnet")
    assert candidates2[0].provider_id == "fast"
    assert candidates2[1].provider_id == "slow"


def test_dynamic_routing_reliability():
    settings = Settings.model_construct(model="unreliable/m1,reliable/m1")
    router = ModelRouter(settings)

    # Record metrics: 'reliable' has success, 'unreliable' has failures
    performance_tracker.record_request("reliable", 0.5, 200)
    performance_tracker.record_request("unreliable", 0.5, 500)

    candidates = router.resolve_candidates("claude-3-sonnet")
    assert candidates[0].provider_id == "reliable"
