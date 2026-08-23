#!/usr/bin/env python3
"""
Test all available models through the free-claude-code API.
Measures response time, token usage, and generates a comparison report.
"""

import json
import time
import requests
from typing import Optional
from collections import defaultdict
import sys

# Configuration
API_BASE_URL = "http://0.0.0.0:8082"
AUTH_TOKEN = "freecc"
DEFAULT_PROMPT = "What is artificial intelligence in one sentence?"
MAX_TOKENS = 100
TIMEOUT = 30

# Headers
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}


def get_all_models() -> list[dict]:
    """Fetch all available models from the API."""
    print("📦 Fetching all available models...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/v1/models",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        print(f"✓ Found {len(models)} models")
        return models
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        return []


def test_model(model_id: str, prompt: str) -> Optional[dict]:
    """Test a single model and return results."""
    try:
        start_time = time.time()

        response = requests.post(
            f"{API_BASE_URL}/v1/messages",
            headers=HEADERS,
            json={
                "model": model_id,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=TIMEOUT
        )

        elapsed = time.time() - start_time

        if response.status_code != 200:
            return {
                "model": model_id,
                "status": "error",
                "error": f"HTTP {response.status_code}",
                "time": elapsed
            }

        data = response.json()

        return {
            "model": model_id,
            "status": "success",
            "time": elapsed,
            "input_tokens": data.get("usage", {}).get("input_tokens", 0),
            "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            "content": data.get("content", [{}])[0].get("text", "")[:100] if data.get("content") else "",
            "stop_reason": data.get("stop_reason", "unknown")
        }
    except requests.Timeout:
        return {
            "model": model_id,
            "status": "timeout",
            "time": TIMEOUT,
            "error": f"Timeout after {TIMEOUT}s"
        }
    except Exception as e:
        return {
            "model": model_id,
            "status": "error",
            "error": str(e),
            "time": 0
        }


def run_tests(models: list[dict], prompt: str, max_models: Optional[int] = None) -> list[dict]:
    """Test multiple models."""
    results = []

    # Filter to unique model IDs and optionally limit
    unique_models = {}
    for model in models:
        model_id = model.get("id", "")
        if model_id and model_id not in unique_models:
            unique_models[model_id] = model

    model_ids = list(unique_models.keys())
    if max_models:
        model_ids = model_ids[:max_models]

    print(f"\n🧪 Testing {len(model_ids)} models...")
    print("=" * 80)

    for i, model_id in enumerate(model_ids, 1):
        print(f"[{i}/{len(model_ids)}] Testing: {model_id}...", end=" ", flush=True)

        result = test_model(model_id, prompt)
        results.append(result)

        if result["status"] == "success":
            print(f"✓ {result['time']:.2f}s | {result['input_tokens']}in {result['output_tokens']}out")
        elif result["status"] == "timeout":
            print(f"⏱ Timeout")
        else:
            print(f"❌ {result.get('error', 'Error')}")

    return results


def generate_report(results: list[dict], prompt: str) -> str:
    """Generate a comprehensive report."""
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    report = []
    report.append("\n" + "=" * 80)
    report.append("📊 TEST REPORT")
    report.append("=" * 80)
    report.append(f"\nPrompt: {prompt}")
    report.append(f"Max Tokens: {MAX_TOKENS}")
    report.append(f"\n📈 Summary:")
    report.append(f"  • Total Models Tested: {len(results)}")
    report.append(f"  • Successful: {len(successful)} ✓")
    report.append(f"  • Failed: {len(failed)} ❌")
    report.append(f"  • Success Rate: {len(successful)/len(results)*100:.1f}%")

    if successful:
        times = [r["time"] for r in successful]
        total_input = sum(r["input_tokens"] for r in successful)
        total_output = sum(r["output_tokens"] for r in successful)

        report.append(f"\n⏱ Response Times:")
        report.append(f"  • Fastest: {min(times):.2f}s")
        report.append(f"  • Slowest: {max(times):.2f}s")
        report.append(f"  • Average: {sum(times)/len(times):.2f}s")

        report.append(f"\n📝 Token Usage (across all successful tests):")
        report.append(f"  • Total Input Tokens: {total_input}")
        report.append(f"  • Total Output Tokens: {total_output}")
        report.append(f"  • Avg Input per Model: {total_input/len(successful):.0f}")
        report.append(f"  • Avg Output per Model: {total_output/len(successful):.0f}")

    if successful:
        report.append(f"\n🏆 Top 10 Fastest Models:")
        sorted_by_speed = sorted(successful, key=lambda x: x["time"])[:10]
        for i, r in enumerate(sorted_by_speed, 1):
            report.append(f"  {i}. {r['model']:<60} {r['time']:>7.2f}s")

    if failed:
        report.append(f"\n⚠️ Failed Models ({len(failed)}):")
        for r in failed[:20]:  # Show first 20 failures
            error = r.get("error", "Unknown error")
            report.append(f"  • {r['model']:<60} {r['status']}: {error}")
        if len(failed) > 20:
            report.append(f"  ... and {len(failed) - 20} more")

    report.append("\n" + "=" * 80 + "\n")
    return "\n".join(report)


def save_results(results: list[dict], filename: str = "model_test_results.json"):
    """Save detailed results to JSON."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {filename}")


def main():
    """Main entry point."""
    import argparse
    global TIMEOUT

    parser = argparse.ArgumentParser(description="Test all available models")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Custom prompt to test")
    parser.add_argument("--max-models", type=int, help="Limit number of models to test")
    parser.add_argument("--output", default="model_test_results.json", help="Output JSON file")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="Request timeout in seconds")

    args = parser.parse_args()

    TIMEOUT = args.timeout

    print("\n🚀 Free Claude Code - Model Benchmark Tool")
    print("=" * 80)

    # Get all models
    models = get_all_models()
    if not models:
        print("❌ No models found. Make sure the server is running.")
        sys.exit(1)

    # Run tests
    results = run_tests(models, args.prompt, args.max_models)

    # Generate and display report
    report = generate_report(results, args.prompt)
    print(report)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
