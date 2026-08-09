#!/usr/bin/env python3
"""Correctness and sequential stability acceptance test for the DSpark API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_KEY = ""


def request_json(url: str, payload: dict | None = None, timeout: int = 300) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def chat(base_url: str, model: str, messages: list[dict], **kwargs: object) -> dict:
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 32,
    }
    payload.update(kwargs)
    result = request_json(f"{base_url}/chat/completions", payload)
    choices = result.get("choices") or []
    if not choices:
        raise AssertionError(f"response has no choices: {result}")
    return result


def message(result: dict) -> dict:
    return result["choices"][0]["message"]


def final_text(result: dict) -> str:
    return str(message(result).get("content") or "").strip()


def main() -> int:
    global API_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--sequential", type=int, default=200)
    parser.add_argument("--long-tokens", type=int, default=32768)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    args = parser.parse_args()
    API_KEY = args.api_key
    base_url = args.base_url.rstrip("/")

    models = request_json(f"{base_url}/models", timeout=10)
    ids = {item.get("id") for item in models.get("data", [])}
    assert args.model in ids, f"{args.model!r} not advertised: {sorted(ids)}"
    print(f"models: PASS ({args.model})", flush=True)

    basic = chat(
        base_url,
        args.model,
        [{"role": "user", "content": "Reply with exactly OK."}],
        max_tokens=128,
    )
    text = final_text(basic)
    assert "OK" in text.upper(), f"unexpected basic response: {text!r}"
    assert "<think>" not in text and "</think>" not in text
    assert message(basic).get("reasoning"), "thinking output was not separated"
    print("basic and reasoning-boundary: PASS", flush=True)

    multi = chat(
        base_url,
        args.model,
        [
            {"role": "system", "content": "The verification word is ALBATROSS."},
            {"role": "user", "content": "What is the verification word?"},
        ],
        chat_template_kwargs={"thinking": False},
    )
    assert "ALBATROSS" in final_text(multi).upper(), final_text(multi)
    print("role boundary: PASS", flush=True)

    tool = chat(
        base_url,
        args.model,
        [{"role": "user", "content": "Use the add tool to add 17 and 25."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
        tool_choice="auto",
        chat_template_kwargs={"thinking": False},
        max_tokens=128,
    )
    calls = message(tool).get("tool_calls") or []
    assert calls, f"no tool call: {message(tool)}"
    function = calls[0].get("function") or {}
    assert function.get("name") == "add", function
    arguments = json.loads(function.get("arguments") or "{}")
    assert sorted((arguments.get("a"), arguments.get("b"))) == [17, 25], arguments
    print("tool parser and JSON arguments: PASS", flush=True)

    # Repeating a short, stable token gives a large prefill without requiring a
    # local tokenizer. prompt token counts are checked from vLLM usage details.
    long_prompt = ("alpha " * args.long_tokens) + "\nReply with LONG_CONTEXT_OK."
    started = time.monotonic()
    long_result = chat(
        base_url,
        args.model,
        [{"role": "user", "content": long_prompt}],
        max_tokens=32,
        chat_template_kwargs={"thinking": False},
    )
    elapsed = time.monotonic() - started
    assert "LONG_CONTEXT_OK" in final_text(long_result).upper(), final_text(long_result)
    prompt_tokens = int((long_result.get("usage") or {}).get("prompt_tokens") or 0)
    assert prompt_tokens >= args.long_tokens // 2, prompt_tokens
    print(f"long prefill: PASS ({prompt_tokens} tokens, {elapsed:.1f}s)", flush=True)

    started = time.monotonic()
    for index in range(1, args.sequential + 1):
        result = chat(
            base_url,
            args.model,
            [{"role": "user", "content": "Reply with exactly OK."}],
            max_tokens=16,
            chat_template_kwargs={"thinking": False},
        )
        text = final_text(result)
        assert "OK" in text.upper(), f"request {index}: {text!r}"
        if index % 20 == 0 or index == args.sequential:
            print(f"sequential: {index}/{args.sequential}", flush=True)
    elapsed = time.monotonic() - started
    print(f"sequential stability: PASS ({args.sequential} requests, {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, urllib.error.URLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
