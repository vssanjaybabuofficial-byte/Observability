"""LLM-call instrumentation, duration tracking, and run-level aggregations.

Telemetry data (durations, token counts, cost, wall-clock timestamps) lives in
`Event.telemetry` — a field deliberately *excluded* from the content hash so
recording it does not destroy the cross-run determinism property.
"""
from __future__ import annotations
import functools
import time
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Optional, TYPE_CHECKING

from .ergonomics import _current_chronicle, _current_parent, _json_safe

if TYPE_CHECKING:
    from .core import Chronicle


# ---- approximate USD price per 1M tokens (input, output) ---------------------
# Snapshot for illustrative dashboard cost estimates. Override via
# `chronicle.set_prices({...})` or pass `prices=` to `Chronicle.stats(...)`.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "o1": (15.0, 60.0),
}

_PRICES = dict(DEFAULT_PRICES)


def set_prices(prices: Mapping[str, tuple[float, float]]) -> None:
    """Override the default price table at module level."""
    global _PRICES
    _PRICES = dict(prices)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int,
                      prices: Optional[Mapping[str, tuple]] = None) -> float:
    pt = prices if prices is not None else _PRICES
    p_in, p_out = pt.get(model, (0.0, 0.0))
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000.0


# ---- response sniffing -------------------------------------------------------

def _extract_usage(response: Any) -> dict:
    """Best-effort token-count extraction. Recognizes Anthropic (`input_tokens`/
    `output_tokens`) and OpenAI (`prompt_tokens`/`completion_tokens`) response
    shapes, plain dicts with the same keys, and anything `usage`-shaped.
    """
    out = {"input_tokens": 0, "output_tokens": 0}
    u = None
    if hasattr(response, "usage"):
        u = response.usage
    elif isinstance(response, Mapping) and "usage" in response:
        u = response["usage"]
    if u is None:
        return out

    def _get(obj, key):
        if isinstance(obj, Mapping):
            return obj.get(key)
        return getattr(obj, key, None)

    for src, dst in [
        ("input_tokens", "input_tokens"),
        ("prompt_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("completion_tokens", "output_tokens"),
    ]:
        v = _get(u, src)
        if v is not None:
            out[dst] = int(v)
    out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


def _extract_text(response: Any) -> str:
    """Best-effort text extraction from an Anthropic or OpenAI response."""
    if hasattr(response, "content"):
        try:
            c = response.content
            if isinstance(c, list) and c:
                first = c[0]
                return getattr(first, "text", None) or (first.get("text") if isinstance(first, Mapping) else None) or str(c)
            return str(c)
        except Exception:
            pass
    if hasattr(response, "choices"):
        try:
            ch = response.choices[0]
            msg = getattr(ch, "message", None) or (ch.get("message") if isinstance(ch, Mapping) else None)
            if msg is not None:
                return getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, Mapping) else None) or str(msg)
        except Exception:
            pass
    if isinstance(response, Mapping):
        for k in ("text", "content", "output_text"):
            if k in response:
                return str(response[k])
    return str(response)


def _extract_model(response: Any, default: Optional[str]) -> str:
    for src in ("model", "_model"):
        v = getattr(response, src, None)
        if v:
            return str(v)
        if isinstance(response, Mapping) and src in response:
            return str(response[src])
    return default or "unknown"


# ---- @llm decorator ----------------------------------------------------------

def llm(name_or_fn=None, *, model: Optional[str] = None) -> Callable:
    """Decorator for functions that call an LLM. When invoked inside an active
    Chronicle, records `llm_call` then `llm_result`, with token counts, model,
    and wall-clock duration in `telemetry`.

    Usage::

        @llm(model="claude-opus-4-7")
        def ask(messages):
            return client.messages.create(model="claude-opus-4-7", messages=messages)

        # or just @llm — model is read from the response
        @llm
        def ask(messages): ...
    """

    def _decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            c = _current_chronicle.get()
            if c is None:
                return fn(*args, **kwargs)
            parent = _current_parent.get()
            parents = (parent,) if parent else ()
            started = time.time_ns()
            t0 = time.perf_counter_ns()
            call = c.record(
                "llm_call",
                actor=f"llm:{model or fn.__name__}",
                payload={
                    "model": model,
                    "args": [_json_safe(a) for a in args],
                    "kwargs": {k: _json_safe(v) for k, v in kwargs.items()},
                },
                parents=parents,
                telemetry={"started_at_ns": started},
            )
            _current_parent.set(call.id)
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                dt_ms = (time.perf_counter_ns() - t0) / 1e6
                err = c.record(
                    "error",
                    actor=f"llm:{model or fn.__name__}",
                    payload={"error": repr(e)},
                    parents=(call.id,),
                    telemetry={"duration_ms": dt_ms},
                )
                _current_parent.set(err.id)
                raise
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            usage = _extract_usage(result)
            text = _extract_text(result)
            actual_model = _extract_model(result, model)
            res = c.record(
                "llm_result",
                actor=f"llm:{actual_model}",
                payload={
                    "model": actual_model,
                    "text": text,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                },
                parents=(call.id,),
                telemetry={
                    "duration_ms": dt_ms,
                    "cost_usd": estimate_cost_usd(actual_model, usage["input_tokens"], usage["output_tokens"]),
                },
            )
            _current_parent.set(res.id)
            return result

        return wrapper

    # Support @llm and @llm(model="...")
    if callable(name_or_fn):
        return _decorate(name_or_fn)
    return _decorate


# ---- manual recording API ----------------------------------------------------

def record_llm(
    chronicle: "Chronicle",
    *,
    model: str,
    prompt: Any,
    response: Any,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms: float = 0.0,
    parents: Iterable[str] = (),
    extra_telemetry: Optional[Mapping[str, Any]] = None,
) -> tuple:
    """Manual API for recording an LLM call when you can't or don't want to use
    the `@llm` decorator. Returns (call_event, result_event).
    """
    parents = tuple(parents)
    call = chronicle.record(
        "llm_call",
        actor=f"llm:{model}",
        payload={"model": model, "prompt": _json_safe(prompt)},
        parents=parents,
    )
    tele = {"duration_ms": float(duration_ms),
            "cost_usd": estimate_cost_usd(model, input_tokens, output_tokens)}
    if extra_telemetry:
        tele.update(extra_telemetry)
    res = chronicle.record(
        "llm_result",
        actor=f"llm:{model}",
        payload={
            "model": model,
            "text": _extract_text(response) if not isinstance(response, str) else response,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens) + int(output_tokens),
        },
        parents=(call.id,),
        telemetry=tele,
    )
    return call, res


# ---- run summaries -----------------------------------------------------------

def summarize(chronicle: "Chronicle", prices: Optional[Mapping[str, tuple]] = None) -> dict:
    """Roll up per-run telemetry. Returns counts, total tokens, latency, cost,
    and a per-model breakdown.
    """
    by_kind: dict[str, int] = defaultdict(int)
    by_actor: dict[str, int] = defaultdict(int)
    by_model: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "duration_ms": 0.0, "cost_usd": 0.0,
    })
    llm_calls = 0
    tool_calls = 0
    in_tok = out_tok = 0
    llm_ms = 0.0
    tool_ms = 0.0
    total_cost = 0.0

    for evt in chronicle:
        by_kind[evt.kind] += 1
        by_actor[evt.actor] += 1
        if evt.kind == "llm_result":
            llm_calls += 1
            it = int(evt.payload.get("input_tokens", 0))
            ot = int(evt.payload.get("output_tokens", 0))
            in_tok += it
            out_tok += ot
            dur = float(evt.telemetry.get("duration_ms", 0.0))
            llm_ms += dur
            m = str(evt.payload.get("model", "unknown"))
            cost = (
                estimate_cost_usd(m, it, ot, prices)
                if prices is not None
                else float(evt.telemetry.get("cost_usd", estimate_cost_usd(m, it, ot)))
            )
            total_cost += cost
            row = by_model[m]
            row["calls"] += 1
            row["input_tokens"] += it
            row["output_tokens"] += ot
            row["duration_ms"] += dur
            row["cost_usd"] += cost
        elif evt.kind == "tool_result":
            tool_calls += 1
            tool_ms += float(evt.telemetry.get("duration_ms", 0.0))

    return {
        "total_events": len(chronicle),
        "by_kind": dict(by_kind),
        "by_actor": dict(by_actor),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "llm_duration_ms": llm_ms,
        "tool_duration_ms": tool_ms,
        "estimated_cost_usd": total_cost,
        "by_model": {k: dict(v) for k, v in by_model.items()},
    }
