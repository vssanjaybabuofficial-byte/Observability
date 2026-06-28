from __future__ import annotations
import contextvars
import functools
import json
from contextlib import contextmanager
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Chronicle

_current_parent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "chronicle_current_parent", default=None
)
_current_chronicle: contextvars.ContextVar[Optional["Chronicle"]] = contextvars.ContextVar(
    "chronicle_current", default=None
)


def _json_safe(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)


class Span:
    """A helper inside `span(...)`. Each emit advances the contextvar parent so
    subsequent emits (and any `@tool`-decorated calls) chain automatically.
    """

    def __init__(self, chronicle: "Chronicle", actor: str):
        self._c = chronicle
        self._actor = actor

    def _emit(self, kind: str, payload: dict, meta: Optional[dict] = None) -> str:
        parent = _current_parent.get()
        parents = (parent,) if parent else ()
        evt = self._c.record(
            kind, actor=self._actor, payload=payload, parents=parents, meta=meta or {}
        )
        _current_parent.set(evt.id)
        return evt.id

    def thought(self, text: str, **meta) -> str:
        return self._emit("thought", {"text": text}, meta)

    def answer(self, text: str, **meta) -> str:
        return self._emit("answer", {"text": text}, meta)

    def emit(self, kind: str, payload: dict, **meta) -> str:
        return self._emit(kind, payload, meta)


@contextmanager
def span(chronicle: "Chronicle", actor: str, parent: Optional[str] = None):
    """Context manager: events emitted inside (via `Span` or `@tool`) chain
    from `parent` automatically via a contextvar.
    """
    if parent is None:
        parent = _current_parent.get()
    sp = Span(chronicle, actor)
    tok_c = _current_chronicle.set(chronicle)
    tok_p = _current_parent.set(parent)
    try:
        yield sp
    finally:
        _current_parent.reset(tok_p)
        _current_chronicle.reset(tok_c)


@contextmanager
def active(chronicle: "Chronicle", parent: Optional[str] = None):
    """Make `chronicle` the active one for `@tool`-decorated calls without
    introducing an actor/span. Useful for top-level scripts.
    """
    tok_c = _current_chronicle.set(chronicle)
    tok_p = _current_parent.set(parent)
    try:
        yield chronicle
    finally:
        _current_parent.reset(tok_p)
        _current_chronicle.reset(tok_c)


def tool(name: Optional[str] = None) -> Callable:
    """Decorator. When the wrapped function is called inside an active
    Chronicle, records a tool_call before and a tool_result after, chained
    from the current contextvar parent.
    """
    def _decorate(fn: Callable) -> Callable:
        tool_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            c = _current_chronicle.get()
            if c is None:
                return fn(*args, **kwargs)
            parent = _current_parent.get()
            parents = (parent,) if parent else ()
            call = c.record(
                "tool_call",
                actor=f"tool:{tool_name}",
                payload={
                    "name": tool_name,
                    "args": [_json_safe(a) for a in args],
                    "kwargs": {k: _json_safe(v) for k, v in kwargs.items()},
                },
                parents=parents,
            )
            _current_parent.set(call.id)
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                err = c.record(
                    "error",
                    actor=f"tool:{tool_name}",
                    payload={"name": tool_name, "error": repr(e)},
                    parents=(call.id,),
                )
                _current_parent.set(err.id)
                raise
            res = c.record(
                "tool_result",
                actor=f"tool:{tool_name}",
                payload={"name": tool_name, "result": _json_safe(out)},
                parents=(call.id,),
            )
            _current_parent.set(res.id)
            return out

        return wrapper

    # Allow @tool as well as @tool("custom_name")
    if callable(name):
        fn, name = name, None
        return _decorate(fn)
    return _decorate
