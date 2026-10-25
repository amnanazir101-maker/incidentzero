from __future__ import annotations

import json
import os
import re
from typing import Any

from groq import Groq

from incidentzero.domain.models import ModelReply, ToolCall
from .errors import PermanentModelError, TransientModelError

#: Harmony-format control tokens emitted by gpt-oss models.
#:
#: These models structure their output into channels (``analysis``, ``commentary``,
#: ``final``) and place tool calls on the commentary channel. When decoding is
#: imperfect the channel marker can be concatenated into the function-name field,
#: producing names such as ``get_logs<|channel|>commentary``.
HARMONY_TOKEN = re.compile(r"<\|[a-z_]+\|>.*$", re.IGNORECASE | re.DOTALL)

#: Pulls the tool name out of a ``failed_generation`` string that is not valid JSON.
FAILED_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')

#: Provider messages that indicate a malformed generation rather than a bad request.
#: The request itself was valid, so resampling is the correct response (R5, R11).
MALFORMED_GENERATION_MARKERS = (
    "which was not in request.tools",
    "failed to call a function",
    "tool call validation failed",
)


def sanitise_tool_name(raw: str | None) -> str:
    """Strip harmony control tokens from a tool name.

    ``get_logs<|channel|>commentary`` becomes ``get_logs``. A name that is not
    contaminated is returned unchanged, so this cannot turn a genuinely invented
    tool name into a real one: only the control-token suffix is removed, and the
    result is still validated against the registry before execution (R8).
    """
    if not raw:
        return ""
    return HARMONY_TOKEN.sub("", raw).strip()


def recover_failed_generation(exc: Exception) -> ToolCall | None:
    """Extract the tool call the model attempted from a provider-side rejection.

    Groq validates tool calls server-side and returns ``tool_use_failed`` with a
    ``failed_generation`` field holding the exact call. Resending the identical
    context would only reproduce the mistake; instead the call is handed to the
    controller's own validation boundary, which sanitises the name and returns a
    corrective observation the model can act on (R8). Returns None when the
    error carries no recoverable call.
    """
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != "tool_use_failed":
        return None
    raw = error.get("failed_generation")
    if not isinstance(raw, str):
        return None
    try:
        attempted = json.loads(raw)
    except ValueError:
        # The generation itself is broken JSON. Recover just the tool name so the
        # controller can reject the call with a corrective observation, the same
        # way the success path treats unparseable arguments.
        match = FAILED_NAME.search(raw)
        if not match:
            return None
        attempted = {"name": match.group(1), "arguments": {"__malformed_arguments__": raw}}
    if not isinstance(attempted, dict) or not attempted.get("name"):
        return None
    arguments = attempted.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"__malformed_arguments__": arguments}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(
        id=f"recovered-{abs(hash(raw)) % 10_000_000:07d}",
        name=sanitise_tool_name(str(attempted["name"])),
        arguments=arguments,
    )


class GroqModelClient:
    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-20b") -> None:
        self.model = model
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def _translate_error(self, exc: Exception) -> Exception:
        status = getattr(exc, "status_code", None)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return TransientModelError(str(exc))
        # A malformed tool call is a sampling artifact, not a configuration fault.
        # Our request carried the correct tools, so a bounded retry is the right
        # response; treating it as permanent aborts a run that would likely succeed.
        body = getattr(exc, "body", None)
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("code") == "tool_use_failed":
            return TransientModelError(str(exc))
        message = str(exc).lower()
        if any(marker in message for marker in MALFORMED_GENERATION_MARKERS):
            return TransientModelError(str(exc))
        return PermanentModelError(str(exc))

    def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0.1,
                reasoning_effort="low",
            )
        except Exception as exc:  # provider-specific classes intentionally kept out of student code
            recovered = recover_failed_generation(exc)
            if recovered is not None:
                # The provider rejected the model's call before we could. Surface
                # it as an ordinary proposal so the controller validates it and
                # gives the model feedback, instead of retrying blind.
                return ModelReply(
                    content=None,
                    tool_calls=[recovered],
                    usage={},
                    finish_reason="tool_calls",
                )
            raise self._translate_error(exc) from exc
        msg = response.choices[0].message
        calls: list[ToolCall] = []
        for call in (msg.tool_calls or []):
            try:
                args = json.loads(call.function.arguments)
            except Exception:
                args = {"__malformed_arguments__": call.function.arguments}
            # Remove harmony control tokens so a well-formed call is not discarded
            # over a decoding artifact. The name is still validated downstream.
            name = sanitise_tool_name(call.function.name)
            calls.append(ToolCall(id=call.id, name=name, arguments=args))
        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }
        return ModelReply(
            content=msg.content,
            tool_calls=calls,
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
        )

    def structured(self, messages: list[dict[str, Any]], schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                temperature=0.1,
                reasoning_effort="low",
            )
        except Exception as exc:
            raise self._translate_error(exc) from exc
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise TransientModelError(f"Model returned invalid structured JSON: {content[:160]}") from exc
