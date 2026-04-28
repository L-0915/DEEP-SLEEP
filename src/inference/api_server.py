"""FastAPI server providing an OpenAI-compatible API for the DeepSleep model.

Endpoints:
    - POST /v1/chat/completions  -- Chat completion (supports SSE streaming).
    - POST /v1/completions       -- Text completion.
    - GET  /v1/models            -- List available models.
    - GET  /health               -- Health check.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in the chat conversation."""

    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatCompletionRequest(BaseModel):
    """Request body for POST /v1/chat/completions."""

    model: str = "deepsleep-2b"
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    stream: bool = Field(default=False, description="Use server-sent events streaming")


class CompletionRequest(BaseModel):
    """Request body for POST /v1/completions."""

    model: str = "deepsleep-2b"
    prompt: str = Field(..., description="Input text prompt")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    stream: bool = Field(default=False)


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    finish_reason: str = "stop"
    index: int = 0


class CompletionChoice(BaseModel):
    text: str
    finish_reason: str = "stop"
    index: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict[str, int] = Field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: dict[str, int] = Field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "deepsleep"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_app_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "ready": False,
}

_DEFAULT_SYSTEM_PROMPT = (
    "你是一位专业的睡眠健康顾问，拥有丰富的睡眠医学知识。"
    "请基于循证医学为用户提供准确、有用的睡眠健康建议。"
    "注意：你的建议仅供参考，不能替代专业医生的诊断和治疗。"
)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    model_path: str,
    tokenizer_path: str,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_path: Path to the HuggingFace model.
        tokenizer_path: Path to the tokenizer.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="DeepSleep API",
        description="OpenAI-compatible API for the DeepSleep sleep-medicine LLM",
        version="0.1.0",
    )

    @app.on_event("startup")
    def _load_model() -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model from %s ...", model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        _app_state["model"] = model
        _app_state["tokenizer"] = tokenizer
        _app_state["ready"] = True
        logger.info("Model loaded and ready to serve requests.")

    # -- Health check -------------------------------------------------------

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        if not _app_state["ready"]:
            raise HTTPException(status_code=503, detail="Model not yet loaded")
        return {"status": "ok", "model": model_path}

    # -- List models --------------------------------------------------------

    @app.get("/v1/models")
    async def list_models() -> dict[str, list[ModelInfo]]:
        return {
            "data": [
                ModelInfo(id="deepsleep-2b"),
            ],
            "object": "list",
        }

    # -- Chat completions ---------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        if not _app_state["ready"]:
            raise HTTPException(status_code=503, detail="Model not yet loaded")

        if request.stream:
            return StreamingResponse(
                _stream_chat(request),
                media_type="text/event-stream",
            )

        result = await _generate_chat(request)
        return JSONResponse(content=result)

    # -- Text completions ---------------------------------------------------

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> Any:
        if not _app_state["ready"]:
            raise HTTPException(status_code=503, detail="Model not yet loaded")

        if request.stream:
            return StreamingResponse(
                _stream_completion(request),
                media_type="text/event-stream",
            )

        result = await _generate_completion(request)
        return JSONResponse(content=result)

    return app


# ---------------------------------------------------------------------------
# Generation helpers (module-level so the closures can access _app_state)
# ---------------------------------------------------------------------------


def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    """Convert a list of ChatMessages into a ChatML-formatted prompt string."""
    parts: list[str] = []

    # Inject default system prompt if none provided
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        parts.append(f"<|im_start|>system\n{_DEFAULT_SYSTEM_PROMPT}<|im_end|>")

    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


async def _generate_chat(request: ChatCompletionRequest) -> dict[str, Any]:
    """Generate a non-streaming chat completion."""
    import torch

    model = _app_state["model"]
    tokenizer = _app_state["tokenizer"]

    prompt = _build_chat_prompt(request.messages)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    max_new = min(request.max_tokens, 8192 - input_len)

    do_sample = request.temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=request.temperature if do_sample else None,
            top_p=request.top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    completion_len = outputs.shape[1] - input_len

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(
            message=ChatMessage(role="assistant", content=response_text),
        )],
        usage={
            "prompt_tokens": input_len,
            "completion_tokens": completion_len,
            "total_tokens": input_len + completion_len,
        },
    ).model_dump()


async def _generate_completion(request: CompletionRequest) -> dict[str, Any]:
    """Generate a non-streaming text completion."""
    import torch

    model = _app_state["model"]
    tokenizer = _app_state["tokenizer"]

    device = next(model.parameters()).device
    inputs = tokenizer(
        request.prompt, return_tensors="pt", truncation=True, max_length=8192
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    max_new = min(request.max_tokens, 8192 - input_len)

    do_sample = request.temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=request.temperature if do_sample else None,
            top_p=request.top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    completion_len = outputs.shape[1] - input_len

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[CompletionChoice(text=text)],
        usage={
            "prompt_tokens": input_len,
            "completion_tokens": completion_len,
            "total_tokens": input_len + completion_len,
        },
    ).model_dump()


async def _stream_chat(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens as server-sent events."""
    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    model = _app_state["model"]
    tokenizer = _app_state["tokenizer"]

    prompt = _build_chat_prompt(request.messages)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    max_new = min(request.max_tokens, 8192 - input_len)

    do_sample = request.temperature > 0

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new,
        "temperature": request.temperature if do_sample else None,
        "top_p": request.top_p if do_sample else 1.0,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for chunk in streamer:
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None,
            }],
        }
        yield f"data: {__import__('json').dumps(data, ensure_ascii=False)}\n\n"

    # Final chunk
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {__import__('json').dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()


async def _stream_completion(request: CompletionRequest) -> AsyncGenerator[str, None]:
    """Stream text completion tokens as server-sent events."""
    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    model = _app_state["model"]
    tokenizer = _app_state["tokenizer"]

    device = next(model.parameters()).device
    inputs = tokenizer(
        request.prompt, return_tensors="pt", truncation=True, max_length=8192
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    max_new = min(request.max_tokens, 8192 - input_len)

    do_sample = request.temperature > 0

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new,
        "temperature": request.temperature if do_sample else None,
        "top_p": request.top_p if do_sample else 1.0,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for chunk in streamer:
        data = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "text": chunk,
                "finish_reason": None,
            }],
        }
        yield f"data: {__import__('json').dumps(data, ensure_ascii=False)}\n\n"

    final = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "text": "",
            "finish_reason": "stop",
        }],
    }
    yield f"data: {__import__('json').dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="DeepSleep API Server")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, required=True,
        help="Path to the tokenizer directory",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of Uvicorn workers (default: 1)",
    )

    args = parser.parse_args()

    app = create_app(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
