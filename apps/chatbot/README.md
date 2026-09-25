# Chatbot reference

Settings, API and error codes for the chatbot in this folder. For how to install and run it, how it works, and how chat memory works (and its limits), see the [main README](../../README.md).

## Configuration

Set these as environment variables or in `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | required | Your Groq API key. |
| `MODEL_NAME` | `groq:qwen/qwen3.8-27b` | Model as `provider:model`, passed to LangChain's `init_chat_model`. |
| `TEMPERATURE` | `0.7` | Sampling temperature, from 0 to 2. |
| `MAX_TOKENS` | `1024` | Maximum length of a reply, in tokens. |
| `SYSTEM_PROMPT` | a short "helpful assistant" prompt | Sent before the conversation on every call. |
| `MAX_HISTORY_MESSAGES` | `7` | How many recent messages the model sees, including the new one. 7 is about the last 3 questions and answers. |
| `REQUEST_TIMEOUT_S` | `60` | Timeout for model requests, and the overall deadline for `/api/chat`. |
| `CORS_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Browser origins allowed to call the API, comma-separated. |
| `LOG_LEVEL` | `INFO` | Python log level. |

## API

Errors from the chat endpoints use this JSON shape:

```json
{ "error": { "code": "llm_rate_limited", "message": "The AI model is busy. Please wait a moment and try again." } }
```

### `GET /health`

```bash
curl -s localhost:8000/health
# {"status":"ok"}
```

### `POST /api/chat`

The body is `{"message": str, "thread_id": str | null, "message_id": str | null}`.

- `message`: 1–4000 characters, not only whitespace.
- `thread_id`: the conversation to continue. Leave it out to start a new one; the response returns the new ID. Send the same ID with follow-up questions.
- `message_id` (optional): an ID for this message. If a request fails, resend it with the same `message_id`, and the retry replaces the failed message instead of adding a copy.

IDs can contain letters, digits, `-` and `_`, up to 64 characters.

```bash
curl -s localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "My name is Ada."}'
# {"reply":"Nice to meet you, Ada!","thread_id":"2f0c…"}

curl -s localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is my name?", "thread_id": "2f0c…"}'
# {"reply":"Your name is Ada.","thread_id":"2f0c…"}
```

### `POST /api/chat/stream`

Takes the same body as `/api/chat`. The response is `text/event-stream`:

| Event | Data |
| --- | --- |
| `token` | `{"content": "Hel"}`, one per streamed chunk |
| `done` | `{"thread_id": "…"}`, the last event on success |
| `error` | `{"error": {"code", "message"}}`, the last event on failure |

Invalid input still gets a JSON `422` before the stream starts. If the stream is idle, the server sends a `: ping` comment every 15 seconds to keep the connection open.

```bash
curl -N localhost:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message": "Write a haiku about rivers"}'
# event: token
# data: {"content":"Water"}
# …
# event: done
# data: {"thread_id":"…"}
```

### Error codes

| Status | Code | When |
| --- | --- | --- |
| 422 | `validation_error` | The request body is invalid. |
| 429 | `llm_rate_limited` | Groq rate limit. |
| 502 | `llm_auth_failed` | Groq rejected the API key. Check `GROQ_API_KEY`. |
| 502 | `llm_model_not_found` | Groq doesn't know the model. Check `MODEL_NAME`. |
| 502 | `llm_error` | Any other error response from Groq. |
| 503 | `llm_unavailable` | Groq couldn't be reached. |
| 504 | `llm_timeout` | The model didn't answer within `REQUEST_TIMEOUT_S`. |
| 500 | `internal_error` | An unexpected bug. The details go to the server log only. |

Users only see the short messages above. In the server log, a problem with Groq (timeout, rate limit, bad key) is one `WARNING` line, and anything else is logged as an `ERROR` with the full traceback.

## Tests

```bash
uv run pytest apps/chatbot/tests
```

The tests replace Groq with a fake chat model based on `GenericFakeChatModel`, so they don't need an API key or network access. They cover:

- the graph: the node appends an AI reply, follow-ups see earlier messages, threads are kept apart, history is trimmed, and a retry doesn't add a duplicate message
- the API: health, chat, follow-ups by `thread_id`, validation, the SSE `token`/`done`/`error` events, and error mapping without leaking secrets
- loading the config, including the startup error when the key is missing

## Notes

- The UI loads `marked` and `DOMPurify` from jsDelivr, pinned to exact versions with integrity hashes, to render Markdown. Only formatting tags are allowed, not images or raw HTML. If the libraries don't load, replies show as plain text.
- Conversations live only in the server's memory. See [Memory limitations](../../README.md#memory-limitations) for what that means.
