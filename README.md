# Simple AI Chatbot

A small chatbot web app that is easy to read and run: a FastAPI backend, a one-node LangGraph graph, the Groq API for the model, and a plain HTML/CSS/JS chat page with no build step.

- Replies stream in token by token.
- Follow-up questions work: LangGraph's built-in `InMemorySaver` keeps each conversation in server memory, with no database.
- When something goes wrong, users see a short, friendly message and the details go to the server log.

It has no guardrails against prompt injection, no tool calling and only minimal memory. Read [Limitations](#limitations) before putting it anywhere public.

## How to run

### 1. Install

You need [uv](https://docs.astral.sh/uv/), Python 3.14+ and a [Groq API key](https://console.groq.com/keys).

```bash
uv sync
```

Without uv, use pip and the pinned `requirements.txt`:

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
```

Open `.env` and paste your key after `GROQ_API_KEY=`. The other settings are optional; the [configuration table](apps/chatbot/README.md#configuration) explains each one.

### 3. Start the server

Run this from the repo root, because the app reads `.env` from the folder you start it in:

```bash
uv run uvicorn apps.chatbot.app.main:app --reload
```

With pip, run the same command without `uv run`. Then open <http://localhost:8000> to chat, or <http://localhost:8000/docs> for the interactive API docs.

`--reload` restarts the server whenever you save a code file, which is handy while developing. Leave it off otherwise, because every restart clears all conversations (see [Memory limitations](#memory-limitations)).

If `GROQ_API_KEY` is missing, the app doesn't start, and the error ends with:

```
apps.chatbot.app.config.ConfigError: Missing or invalid settings. Set them in .env or as environment variables:
  - GROQ_API_KEY: Field required
```

Add the key to `.env`, then restart the server: press Ctrl+C and run the command again.

### Run the tests

The tests use a fake model, so they don't need an API key or network access:

```bash
uv run pytest
```

## How the app works

```
Browser    chat page (static/index.html, app.js, styles.css)
   │  ▲    POST /api/chat/stream {"message", "thread_id"}
   │  │    the reply comes back token by token (Server-Sent Events)
   ▼  │
FastAPI    api/routes.py: checks the request, runs the graph, turns errors into friendly messages
   │
   ▼
LangGraph  graph.py: START → chatbot → END   ◄──►   InMemorySaver: each thread's messages, in RAM
   │
   ▼
Groq API   the AI model (MODEL_NAME), called through LangChain (llm.py)
```

What happens when you press **Send**:

1. The chat page sends your message and the conversation's `thread_id` to `POST /api/chat/stream`. The page makes a new `thread_id` when it loads and when you click **New chat**.
2. FastAPI checks the request (`schemas.py`). The message must be 1–4000 characters and not blank; otherwise the reply is a `422` error.
3. LangGraph loads the messages saved under that `thread_id` and adds your new message.
4. The `chatbot` node takes the last 7 messages, puts the system prompt in front, and sends them to Groq.
5. The reply streams back to the browser as `token` events, then a final `done` event.
6. LangGraph saves the updated conversation (your message and the reply) under the same `thread_id`.

If anything fails along the way, the browser gets an `error` event with a short message, such as "The AI model took too long to respond. Please try again.", and the details go to the server log. All of this is in one file, `errors.py`. `POST /api/chat` works the same way but returns the whole reply at once as JSON.

A few design choices keep the code small:

- **One node, built once.** The graph has a single `chatbot` node. It is compiled when the server starts (in `main.py`) and shared by all requests.
- **The state is just a list of messages:** `messages: Annotated[list[AnyMessage], add_messages]`. `add_messages` appends new messages and replaces a message that has the same ID. That's how **Retry** resends a failed message without adding a copy.
- **The system prompt isn't stored.** It's added in front of the messages on every model call.

## Chat memory

The app uses LangGraph's built-in short-term memory: a **checkpointer**. `main.py` creates one `InMemorySaver` for the whole app and compiles the graph with it:

```python
graph = build_graph(llm, InMemorySaver(), ...)   # main.py
builder.compile(checkpointer=checkpointer)       # graph.py
```

Every request names its conversation with a `thread_id`:

```python
graph.astream({"messages": [HumanMessage("What is my name?")]}, {"configurable": {"thread_id": "abc123"}})
```

With that ID, LangGraph loads the conversation's saved messages before the `chatbot` node runs and saves them again afterwards. That's what makes follow-up questions work.

**What the model sees.** The thread keeps every message, but the model only gets the last `MAX_HISTORY_MESSAGES` (default 7): your new question plus the last 3 questions and answers. `trim_messages` does the cutting in `graph.py`, and the window always starts on one of your messages.

### Memory limitations

`InMemorySaver` is a Python dictionary inside the server process. That keeps the app simple, with no database to set up, but it has limits:

- **A restart wipes every conversation.** That includes the automatic restart `--reload` does each time you save a code file.
- **Only run one server process.** Each process has its own memory. With `--workers 2`, or two copies behind a load balancer, a follow-up question can reach a process that has never seen the conversation.
- **Memory use only grows.** Nothing is deleted while the server runs. `InMemorySaver` also keeps every checkpoint, not just the latest: 3 per question, each with a full copy of the conversation. So storage grows faster than the chat itself. In a quick test, one conversation of 20 questions and answers (about 200 characters each) used about 375 KB. Restarting the server is the only cleanup.
- **The model forgets older messages.** Only the last 7 messages are sent. If you tell it your name in your first message, it no longer knows it by your fifth question. Raise `MAX_HISTORY_MESSAGES` to remember more, at the cost of more tokens per request.
- **Reloading the page starts over.** The page keeps its `thread_id` in a JavaScript variable only. After a reload or **New chat**, the old conversation stays in server memory, but you can't get back to it.
- **There's no login.** Anyone who knows a `thread_id` can continue that conversation. Put the app behind your own authentication before exposing it publicly.

To keep conversations across restarts or share them between servers, LangGraph has database-backed checkpointers (for example `langgraph-checkpoint-sqlite` or `langgraph-checkpoint-postgres`) that take `InMemorySaver`'s place. Those, along with long-term memory and monitoring, are out of scope for this project.

## Limitations

This is a small learning project. The model is called directly, with no safety layer around it, and the app isn't ready to face the public internet as it is. These are left out on purpose to keep the code small.

### No guardrails against prompt injection

Your message goes to the model exactly as you typed it. The only checks are in `schemas.py`: 1–4000 characters, not blank. Nothing filters what goes in or checks what comes out.

- **The system prompt is only a request.** It's plain text sent in front of the conversation, and a message like "Ignore your previous instructions and…" can talk the model out of it. Jailbreaks work as well as the model lets them.
- **The system prompt isn't secret.** Users can ask the model to repeat it, so don't put keys, internal URLs or anything private in `SYSTEM_PROMPT`.
- **Injected instructions stick around.** A message stays in the thread and is resent with the next few questions, until it falls out of the last-7 window.
- **No content moderation.** Nothing checks messages or replies for harmful, offensive or off-topic content. What the model refuses depends only on its own safety training.
- **Replies aren't fact-checked.** The model can make things up and state them confidently.

DOMPurify in the browser only stops a reply from running scripts or loading images on the chat page. It protects the page (XSS), not the model.

For now the risk is limited. The app reads no web pages, files or documents, so the only way in is a user's own message, and the model has no tools or private data, so the worst it can do is write text. That changes as soon as you add tools, retrieval over your documents or user data: add guardrails first.

### No tool calling

The graph has one node that sends the conversation to the model and saves its text reply. The model isn't given any tools (no `bind_tools`, no `ToolNode`), so it can't search the web, read files, call APIs or run code. Everything it says comes from its training data: answers about recent events or anything after the model's training cutoff can be out of date or wrong, and it can't look anything up to check.

### No memory management

Chat memory is the bare minimum: a fixed window over one conversation, kept in RAM. [Memory limitations](#memory-limitations) has the details. In short:

- **Older messages are dropped, not summarized.** Anything before the last 7 messages is gone as far as the model is concerned.
- **The window counts messages, not tokens.** Seven long messages cost far more tokens than seven short ones. With the defaults this stays small, but a much larger `MAX_HISTORY_MESSAGES` can run past the model's context window.
- **No long-term memory.** Nothing carries over between conversations, so every new chat starts knowing nothing about you.
- **No cleanup.** Conversations are never deleted or expired while the server runs, and a restart wipes them all.

### Other things that aren't there

- **No login and no rate limiting.** Anyone who can reach the server can chat as much as they like on your Groq API key, and can continue any conversation whose `thread_id` they know. `CORS_ORIGINS` only restricts browsers, not `curl` or scripts.
- **No handling of personal data.** Every message is sent to Groq and kept in server memory as typed. Nothing detects or removes personal information.
- **No monitoring or evaluation.** There's no tracing, metrics or quality checks on replies, only the server log.
- **One model, no fallback.** If Groq is down or rate-limited, the user gets an error message and has to try again.

Before deploying it anywhere public, put it behind authentication and rate limiting at the very least, and add input and output guardrails before giving the model tools or access to data.

## Project layout

```
.env.example          every setting with its default; copy it to .env
pyproject.toml        dependencies and tool settings
uv.lock               exact dependency versions for uv
requirements.txt      the same versions for pip
apps/chatbot/
  app/
    main.py           creates the app: settings, the graph with InMemorySaver, routes, chat page
    config.py         settings, read from .env
    graph.py          the LangGraph graph: State, chatbot node, history trimming
    llm.py            creates the Groq chat model
    schemas.py        request and response shapes
    errors.py         error handling: user-facing messages, logging, JSON error shape
    api/routes.py     /api/chat, /api/chat/stream, /health
    static/           chat page (HTML, CSS, JS)
  tests/              pytest suite (uses a fake model, never calls Groq)
```

The [chatbot reference](apps/chatbot/README.md) lists every setting, the API with `curl` examples, and the error codes.

## Updating dependencies

`requirements.txt` is generated from `uv.lock` and holds runtime dependencies only. After changing dependencies, regenerate it:

```bash
uv export --format requirements-txt --no-hashes --no-dev --no-emit-project --no-annotate -o requirements.txt
```

## License

[MIT](LICENSE)
