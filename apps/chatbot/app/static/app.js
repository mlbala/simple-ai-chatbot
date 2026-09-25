"use strict";

const NEAR_BOTTOM_PX = 80;
const MAX_INPUT_HEIGHT_PX = 200;

const els = {
  scroller: document.getElementById("messages"),
  list: document.getElementById("message-list"),
  emptyState: document.getElementById("empty-state"),
  form: document.getElementById("composer"),
  input: document.getElementById("message-input"),
  send: document.getElementById("send-button"),
  newChat: document.getElementById("new-chat"),
};

const state = {
  threadId: newId(), // this conversation; a reload or "New chat" starts a new one
  busy: false,
  controller: null, // AbortController for the in-flight reply, if any
};

/** An error whose message came from the server and is safe to show as-is. */
class ChatError extends Error {}

/* ---------- Rendering ---------- */

// Only the tags Markdown needs. No images, so model output can't make the browser
// fetch arbitrary URLs (a known way to leak chat content).
const SANITIZE_OPTIONS = {
  ALLOWED_TAGS: [
    "p", "br", "hr", "strong", "em", "del", "code", "pre", "blockquote", "ul", "ol", "li", "a",
    "h1", "h2", "h3", "h4", "h5", "h6", "table", "thead", "tbody", "tr", "th", "td",
  ],
  ALLOWED_ATTR: ["href", "title", "class", "start", "align"],
};

const markdownEnabled = setUpMarkdown();

function setUpMarkdown() {
  if (!window.marked || !window.DOMPurify) return false;
  window.marked.use({ gfm: true, breaks: true });
  window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
  return true;
}

function renderMarkdown(target, text) {
  if (markdownEnabled) {
    target.innerHTML = window.DOMPurify.sanitize(window.marked.parse(text), SANITIZE_OPTIONS);
  } else {
    target.classList.add("plain");
    target.textContent = text;
  }
}

function appendMessage(role, text) {
  els.emptyState.hidden = true;
  const item = document.createElement("div");
  item.className = `message message-${role}`;
  const label = document.createElement("span");
  label.className = "visually-hidden";
  label.textContent = role === "user" ? "You said:" : "Assistant said:";
  const body = document.createElement("div");
  body.className = "message-body";
  item.append(label, body);

  // User text is never interpreted as HTML or Markdown.
  if (role === "user") body.textContent = text;
  else renderMarkdown(body, text);

  els.list.append(item);
  return { item, body };
}

function appendTypingIndicator() {
  const message = appendMessage("assistant", "");
  message.body.innerHTML =
    '<span class="typing"><span class="typing-dot"></span><span class="typing-dot"></span>' +
    '<span class="typing-dot"></span><span class="visually-hidden">Assistant is typing…</span></span>';
  return message;
}

/** Accumulates streamed tokens and re-renders at most once per animation frame. */
function createReplyRenderer(message) {
  let text = "";
  let frame = 0;
  const flush = () => {
    frame = 0;
    const pinned = isNearBottom();
    renderMarkdown(message.body, text);
    if (pinned) scrollToBottom();
  };
  return {
    append(token) {
      text += token;
      if (!frame) frame = requestAnimationFrame(flush);
    },
    finish() {
      if (frame) cancelAnimationFrame(frame);
      flush();
    },
  };
}

function showError(error, retry) {
  if (!(error instanceof ChatError)) console.error(error);
  const notice = document.createElement("div");
  notice.className = "notice";
  notice.setAttribute("role", "alert");
  const text = document.createElement("p");
  text.textContent =
    error instanceof ChatError
      ? error.message
      : "Couldn't reach the server. Check your connection and try again.";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "button button-small";
  button.textContent = "Retry";
  button.addEventListener("click", () => {
    if (state.busy) return;
    notice.remove();
    retry();
  });
  notice.append(text, button);
  els.list.append(notice);
  scrollToBottom();
}

function clearErrors() {
  els.list.querySelectorAll(".notice").forEach((notice) => notice.remove());
}

function isNearBottom() {
  const { scrollHeight, scrollTop, clientHeight } = els.scroller;
  return scrollHeight - scrollTop - clientHeight < NEAR_BOTTOM_PX;
}

function scrollToBottom() {
  els.scroller.scrollTop = els.scroller.scrollHeight;
}

function setBusy(busy) {
  state.busy = busy;
  // Screen readers announce the finished reply once instead of every token.
  els.list.setAttribute("aria-busy", String(busy));
  updateSendButton();
}

function updateSendButton() {
  els.send.disabled = state.busy || !els.input.value.trim();
}

function autosizeInput() {
  const input = els.input;
  input.style.height = "auto";
  // scrollHeight excludes the border, which border-box sizing counts in `height`.
  const border = input.offsetHeight - input.clientHeight;
  const height = Math.min(input.scrollHeight + border, MAX_INPUT_HEIGHT_PX);
  input.style.height = `${height}px`;
  input.style.overflowY = height >= MAX_INPUT_HEIGHT_PX ? "auto" : "hidden";
}

/* ---------- Server-Sent Events over fetch (EventSource can't POST) ---------- */

async function* readServerSentEvents(body) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value.replace(/\r\n?/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const event = parseEventBlock(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      if (event) yield event;
    }
  }
}

function parseEventBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue; // blank or keep-alive comment
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: JSON.parse(data.join("\n")) } : null;
}

/* ---------- API calls ---------- */

async function errorFromResponse(response) {
  try {
    const body = await response.json();
    if (body?.error?.message) return new ChatError(body.error.message);
  } catch {
    /* not a JSON error body */
  }
  return new ChatError(`The server returned an error (HTTP ${response.status}).`);
}

async function sendMessage(text, messageId) {
  clearErrors();
  setBusy(true);
  const controller = new AbortController();
  state.controller = controller;
  const placeholder = appendTypingIndicator();
  scrollToBottom();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message: text, thread_id: state.threadId, message_id: messageId }),
      signal: controller.signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    await streamReply(response.body, createReplyRenderer(placeholder));
  } catch (error) {
    placeholder.item.remove();
    // Retrying with the same message ID replaces the failed message on the server.
    if (error.name !== "AbortError") showError(error, () => sendMessage(text, messageId));
  } finally {
    // A newer request (after "New chat") owns the busy state now.
    if (state.controller === controller) {
      state.controller = null;
      setBusy(false);
    }
  }
}

async function streamReply(body, reply) {
  for await (const { event, data } of readServerSentEvents(body)) {
    if (event === "token") {
      reply.append(data.content);
    } else if (event === "done") {
      reply.finish();
      return;
    } else if (event === "error") {
      throw new ChatError(data.error.message);
    }
  }
  throw new ChatError("The connection closed before the reply finished.");
}

/* ---------- Wiring ---------- */

function newId() {
  // crypto.randomUUID needs a secure context; getRandomValues works everywhere.
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

function startNewChat() {
  state.controller?.abort();
  state.controller = null;
  state.threadId = newId();
  els.list.replaceChildren();
  els.emptyState.hidden = false;
  setBusy(false);
  els.input.focus();
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = els.input.value;
  if (state.busy || !text.trim()) return;
  appendMessage("user", text);
  els.input.value = "";
  autosizeInput();
  sendMessage(text, newId());
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

els.input.addEventListener("input", () => {
  autosizeInput();
  updateSendButton();
});

els.newChat.addEventListener("click", startNewChat);

els.input.focus();
