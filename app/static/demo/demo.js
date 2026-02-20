const chatBody = document.getElementById("chatBody");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const sessionPill = document.getElementById("sessionPill");
const statusPill = document.getElementById("statusPill");
const completionModal = document.getElementById("completionModal");
const guviPayloadEl = document.getElementById("guviPayload");
const modalSubtitle = document.getElementById("modalSubtitle");
const newSessionBtn = document.getElementById("newSessionBtn");
const closeModalBtn = document.getElementById("closeModalBtn");
const dismissBtn = document.getElementById("dismissBtn");
const chatTitle = document.querySelector(".chat-title");
const chatSubtitle = document.querySelector(".chat-subtitle");
const headerAvatar = document.getElementById("headerAvatar");
const chatItems = Array.from(document.querySelectorAll(".chat-item[data-chat-id]"));

const broadcastWidget = document.getElementById("broadcastWidget");
const broadcastToggle = document.getElementById("broadcastToggle");
const broadcastMinimize = document.getElementById("broadcastMinimize");
const broadcastInput = document.getElementById("broadcastInput");
const broadcastSend = document.getElementById("broadcastSend");
const broadcastClear = document.getElementById("broadcastClear");
const DEMO_API_KEY_STORAGE_KEY = "demo_api_key";

const sampleIntro = [
  "You are chatting with the honeypot agent. Send scam-like prompts to begin.",
  "Each persona has its own session and completion status."
];

const chats = {};
let activeChatId = "ramu_uncle";

initChats();
renderActiveChat();

function ensureDemoApiKey() {
  const existing = (window.DEMO_API_KEY || "").trim();
  if (existing) {
    return existing;
  }

  const stored = (window.localStorage.getItem(DEMO_API_KEY_STORAGE_KEY) || "").trim();
  if (stored) {
    window.DEMO_API_KEY = stored;
    return stored;
  }

  const entered = window.prompt("Enter API key for demo requests:", "");
  if (entered && entered.trim()) {
    const sanitized = entered.trim();
    window.DEMO_API_KEY = sanitized;
    window.localStorage.setItem(DEMO_API_KEY_STORAGE_KEY, sanitized);
    return sanitized;
  }

  return "";
}

function createSessionId() {
  if (window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID().slice(0, 8);
  }
  return `demo-${Math.random().toString(36).slice(2, 10)}`;
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function scrollToBottom() {
  chatBody.scrollTop = chatBody.scrollHeight;
}

function initChats() {
  chatItems.forEach((item) => {
    const chatId = item.dataset.chatId;
    const chatName = item.dataset.chatName || chatId;
    const personaKey = item.dataset.persona || chatId;
    const avatarUrl = item.dataset.avatar || "";

    chats[chatId] = {
      id: chatId,
      name: chatName,
      personaKey,
      avatarUrl,
      sessionId: createSessionId(),
      history: [],
      uiMessages: [],
      completed: false,
      completionReason: null,
      guviPayload: null,
      isLoading: false
    };

    sampleIntro.forEach((line) => addUiMessage(chats[chatId], line, "system"));

    const avatarEl = item.querySelector(".avatar");
    if (avatarEl && avatarUrl) {
      avatarEl.style.backgroundImage = `url('${avatarUrl}')`;
    }

    item.addEventListener("click", () => setActiveChat(chatId));
  });
}

function setActiveChat(chatId) {
  if (!chats[chatId]) return;
  activeChatId = chatId;
  chatItems.forEach((item) => {
    item.classList.toggle("active", item.dataset.chatId === chatId);
  });
  renderActiveChat();
}

function getActiveChat() {
  return chats[activeChatId];
}

function renderActiveChat() {
  const chat = getActiveChat();
  if (!chat) return;

  chatTitle.textContent = chat.name;
  chatSubtitle.textContent = chat.completed ? "Session completed" : "Online";
  sessionPill.textContent = `Session: ${chat.sessionId}`;
  if (headerAvatar) {
    headerAvatar.style.backgroundImage = chat.avatarUrl ? `url('${chat.avatarUrl}')` : "";
  }

  chatBody.innerHTML = "";
  chat.uiMessages.forEach((msg) => {
    const node = createMessageNode(msg);
    chatBody.appendChild(node);
  });
  scrollToBottom();
  setLoading(chat.isLoading);
}

function createMessageNode(msg) {
  const node = document.createElement("div");
  const type = msg.type === "system" ? "incoming" : msg.type;
  node.className = `message ${type}`;
  const time = formatTime(new Date(msg.timestamp));
  const ticks = msg.type === "outgoing" ? "<span class=\"ticks\">✓✓</span>" : "";
  node.innerHTML = `${escapeHtml(msg.text)}<span class=\"time\">${time}${ticks}</span>`;
  return node;
}

function addUiMessage(chat, text, type, sender = null) {
  const timestamp = new Date().toISOString();
  chat.uiMessages.push({ text, type, timestamp });

  if (sender) {
    chat.history.push({ sender, text, timestamp });
  }

  if (type !== "system") {
    updateChatPreview(chat, text, timestamp);
  }

  if (chat.id === activeChatId) {
    renderActiveChat();
  }
}

function updateChatPreview(chat, text, timestamp) {
  const item = chatItems.find((el) => el.dataset.chatId === chat.id);
  if (!item) return;

  const preview = item.querySelector(".chat-preview");
  const time = item.querySelector(".chat-time");
  if (preview) {
    preview.textContent = text.length > 40 ? `${text.slice(0, 40)}…` : text;
  }
  if (time) {
    time.textContent = formatTime(new Date(timestamp));
  }
}

async function sendMessageForChat(chat, text, showCompletionModal) {
  if (!chat || chat.completed || chat.isLoading) return;

  addUiMessage(chat, text, "outgoing", "scammer");

  const conversationHistory = [...chat.history];
  const payload = {
    sessionId: chat.sessionId,
    demoPersona: chat.personaKey,
    message: {
      sender: "scammer",
      text,
      timestamp: new Date().toISOString()
    },
    conversationHistory,
    metadata: {
      channel: "WhatsApp",
      language: "English",
      locale: "IN"
    }
  };

  chat.isLoading = true;
  if (chat.id === activeChatId) {
    setLoading(true);
  }

  const demoApiKey = ensureDemoApiKey();
  if (!demoApiKey) {
    addUiMessage(chat, "Error: API key is required to send requests.", "incoming", "user");
    chat.isLoading = false;
    if (chat.id === activeChatId) {
      setLoading(false);
    }
    return;
  }

  try {
    const response = await fetch("/api/honeypot-demo", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": demoApiKey
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    const reply = data.reply || "(No response)";
    addUiMessage(chat, reply, "incoming", "user");

    if (data.completed) {
      chat.completed = true;
      chat.completionReason = data.completionReason;
      chat.guviPayload = data.guviPayload;
      if (showCompletionModal && chat.id === activeChatId) {
        showCompletion(data);
      }
    }
  } catch (error) {
    addUiMessage(chat, `Error: ${error.message}`, "incoming", "user");
  } finally {
    chat.isLoading = false;
    if (chat.id === activeChatId) {
      setLoading(false);
    }
  }
}

async function sendMessage() {
  const chat = getActiveChat();
  const text = messageInput.value.trim();
  if (!text || !chat) return;

  messageInput.value = "";
  await sendMessageForChat(chat, text, true);
}

async function broadcastMessage() {
  const text = broadcastInput.value.trim();
  if (!text) return;

  broadcastInput.value = "";
  toggleBroadcast(false);

  const sendPromises = Object.values(chats).map((chat) =>
    sendMessageForChat(chat, text, chat.id === activeChatId)
  );
  await Promise.all(sendPromises);
}

function setLoading(isLoading) {
  const chat = getActiveChat();
  const locked = chat ? chat.completed : false;
  sendBtn.disabled = isLoading || locked;
  messageInput.disabled = isLoading || locked;
  statusPill.textContent = locked ? "Completed" : isLoading ? "Thinking" : "Running";
  statusPill.classList.toggle("completed", locked);
}

function showCompletion(data) {
  setLoading(false);
  const reason = data.completionReason || "completed";
  modalSubtitle.textContent = `Conversation ended (${reason}).`;

  const payload = data.guviPayload || {};
  guviPayloadEl.textContent = JSON.stringify(payload, null, 2);
  completionModal.classList.remove("hidden");
}

function resetSession() {
  const chat = getActiveChat();
  if (!chat) return;

  chat.completed = false;
  chat.completionReason = null;
  chat.guviPayload = null;
  chat.history = [];
  chat.uiMessages = [];
  chat.sessionId = createSessionId();
  chat.isLoading = false;

  sampleIntro.forEach((line) => addUiMessage(chat, line, "system"));
  completionModal.classList.add("hidden");
  renderActiveChat();
}

function toggleBroadcast(open) {
  if (open) {
    broadcastWidget.classList.remove("minimized");
    broadcastToggle.textContent = "Send All";
  } else {
    broadcastWidget.classList.add("minimized");
    broadcastToggle.textContent = "✉";
  }
}

sendBtn.addEventListener("click", sendMessage);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    sendMessage();
  }
});

broadcastToggle.addEventListener("click", () => toggleBroadcast(true));
broadcastMinimize.addEventListener("click", () => toggleBroadcast(false));
broadcastSend.addEventListener("click", broadcastMessage);
broadcastClear.addEventListener("click", () => {
  broadcastInput.value = "";
});

newSessionBtn.addEventListener("click", resetSession);
closeModalBtn.addEventListener("click", () => completionModal.classList.add("hidden"));
dismissBtn.addEventListener("click", () => completionModal.classList.add("hidden"));
