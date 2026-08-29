/* bob.js — Ask BOB floating assistant (Stage 16 UI).
   Talks ONLY to the backend /api/v1/ask endpoint. No polling, no background
   AI work; requests are sent only when the user submits a question. */
(function () {
  "use strict";

  // Backend base URL. Defaults to the local REST API; override with
  // window.BOB_API_BASE if the API is served elsewhere / same-origin.
  var API_BASE = window.BOB_API_BASE || "http://127.0.0.1:8000";

  var fab = document.getElementById("bobFab");
  var panel = document.getElementById("bobPanel");
  var closeBtn = document.getElementById("bobClose");
  var messages = document.getElementById("bobMessages");
  var suggestions = document.getElementById("bobSuggestions");
  var form = document.getElementById("bobForm");
  var input = document.getElementById("bobInput");
  var sendBtn = document.getElementById("bobSend");
  var AVATAR = "assets/bob-avatar.png";

  var busy = false;
  var welcomed = false;
  var history = [];  // short conversation context sent with each question

  var NEAR_BOTTOM_PX = 48;
  function nearBottom() {
    return messages.scrollHeight - messages.scrollTop - messages.clientHeight <= NEAR_BOTTOM_PX;
  }
  // Auto-scroll only when the user is already near the bottom, or when they just
  // sent a message (force). This keeps the newest message visible without yanking
  // the user down while they scroll up to read older conversation.
  function scrollToBottom() {
    // Scroll instantly so the newest message is shown without smooth-animation
    // lag (manual wheel scrolling stays smooth via the CSS rule).
    var prev = messages.style.scrollBehavior;
    messages.style.scrollBehavior = "auto";
    messages.scrollTop = messages.scrollHeight;
    messages.style.scrollBehavior = prev;
  }

  function openBob() {
    panel.hidden = false;
    fab.setAttribute("aria-expanded", "true");
    if (!welcomed) {
      renderWelcome();
      welcomed = true;
    }
    // Defer focus until after layout so the input is focusable.
    setTimeout(function () { input.focus(); }, 0);
  }

  function closeBob() {
    panel.hidden = true;
    fab.setAttribute("aria-expanded", "false");
    fab.focus();
  }

  function toggleBob() {
    if (panel.hidden) openBob(); else closeBob();
  }

  function renderWelcome() {
    messages.innerHTML = "";
    var w = document.createElement("div");
    w.className = "bob-welcome";
    w.innerHTML =
      '<img src="' + AVATAR + '" alt="BOB AI Energy Assistant">' +
      "<p><strong>Hi, I'm BOB.</strong></p>" +
      "<p>Ask me about your energy system, PZEM meters, faults, forecasts, " +
      "bill predictions, or energy-saving opportunities.</p>";
    messages.appendChild(w);
  }

  function addMessage(role, text, isError) {
    // Clear the welcome placeholder once a real conversation starts.
    if (welcomed && messages.querySelector(".bob-welcome")) {
      messages.innerHTML = "";
    }
    // Decide stick-to-bottom BEFORE appending: once a tall message is added the
    // distance-to-bottom grows and a post-append nearBottom() check would miss it.
    var stick = (role === "user") || nearBottom();
    var wrap = document.createElement("div");
    wrap.className = "bob-msg " + (role === "user" ? "bob-user" : "bob-bot") +
      (isError ? " bob-error" : "");

    if (role === "bot") {
      var img = document.createElement("img");
      img.src = AVATAR;
      img.alt = "BOB";
      img.className = "bob-mini-avatar";
      wrap.appendChild(img);
    }
    var bubble = document.createElement("div");
    bubble.className = "bob-bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    messages.appendChild(wrap);
    if (stick) scrollToBottom();
  }

  function setBusy(state) {
    busy = state;
    sendBtn.disabled = state;
    input.disabled = state;
    sendBtn.textContent = state ? "…" : "→";
  }

  function showTyping() {
    if (messages.querySelector(".bob-welcome")) messages.innerHTML = "";
    var wrap = document.createElement("div");
    wrap.className = "bob-msg bob-bot bob-typing";
    wrap.id = "bobTyping";
    var img = document.createElement("img");
    img.src = AVATAR; img.alt = "BOB"; img.className = "bob-mini-avatar";
    var bubble = document.createElement("div");
    bubble.className = "bob-bubble";
    bubble.textContent = "BOB is thinking…";
    wrap.appendChild(img); wrap.appendChild(bubble);
    var stickTyping = nearBottom();
    messages.appendChild(wrap);
    if (stickTyping) scrollToBottom();
  }

  function removeTyping() {
    var t = document.getElementById("bobTyping");
    if (t) t.remove();
  }

  function sendQuestion(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    addMessage("user", text);
    history.push({ role: "user", content: text });
    setBusy(true);
    showTyping();

    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, 15000);

    fetch(API_BASE + "/api/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, history: history.slice(-4) }),
      signal: controller.signal
    })
      .then(function (resp) {
        return resp.json().then(function (data) {
          if (!resp.ok || (data && data.status === "error")) {
            throw new Error((data && data.error && data.error.message) || "Request failed");
          }
          return data;
        });
      })
      .then(function (data) {
        removeTyping();
        var answer = (data && data.answer) ||
          "Sorry, I couldn't generate a response right now.";
        addMessage("bot", answer);
        history.push({ role: "bot", content: answer });
      })
      .catch(function (err) {
        removeTyping();
        addMessage("bot",
          "I couldn't reach the Ask BOB service. Please make sure the backend " +
          "API is running and try again.", true);
      })
      .finally(function () {
        clearTimeout(timer);
        setBusy(false);
        input.focus();
      });
  }

  // Events
  fab.addEventListener("click", toggleBob);
  closeBtn.addEventListener("click", closeBob);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !panel.hidden) closeBob();
  });
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    sendQuestion(input.value);
    input.value = "";
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });
  suggestions.addEventListener("click", function (e) {
    var btn = e.target.closest(".bob-chip");
    if (!btn) return;
    sendQuestion(btn.getAttribute("data-q"));
  });
})();
