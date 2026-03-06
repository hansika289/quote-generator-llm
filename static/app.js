const promptInput = document.getElementById("prompt");
const generateBtn = document.getElementById("generate-btn");
const statusEl = document.getElementById("status");
const quoteEl = document.getElementById("quote");
const historyEl = document.getElementById("history");

function renderHistory(history) {
  historyEl.innerHTML = "";

  if (!history || history.length === 0) {
    const placeholder = document.createElement("div");
    placeholder.className = "history-placeholder";
    placeholder.textContent =
      "No quotes yet. Generate one to see your prompt history.";
    historyEl.appendChild(placeholder);
    return;
  }

  for (const item of history) {
    const container = document.createElement("div");
    container.className = "history-item";

    const promptLine = document.createElement("div");
    promptLine.className = "history-prompt";
    promptLine.innerHTML = "<span>Prompt:</span> " + (item.prompt || "(empty)");

    const quoteLine = document.createElement("div");
    quoteLine.className = "history-quote";
    quoteLine.textContent = item.quote || "";

    container.appendChild(promptLine);
    container.appendChild(quoteLine);
    historyEl.appendChild(container);
  }

  historyEl.scrollTop = historyEl.scrollHeight;
}

async function generateQuote() {
  const prompt = promptInput.value.trim();

  generateBtn.disabled = true;
  statusEl.textContent = "Generating...";
  statusEl.classList.remove("status--error");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        max_new_tokens: 200,
      }),
    });

    const data = await response.json();

    if (!response.ok || !data.ok) {
      const msg = data.error || "Failed to generate quote.";
      statusEl.textContent = msg;
      statusEl.classList.add("status--error");
      return;
    }

    if (typeof data.quote === "string" && data.quote.length > 0) {
      quoteEl.textContent = data.quote;
    } else {
      quoteEl.textContent = "No quote generated.";
    }

    if (Array.isArray(data.history)) {
      renderHistory(data.history);
    }

    statusEl.textContent = "Done.";
    promptInput.value = "";
  } catch (err) {
    statusEl.textContent = "Network error while generating quote.";
    statusEl.classList.add("status--error");
  } finally {
    generateBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", () => {
  void generateQuote();
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    void generateQuote();
  }
});

// initial render
renderHistory([]);
