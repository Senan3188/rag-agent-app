"use strict";

const $ = (id) => document.getElementById(id);
let documents = [];
let busy = false;

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function setBusy(value) {
  busy = value;
  $("upload-button").disabled = value;
  $("file").disabled = value;
  $("documents").disabled = value || documents.length === 0;
  $("ask-button").disabled = value || !$("documents").value;
  $("refresh-button").disabled = value;
  $("question").disabled = value;
  $("question-form").setAttribute("aria-busy", String(value));
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Cannot reach the server. Check that FastAPI is running and try again.");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join("; ")
      : typeof detail === "string" ? detail : `Request failed (${response.status}).`;
    throw new Error(message);
  }
  return data;
}

function updateSelection() {
  $("result").hidden = true;
  const doc = documents.find((item) => item.id === $("documents").value);
  $("document-info").textContent = doc
    ? `${doc.embedding_model === "structured-csv" ? "Structured CSV · Searches all rows" : `${doc.chunk_count} indexed excerpts`} · Uploaded ${new Date(doc.created_at).toLocaleDateString()}`
    : "Upload a document to get started.";
  $("ask-button").disabled = busy || !doc;
}

async function loadDocuments(selectedId = $("documents").value) {
  documents = await request("/documents");
  const options = documents.map((doc) => {
    const option = document.createElement("option");
    option.value = doc.id;
    option.textContent = doc.filename;
    return option;
  });
  if (!options.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No documents yet";
    options.push(option);
  }
  $("documents").replaceChildren(...options);
  if (documents.some((doc) => doc.id === selectedId)) $("documents").value = selectedId;
  updateSelection();
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const file = $("file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  setBusy(true);
  $("result").hidden = true;
  setStatus("Uploading and indexing your document. This may take a moment…");
  try {
    const doc = await request("/documents", { method: "POST", body: form });
    // Keep a successful upload distinct from a later list-refresh failure.
    $("file").value = "";
    try {
      await loadDocuments(doc.id);
      setStatus("Document ready. Ask your first question.");
    } catch (error) {
      setStatus(`Upload succeeded. Could not refresh documents: ${error.message} Use Refresh list.`, true);
    }
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
});

$("question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  const id = $("documents").value;
  if (busy || !question || !id) return;
  setBusy(true);
  $("result").hidden = true;
  setStatus("Finding relevant excerpts and preparing your answer…");
  try {
    const result = await request(`/documents/${encodeURIComponent(id)}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    $("answer").textContent = result.answer;
    $("sources").replaceChildren(...result.sources.map((source) => {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      const page = source.page === null ? "" : ` · Page ${source.page}`;
      summary.textContent = source.kind === "csv_result"
        ? `[${source.citation}] CSV query and results`
        : `[${source.citation}] Excerpt ${source.index + 1}${page}`;
      const excerpt = document.createElement("p");
      excerpt.className = "excerpt";
      excerpt.textContent = source.text;
      details.append(summary, excerpt);
      return details;
    }));
    $("result").hidden = false;
    setStatus("Answer ready. Expand the excerpts below to check the sources.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
});

$("documents").addEventListener("change", () => {
  updateSelection();
  setStatus("Document selected. Ask a question.");
});

async function refresh() {
  setBusy(true);
  setStatus("Loading your documents…");
  try {
    await loadDocuments();
    setStatus(documents.length ? "Choose a document and ask a question." : "Upload a document to get started.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

$("refresh-button").addEventListener("click", refresh);
refresh();
