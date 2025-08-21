/**********************
 * PERSISTENCE SETUP  *
 **********************/

// Restore job description on load
document.addEventListener("DOMContentLoaded", async () => {
  try {
    const { jobDesc } = await chrome.storage.local.get("jobDesc");
    const el = document.getElementById("jobDesc");
    if (jobDesc && el && !el.value) el.value = jobDesc;
  } catch (e) {
    console.error("Failed to restore jobDesc", e);
  }

  // Show saved resume (filename + size) if present
  const rec = await idbGet();
  updateResumeStatus(rec?.meta || null);
});

// Autosave JD while typing
document.getElementById("jobDesc")?.addEventListener("input", debounce(async (e) => {
  try {
    await chrome.storage.local.set({ jobDesc: e.target.value });
  } catch (e2) {
    console.error("Failed to save jobDesc", e2);
  }
}, 200));

function debounce(fn, wait) {
  let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); };
}

// ---- IndexedDB for resume blob ----
const DB_NAME   = "resumefit-db";
const STORE     = "files";
const RESUMEKEY = "currentResume";

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function idbPut(value) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(value, RESUMEKEY);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}
async function idbGet() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(RESUMEKEY);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

// Save newly chosen file
document.getElementById("resumeFile")?.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const meta = {
      name: file.name,
      type: file.type || "application/pdf",
      size: file.size,
      savedAt: Date.now()
    };
    await idbPut({ blob: file, meta });
    updateResumeStatus(meta);
  } catch (err) {
    console.error("Failed to persist resume", err);
  }
});

// Helpers to read from storage
async function getStoredResumeBlob() {
  const rec = await idbGet();
  return rec?.blob || null;
}
async function getStoredResumeMeta() {
  const rec = await idbGet();
  return rec?.meta || null;
}

// Status line updater
function updateResumeStatus(meta) {
  const statusEl = document.getElementById("currResumeStatus");
  if (!statusEl) return;
  if (!meta) {
    statusEl.textContent = "current resume: none";
    return;
  }
  const kb = Math.max(1, Math.round((meta.size || 0) / 1024));
  statusEl.textContent = `current resume: ${truncate(meta.name, 40)} (${kb} KB)`;
}

function truncate(s, max) {
  if (!s) return s;
  return s.length > max ? s.slice(0, Math.max(0, max - 3)) + "..." : s;
}

/*************************
 * YOUR EXISTING HANDLER *
 *************************/

document.getElementById("analyze").addEventListener("click", async () => {
  const scoreText = document.getElementById("scoreText");
  const gFill = document.getElementById("gFill");
  const header = document.getElementById("suggHeader");
  const list = document.getElementById("feedbackList");

  // Prefer live inputs; fallback to storage
  let resumeFile = document.getElementById("resumeFile").files[0];
  let jobDesc = document.getElementById("jobDesc").value;

  if (!jobDesc?.trim()) {
    try {
      const stored = await chrome.storage.local.get("jobDesc");
      if (stored?.jobDesc) jobDesc = stored.jobDesc;
    } catch (e) {
      console.error("Failed to load jobDesc fallback", e);
    }
  }
  if (!resumeFile) {
    try {
      const blob = await getStoredResumeBlob();
      const meta = await getStoredResumeMeta();
      if (blob) {
        resumeFile = new File([blob], meta?.name || "resume.pdf", {
          type: meta?.type || "application/pdf"
        });
      }
    } catch (e) {
      console.error("Failed to load resume fallback", e);
    }
  }

  // reset UI
  scoreText.textContent = "…";
  setGaugePercent(gFill, 0);
  header.style.display = "none";
  list.innerHTML = "";

  if (!resumeFile || !jobDesc?.trim()) {
    scoreText.textContent = "Missing info!";
    return;
  }

  const formData = new FormData();
  formData.append("resume", resumeFile);
  formData.append("job_description", jobDesc);

  try {
    const res = await fetch("http://127.0.0.1:5000/analyze", { method: "POST", body: formData });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    const score = Math.max(0, Math.min(100, Math.round(data.score || 0)));
    animateGaugeTo(gFill, score, 500);
    scoreText.textContent = score + "%";

    const suggestionsRaw = Array.isArray(data.feedback) ? data.feedback : [data.feedback || ""];
    const items = normalizeSuggestions(suggestionsRaw);

    list.innerHTML = "";
    items.forEach((text, i) => {
      const li = document.createElement("li");
      li.textContent = text;
      li.classList.add("fade-in");
      li.style.animationDelay = `${i * 0.15}s`;
      list.appendChild(li);
    });
    header.style.display = items.length ? "block" : "none";
  } catch (err) {
    console.error(err);
    scoreText.textContent = "Error";
    list.innerHTML = `<li>Backend not reachable or invalid response.</li>`;
  }
});

/** Convert raw suggestion text into clean bullet items */
function normalizeSuggestions(arr) {
  const flat = arr.join("\n");
  const parts = flat.split(/\n|•|- |\u2022/g).map(s => s.trim()).filter(Boolean);
  return parts.map(p => {
    let t = p.replace(/^suggestions?:\s*/i, "");
    if (!t) return t;
    return t.charAt(0).toUpperCase() + t.slice(1);
  });
}

function escapeHTML(str) {
  return (str || "").replace(/[&<>"']/g, s => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[s]));
}

/* -------- gauge helpers ---------- */
function setGaugePercent(pathEl, percent) {
  const p = Math.max(0, Math.min(100, percent));
  const len = pathEl.getTotalLength();
  pathEl.style.strokeDasharray = `${len} ${len}`;
  pathEl.style.strokeDashoffset = String(len * (1 - p / 100));
}

function animateGaugeTo(pathEl, targetPercent, durationMs = 500) {
  const len = pathEl.getTotalLength();
  const currentOffset = parseFloat(getComputedStyle(pathEl).strokeDashoffset || len);
  const targetOffset = len * (1 - targetPercent / 100);
  const start = performance.now();
  function tick(ts) {
    const t = Math.min(1, (ts - start) / durationMs);
    const eased = 1 - Math.pow(1 - t, 3);
    const offset = currentOffset + (targetOffset - currentOffset) * eased;
    pathEl.style.strokeDasharray = `${len} ${len}`;
    pathEl.style.strokeDashoffset = String(offset);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
