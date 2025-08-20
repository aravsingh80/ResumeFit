document.getElementById("analyze").addEventListener("click", async () => {
  const scoreText = document.getElementById("scoreText");
  const gFill = document.getElementById("gFill");
  const resumeFile = document.getElementById("resumeFile").files[0];
  const jobDesc = document.getElementById("jobDesc").value;

  const header = document.getElementById("suggHeader");
  const list = document.getElementById("feedbackList");

  // reset UI
  scoreText.textContent = "…";
  setGaugePercent(gFill, 0);
  header.style.display = "none";
  list.innerHTML = "";

  if (!resumeFile || !jobDesc.trim()) {
    scoreText.textContent = "Missing info!";
    return;
  }

  const formData = new FormData();
  formData.append("resume", resumeFile);
  formData.append("job_description", jobDesc);

  try {
    const res = await fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();

    const score = Math.max(0, Math.min(100, Math.round(data.score || 0)));
    animateGaugeTo(gFill, score, 500);
    scoreText.textContent = score + "%";

    // Build pretty suggestions list
    const suggestionsRaw = Array.isArray(data.feedback) ? data.feedback : [data.feedback || ""];
    const items = normalizeSuggestions(suggestionsRaw);

    // Staggered fade-in items
    list.innerHTML = "";
    items.forEach((text, i) => {
      const li = document.createElement("li");
      li.textContent = text;
      li.classList.add("fade-in");
      li.style.animationDelay = `${i * 0.15}s`;  // 150ms stagger
      list.appendChild(li);
    });

    header.style.display = items.length ? "block" : "none";
  } catch (err) {
    scoreText.textContent = "Error";
    list.innerHTML = `<li>Backend not reachable or invalid response.</li>`;
  }
});

/** Convert raw suggestion text into clean bullet items */
function normalizeSuggestions(arr) {
  const flat = arr.join("\n");
  const parts = flat
    .split(/\n|•|- |\u2022/g)
    .map(s => s.trim())
    .filter(Boolean);

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
