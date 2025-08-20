document.getElementById("analyze").addEventListener("click", async () => {
  const resultEl = document.getElementById("result");
  const scoreText = document.getElementById("scoreText");
  const jobTitleEl = document.getElementById("jobTitle");
  const feedbackEl = document.getElementById("feedback");
  const gFill = document.getElementById("gFill");

  const resumeFile = document.getElementById("resumeFile").files[0];
  const jobDesc = document.getElementById("jobDesc").value;

  resultEl.style.display = "block";
  scoreText.textContent = "…";
  jobTitleEl.textContent = "";
  feedbackEl.textContent = "";

  // Reset gauge to 0
  setGaugePercent(gFill, 0);

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
    const feedback = Array.isArray(data.feedback) ? data.feedback : [data.feedback];

    // Animate gauge to score
    animateGaugeTo(gFill, score, 500);
    scoreText.textContent = score + "%";

    jobTitleEl.textContent = data.job_description || "Job Description";
    feedbackEl.innerHTML = feedback.map(f => "• " + f).join("\n");

  } catch (err) {
    scoreText.textContent = "Error";
    feedbackEl.textContent = "Backend not reachable or invalid response.";
  }
});

/**
 * Prepare the stroke-dasharray fill for the semicircle.
 * The arc path length is measured at runtime.
 */
function setGaugePercent(pathEl, percent) {
  const p = Math.max(0, Math.min(100, percent));
  const len = pathEl.getTotalLength();      // full 180° arc length
  pathEl.style.strokeDasharray = `${len} ${len}`;
  // We want 0% = full offset (invisible), 100% = 0 offset (full arc)
  pathEl.style.strokeDashoffset = String(len * (1 - p / 100));
}

/** Smooth animation from current fill to target percent */
function animateGaugeTo(pathEl, targetPercent, durationMs = 500) {
  const len = pathEl.getTotalLength();
  const currentOffset = parseFloat(getComputedStyle(pathEl).strokeDashoffset || len);
  const targetOffset = len * (1 - targetPercent / 100);

  const start = performance.now();
  function tick(ts) {
    const t = Math.min(1, (ts - start) / durationMs);
    const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
    const offset = currentOffset + (targetOffset - currentOffset) * eased;
    pathEl.style.strokeDasharray = `${len} ${len}`;
    pathEl.style.strokeDashoffset = String(offset);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
