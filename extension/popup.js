document.getElementById("analyze").addEventListener("click", async () => {
  const resultEl = document.getElementById("result");
  resultEl.textContent = "Analyzing…";

  try {
    const res = await fetch("http://127.0.0.1:5000/analyze", { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const title = data.job_description || "Job Description";
    const score = data.score != null ? data.score + "%" : "N/A";
    const feedback = data.feedback || "No feedback available.";

    resultEl.textContent = `${title}\nMatch Score: ${score}\n\nFeedback:\n${feedback}`;
  } catch (err) {
    resultEl.textContent = "Error talking to backend. Is Flask running with CORS enabled?";
  }
});
