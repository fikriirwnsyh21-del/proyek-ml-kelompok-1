const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const resultSection = document.getElementById("resultSection");
const resultContent = document.getElementById("resultContent");

const FLOAT_FIELDS = new Set(["BMI"]);

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = {};

  for (const [key, value] of formData.entries()) {
    if (value === "") {
      showError(`Field "${key}" tidak boleh kosong.`);
      return;
    }
    const parsed = FLOAT_FIELDS.has(key) ? parseFloat(value) : parseInt(value, 10);
    if (Number.isNaN(parsed)) {
      showError(`Field "${key}" harus berupa angka yang valid.`);
      return;
    }
    payload[key] = parsed;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Memproses...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      if (Array.isArray(data.detail)) {
        const messages = data.detail.map((err) => {
          const field = err.loc ? err.loc[err.loc.length - 1] : "input";
          return `${field}: ${err.msg}`;
        });
        showError(`Validasi gagal: ${messages.join("; ")}`);
      } else {
        showError(data.detail || "Terjadi kesalahan saat memproses prediksi.");
      }
      return;
    }

    showResult(data);
  } catch (err) {
    showError("Gagal terhubung ke server API. Pastikan server Uvicorn berjalan.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Prediksi Risiko";
  }
});

function showResult(data) {
  const badgeClass = `badge-${data.prediction_code}`;
  const probEntries = Object.entries(data.probabilities_percent || {});

  const probHtml = probEntries
    .map(
      ([label, pct]) => `
      <div class="prob-item">
        <div class="prob-header">
          <span>${label}</span>
          <strong>${pct}%</strong>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width: ${pct}%"></div>
        </div>
      </div>`
    )
    .join("");

  resultContent.innerHTML = `
    <div class="result-badge ${badgeClass}">
      Hasil Prediksi: ${data.prediction_label}
    </div>
    <div class="prob-list">
      ${probHtml}
    </div>
    <p class="subtitle" style="margin-top: 14px; margin-bottom: 0;">
      ${data.disclaimer}
    </p>
  `;
  resultSection.classList.remove("hidden");
}

function showError(message) {
  resultContent.innerHTML = `<div class="error-box">${message}</div>`;
  resultSection.classList.remove("hidden");
}
