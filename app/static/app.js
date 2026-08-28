const languageNames = {
  english: "English",
  russian: "Russian",
  mandarin: "Mandarin · 简体中文",
  korean: "Korean · 한국어",
};

const source = document.querySelector("#source");
const dest = document.querySelector("#dest");
const modelset = document.querySelector("#modelset");
const corpus = document.querySelector("#corpus");
const translation = document.querySelector("#translation");
const form = document.querySelector("#translation-form");
const submit = document.querySelector("#translate");
const status = document.querySelector("#status");
const count = document.querySelector("#character-count");
const copy = document.querySelector("#copy");
const details = document.querySelector("#model-details");

function destinationsFor(value) {
  return value === "english" ? ["russian", "mandarin", "korean"] : ["english"];
}

function refreshDestinations(preferred) {
  const allowed = destinationsFor(source.value);
  dest.replaceChildren(...allowed.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = languageNames[value];
    return option;
  }));
  if (preferred && allowed.includes(preferred)) dest.value = preferred;
}

source.addEventListener("change", () => refreshDestinations());
corpus.addEventListener("input", () => { count.textContent = corpus.value.length.toLocaleString(); });

document.querySelector("#swap").addEventListener("click", () => {
  const oldSource = source.value;
  const oldDest = dest.value;
  source.value = oldDest;
  refreshDestinations(oldSource);
  if (!translation.classList.contains("empty") && !translation.classList.contains("error")) {
    const oldCorpus = corpus.value;
    corpus.value = translation.textContent;
    translation.textContent = oldCorpus;
    count.textContent = corpus.value.length.toLocaleString();
  }
});

copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(translation.textContent);
  copy.textContent = "Copied";
  setTimeout(() => { copy.textContent = "Copy"; }, 1200);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  submit.textContent = "Translating…";
  status.textContent = "The selected model may take a moment to warm up.";
  status.classList.remove("error");
  translation.textContent = "Working…";
  translation.className = "translation empty";
  details.textContent = "";
  copy.disabled = true;

  try {
    const response = await fetch("/api/v1/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: source.value,
        dest: dest.value,
        corpus: corpus.value,
        modelset: modelset.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Translation failed (${response.status})`);
    translation.textContent = data.translation;
    translation.className = "translation";
    copy.disabled = false;
    status.textContent = `Completed in ${(data.timing_ms / 1000).toFixed(2)} seconds.`;
    details.textContent = `${data.model_used.model_id} · ${data.model_used.quantization}`;
  } catch (error) {
    translation.textContent = error.message;
    translation.className = "translation error";
    status.textContent = "The request did not complete.";
    status.classList.add("error");
  } finally {
    submit.disabled = false;
    submit.textContent = "Translate message";
  }
});

refreshDestinations("russian");
