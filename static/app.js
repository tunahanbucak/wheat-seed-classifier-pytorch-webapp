document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("classifyForm");
  const btnSubmit = document.getElementById("btnSubmit");
  const btnSpinner = btnSubmit.querySelector(".btn-spinner");
  const btnText = btnSubmit.querySelector(".btn-text");

  const initialState = document.getElementById("initialState");
  const resultState = document.getElementById("resultState");

  // Sync Slider & Number Input Pairs
  const params = [
    "area",
    "perimeter",
    "compactness",
    "length_of_kernel",
    "width_of_kernel",
    "asymmetry_coefficient",
    "length_of_groove"
  ];

  params.forEach(param => {
    const num = document.getElementById(param);
    const slider = document.getElementById(`${param}_slider`);
    if (num && slider) {
      slider.addEventListener("input", () => {
        num.value = slider.value;
      });
      num.addEventListener("input", () => {
        slider.value = num.value;
      });
    }
  });

  // Presets Mapping
  const PRESET_DATA = {
    kama: {
      area: 14.88,
      perimeter: 14.57,
      compactness: 0.881,
      length_of_kernel: 5.554,
      width_of_kernel: 3.333,
      asymmetry_coefficient: 1.018,
      length_of_groove: 4.956
    },
    rosa: {
      area: 19.13,
      perimeter: 16.53,
      compactness: 0.881,
      length_of_kernel: 6.448,
      width_of_kernel: 3.696,
      asymmetry_coefficient: 3.242,
      length_of_groove: 6.421
    },
    canadian: {
      area: 11.84,
      perimeter: 13.21,
      compactness: 0.853,
      length_of_kernel: 5.175,
      width_of_kernel: 2.836,
      asymmetry_coefficient: 3.598,
      length_of_groove: 5.044
    }
  };

  // Preset Buttons Click Handler
  document.querySelectorAll(".btn-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-preset");
      const data = PRESET_DATA[type];
      if (data) {
        Object.entries(data).forEach(([key, val]) => {
          const numInput = document.getElementById(key);
          const slider = document.getElementById(`${key}_slider`);
          if (numInput) numInput.value = val;
          if (slider) slider.value = val;
        });
        // Auto submit preset to wow user
        form.dispatchEvent(new Event("submit"));
      }
    });
  });

  // Form Submit Handler
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // UI Loading state
    btnSpinner.classList.remove("hidden");
    btnText.textContent = "Model Çıkarım Yapıyor...";
    btnSubmit.disabled = true;

    const payload = {
      area: parseFloat(document.getElementById("area").value),
      perimeter: parseFloat(document.getElementById("perimeter").value),
      compactness: parseFloat(document.getElementById("compactness").value),
      length_of_kernel: parseFloat(document.getElementById("length_of_kernel").value),
      width_of_kernel: parseFloat(document.getElementById("width_of_kernel").value),
      asymmetry_coefficient: parseFloat(document.getElementById("asymmetry_coefficient").value),
      length_of_groove: parseFloat(document.getElementById("length_of_groove").value)
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const resData = await res.json();
      const data = resData.data;

      // Update UI
      renderResults(data, payload);

    } catch (err) {
      alert("Hata oluştu: " + err.message);
    } finally {
      btnSpinner.classList.add("hidden");
      btnText.textContent = "🔬 Tohum Varyetesini Sınıflandır";
      btnSubmit.disabled = false;
    }
  });

  function renderResults(data, inputPayload) {
    initialState.classList.add("hidden");
    resultState.classList.remove("hidden");

    // Top banner
    document.getElementById("varietyName").textContent = data.variety;
    document.getElementById("varietyOrigin").textContent = `📍 ${data.origin}`;
    document.getElementById("confidenceNum").textContent = `%${data.confidence_percentage}`;
    document.getElementById("varietyTitle").textContent = data.title;
    document.getElementById("varietyDesc").textContent = data.description;

    // Banner color accent
    const banner = document.getElementById("varietyBanner");
    banner.style.borderColor = data.badge_color;

    // Probabilities
    const probs = data.probabilities;
    
    // Kama
    const kamaPct = probs["Kama"] || 0;
    document.getElementById("pctKama").textContent = `%${kamaPct}`;
    document.getElementById("barKama").style.width = `${kamaPct}%`;

    // Rosa
    const rosaPct = probs["Rosa"] || 0;
    document.getElementById("pctRosa").textContent = `%${rosaPct}`;
    document.getElementById("barRosa").style.width = `${rosaPct}%`;

    // Canadian
    const canadianPct = probs["Canadian"] || 0;
    document.getElementById("pctCanadian").textContent = `%${canadianPct}`;
    document.getElementById("barCanadian").style.width = `${canadianPct}%`;

    // JSON debug preview
    const debugObj = {
      request_payload: inputPayload,
      inference_response: data
    };
    document.getElementById("jsonPreview").textContent = JSON.stringify(debugObj, null, 2);
  }
});
