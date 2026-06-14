// =============================================================================
// Predictive Maintenance Dashboard — JavaScript Controller
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const btnSubmit = document.getElementById("btn-submit");
    const spinner = document.getElementById("btn-spinner");
    const placeholder = document.getElementById("placeholder-text");
    const resultDisplay = document.getElementById("result-display");
    const resultValue = document.getElementById("prediction-result");
    const indicatorBar = document.getElementById("indicator-bar");
    const fsBadge = document.getElementById("fs-badge");

    // Dynamic slider label values updates
    const sliders = [
        { id: "air_temp", valId: "air_temp_val", suffix: " K" },
        { id: "process_temp", valId: "process_temp_val", suffix: " K" },
        { id: "rotational_speed", valId: "rotational_speed_val", suffix: " rpm" },
        { id: "torque", valId: "torque_val", suffix: " Nm" },
        { id: "tool_wear", valId: "tool_wear_val", suffix: " min" }
    ];

    sliders.forEach(slider => {
        const el = document.getElementById(slider.id);
        const label = document.getElementById(slider.valId);
        if (el && label) {
            el.addEventListener("input", (e) => {
                label.textContent = e.target.value + slider.suffix;
            });
        }
    });

    // Handle form prediction submit
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // UI state: Loading
        spinner.style.display = "block";
        btnSubmit.querySelector(".btn-text").textContent = "Analyzing...";
        btnSubmit.disabled = true;

        // Retrieve data
        const formData = new FormData(form);
        const requestData = {
            "Type": formData.get("Type"),
            "Air temperature [K]": parseFloat(formData.get("Air temperature [K]")),
            "Process temperature [K]": parseFloat(formData.get("Process temperature [K]")),
            "Rotational speed [rpm]": parseFloat(formData.get("Rotational speed [rpm]")),
            "Torque [Nm]": parseFloat(formData.get("Torque [Nm]")),
            "Tool wear [min]": parseFloat(formData.get("Tool wear [min]"))
        };

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Server error occurred");
            }

            const data = await response.json();
            const prediction = data.prediction;

            // Render prediction output
            resultValue.textContent = prediction;
            fsBadge.textContent = data.feature_set_used;

            // Adjust color accents based on failure type
            if (prediction === "No Failure") {
                resultValue.style.color = "var(--accent-green)";
                indicatorBar.style.background = "var(--accent-green)";
                indicatorBar.style.boxShadow = "0 0 15px rgba(16, 185, 129, 0.4)";
            } else {
                resultValue.style.color = "var(--accent-red)";
                indicatorBar.style.background = "var(--accent-red)";
                indicatorBar.style.boxShadow = "var(--glow-red)";
            }

            // Display result panel
            placeholder.classList.add("hidden");
            resultDisplay.classList.remove("hidden");

        } catch (error) {
            alert("Error running telemetry diagnosis:\n" + error.message);
        } finally {
            // Restore button UI state
            spinner.style.display = "none";
            btnSubmit.querySelector(".btn-text").textContent = "Analyze Telemetry";
            btnSubmit.disabled = false;
        }
    });

    // Load best model training statistics
    async function loadMetrics() {
        try {
            const response = await fetch("/api/metrics");
            if (!response.ok) return;

            const data = await response.json();

            // Populate dashboard widgets
            document.getElementById("metric-ba").textContent = (data.balanced_accuracy * 100).toFixed(2) + "%";
            document.getElementById("metric-f1").textContent = (data.f1_macro * 100).toFixed(2) + "%";
            document.getElementById("metric-acc").textContent = (data.accuracy * 100).toFixed(2) + "%";

            // If metadata is present in response, update classifier descriptions
            if (data.feature_set) {
                // Best classifier config names from metadata
                document.getElementById("best-model-name").textContent = formatName(data.model);
                document.getElementById("best-sampler-name").textContent = formatName(data.sampler);
            }
        } catch (e) {
            console.warn("Could not load training metadata metrics:", e);
        }
    }

    function formatName(name) {
        return name.split("_").map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
    }

    // Load metrics on load
    loadMetrics();
});
