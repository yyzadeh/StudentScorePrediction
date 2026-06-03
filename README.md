# 🎓 Student Score Predictor Dashboard

An end-to-end Machine Learning web application that predicts a student's next-term academic performance based on their historical scores and educational profiles. The project utilizes **LightGBM regression models converted to ONNX format**, served via a **FastAPI backend**, and features a responsive, dark-themed, RTL (Farsi) user interface.

The production app is optimized for low-latency inference and is fully deployed on **Railway**.

---

## 🚀 Live Demo & Deployment
* **Backend & Frontend URL:** [[Railway deployment URL](https://studentscoreprediction-production.up.railway.app/)]
* **Environment:** Python 3.11.0+ / FastAPI / ONNX Runtime

---

## 🛠️ Tech Stack

### Machine Learning & Backend
* **Data Preprocessing & Training:** Jupyter Notebook (`.ipynb`), LightGBM, Scikit-Learn
* **Model Format:** ONNX (Open Neural Network Exchange) for highly efficient cross-platform deployment
* **Backend Framework:** FastAPI (Asynchronous, high-performance, automatic OpenAPI docs)
* **Inference Engine:** ONNX Runtime (`onnxruntime` with CPU Execution Provider)
* **Data Validation:** Pydantic v2 (Strict typing, structural validation, and field aliasing)

### Frontend Dashboard
* **Structure & Style:** HTML5, CSS3 Custom Properties (Modern Dark Mode UI, native Blur filters)
* **Typography:** Vazirmatn Font (Optimized for RTL/Persian scannability)
* **Interactivity:** Vanilla JavaScript (Dynamic payload generation, asynchronous Fetch API, state management)

---

## 🧠 System Architecture & How It Works

Because a student gains more semester records as they progress through their academic journey, a single static ML model isn't sufficient. This project implements a **Progressive Multi-Model System**:

1. **Feature Pipeline:** The system ingests 6 static features (`diploma_avg`, `basic_science_avg`, `physiopathology_avg`, `clinical_avg`, `internship_avg`, and `medical_course_type`).
2. **Dynamic Semesters:** Depending on the target term selected (Terms 2 to 7), the system dynamically appends prior term scores as input features.
3. **Z-Score Scaling:** Raw inputs are standardized on-the-fly inside FastAPI using pre-saved standard scaling parameters (`mean` and `std`) computed during the notebook training phase.
4. **ONNX Inference:** The normalized feature array is passed to its respective LightGBM ONNX engine to produce the final predicted score.

---

## 📁 Project Structure

```text
├── ONNX student score predictor/
│   ├── LGBMModel12.onnx               # ONNX Model for Term 2 Prediction
│   ├── LGBMModel21.onnx               # ONNX Model for Term 3 Prediction
│   ├── ...                            # Continuous sequential models up to Term 7
│   ├── scaling_params12.json          # Extracted Mean & Std arrays for normalization
│   └── ...                            # Scaling parameters matching each model
├── web/
│   └── index.html                     # RTL Dashboard UI (Jinja2 Template target)
├── main.py                            # Core FastAPI Web Server & Router
├── requirements.txt                   # Project dependencies
└── README.md                          # Repository documentation
```
