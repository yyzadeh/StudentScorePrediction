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
## 📊 Model Performance & Evaluation (RMSE)

The system's predictive accuracy follows a progressive learning curve based on the depth of the student's academic history. 

In the early stages (`predict_2`), the model exhibits its highest Root Mean Squared Error (RMSE) because it relies on minimal historical data (only Term 1 scores). As the student advances through the semesters, subsequent models ingest a significantly richer chronological dataset containing cumulative prior term performance (`f11`, `f12`, up to `f32`). This expanding feature space allows the higher-term models to capture long-term academic trends and stability, resulting in a much more accurate and reliable prediction with highly optimized RMSE scores in later semesters.

### Evaluation Metrics Table

| Endpoint | Target Prediction | Total Input Features | Root Mean Squared Error (RMSE) |
| :--- | :--- | :---: | :---: |
| `POST /predict_2` | Term 2 Score | 7 Features | **0.510485** |
| `POST /predict_3` | Term 3 Score | 8 Features | **0.227783** |
| `POST /predict_4` | Term 4 Score | 9 Features | **0.252987** |
| `POST /predict_5` | Term 5 Score | 10 Features | **0.165417** |
| `POST /predict_6` | Term 6 Score | 11 Features | **0.214747** |
| `POST /predict_7` | Term 7 Score | 12 Features | **0.192535** |

> 💡 **Key Insight:** The drop in RMSE from **0.510** down to **0.192** demonstrates that the inclusion of sequential historical data dramatically improves the model's capacity to minimize variance and forecast future grades with high precision.

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
