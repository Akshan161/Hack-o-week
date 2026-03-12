# 🎓 InstituteBot — Full-Stack FAQ Chatbot

A full-stack academic FAQ chatbot.  
**Backend:** Python + Flask (Weeks 1–10 pipeline)  
**Frontend:** React + Vite

---

## Project Structure

```
institutebot/
├── backend/
│   ├── app.py            ← Flask API server (all chatbot logic)
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── vite.config.js    ← proxies /api → localhost:5000
    ├── package.json
    └── src/
        ├── main.jsx
        └── App.jsx       ← React UI
```

---

## Quick Start

### 1. Backend (Flask)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Server starts at **http://localhost:5000**

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

App opens at **http://localhost:3000**

> Vite proxies all `/api/*` requests to `localhost:5000`, so no CORS issues in dev.

---

## API Reference

| Method | Endpoint         | Body / Params                         | Description                      |
|--------|-----------------|---------------------------------------|----------------------------------|
| GET    | `/api/health`   | —                                     | Health check                     |
| POST   | `/api/chat`     | `{ query, session_id? }`              | Send a message, get a response   |
| GET    | `/api/analytics`| —                                     | Get usage analytics              |
| POST   | `/api/reset`    | `{ session_id }`                      | Clear session context            |

### Example `/api/chat` response
```json
{
  "text": "Exams are held in November (semester 1) and April (semester 2).",
  "intent": "exams",
  "confidence": 0.872,
  "entities": { "courses": ["CS"], "semesters": ["Semester 5"], "years": [] },
  "session_id": "abc-123"
}
```

---

## Features by Week

| Week | Feature                        |
|------|-------------------------------|
| 1    | Basic FAQ with pattern matching|
| 2    | Query preprocessing            |
| 3    | Synonym-aware retrieval        |
| 4    | TF-IDF cosine similarity       |
| 5    | Intent classification          |
| 6    | Entity extraction (courses, semesters) |
| 7    | Multi-turn context & follow-ups|
| 8    | Fallback & human handover      |
| 9    | Multichannel formatting        |
| 10   | Analytics & improvement suggestions |

---

## Production Build

```bash
# Build frontend
cd frontend && npm run build

# Serve static files from Flask
# Copy frontend/dist/ into backend/static/ and add:
# app.route('/') → send_from_directory('static', 'index.html')
```
