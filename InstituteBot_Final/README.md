# 🎓 InstituteBot — Full-Stack Chatbot

## ▶️ How to Run (2 steps)

### Step 1 — Start the Backend
```bash
cd backend
pip install flask flask-cors
python app.py
```
Backend runs at → **http://localhost:5000**

---

### Step 2 — Open the Frontend
Just open the file directly in your browser:

```
frontend/index.html   ← double-click or drag into Chrome/Firefox
```

That's it. No npm. No Node.js. No build step needed.

---

## 📁 Structure
```
InstituteBot/
├── backend/
│   ├── app.py              ← Flask API (all chatbot logic, Weeks 1–10)
│   └── requirements.txt
├── frontend/
│   └── index.html          ← Complete UI (open in any browser)
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint          | Description           |
|--------|------------------|-----------------------|
| GET    | `/api/health`    | Health check          |
| POST   | `/api/chat`      | Send a message        |
| GET    | `/api/analytics` | Usage statistics      |
| POST   | `/api/reset`     | Clear session         |

## Chat Request
```json
POST /api/chat
{ "query": "What are the fees?", "session_id": "abc" }
```
## Response
```json
{
  "text": "The annual fee is ₹85,000 for general category students.",
  "intent": "fees",
  "confidence": 0.9,
  "entities": { "courses": [], "semesters": [], "years": [] },
  "session_id": "abc"
}
```

---

## ✅ Features (Weeks 1–10)
| Week | Feature |
|------|---------|
| 1  | Basic FAQ (15 topics) |
| 2  | Query preprocessing — stopwords, spelling, lowercase |
| 3  | Synonym-aware matching |
| 4  | TF-IDF cosine similarity retrieval |
| 5  | Intent classification (6 intents) |
| 6  | Entity extraction — courses, semesters, years |
| 7  | Multi-turn context & follow-up resolution |
| 8  | Fallback + human advisor handover |
| 9  | Multi-channel formatting |
| 10 | Analytics — resolution rate, top intents |
