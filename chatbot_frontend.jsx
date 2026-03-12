"""
Institute FAQ Chatbot — Flask Backend (Weeks 1–10)
Run:  python app.py
API:  POST /api/chat       { "query": "...", "session_id": "..." }
      GET  /api/analytics  
      POST /api/reset       { "session_id": "..." }
      GET  /api/health
"""

import re
import math
import string
import uuid
from collections import defaultdict, Counter
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────
# WEEK 1 — FAQ Database
# ─────────────────────────────────────────────────────────
FAQ_DB = {
    "timing":     "College timings are 9 AM to 5 PM, Monday to Saturday.",
    "fees":       "The annual fee is ₹85,000 for general category students.",
    "contact":    "You can contact the admin office at admin@institute.edu or call +91-9000000000.",
    "admission":  "Admissions open in June every year. Visit the admissions office or our website.",
    "hostel":     "Hostel facility is available. Contact the hostel warden at hostel@institute.edu.",
    "exam":       "Exams are held in November (semester 1) and April (semester 2).",
    "timetable":  "Timetables are posted on the notice board and the student portal.",
    "library":    "The library is open from 8 AM to 8 PM on weekdays.",
    "scholarship":"Merit-based scholarships are available. Apply through the scholarship portal by August.",
    "canteen":    "The canteen is open from 8 AM to 6 PM. It offers vegetarian and non-vegetarian meals.",
    "principal":  "The principal's office is on the 2nd floor of the main building.",
    "result":     "Results are declared within 3 weeks after exams on the student portal.",
    "holiday":    "The academic calendar with all holidays is available on the institute website.",
    "bus":        "College buses operate on 12 routes. Contact transport@institute.edu for details.",
    "wifi":       "Free Wi-Fi is available in all academic buildings. Use your student ID to log in.",
}

EXTENDED_FAQ = {
    "What are the college timings?":                    FAQ_DB["timing"],
    "How much are the fees?":                           FAQ_DB["fees"],
    "How can I contact the administration?":            FAQ_DB["contact"],
    "When do admissions open?":                         FAQ_DB["admission"],
    "Is hostel facility available?":                    FAQ_DB["hostel"],
    "When are the exams scheduled?":                    FAQ_DB["exam"],
    "Where can I find the timetable?":                  FAQ_DB["timetable"],
    "What are the library hours?":                      FAQ_DB["library"],
    "How can I apply for a scholarship?":               FAQ_DB["scholarship"],
    "What are the canteen timings?":                    FAQ_DB["canteen"],
    "Where is the principal office?":                   FAQ_DB["principal"],
    "When will results be declared?":                   FAQ_DB["result"],
    "Where can I find the holiday list?":               FAQ_DB["holiday"],
    "How do I get information about the college bus?":  FAQ_DB["bus"],
    "How do I connect to WiFi?":                        FAQ_DB["wifi"],
}

# ─────────────────────────────────────────────────────────
# WEEK 2 — Preprocessing
# ─────────────────────────────────────────────────────────
STOPWORDS = {
    "is","the","a","an","of","in","for","to","what","when","where","how",
    "are","can","i","me","my","do","does","please","tell","know","want",
    "get","about","and","or","on","at","with","this","that","it","its",
    "was","be","have","has","will","would",
}

MISSPELLINGS = {
    "timeing":"timing","timings":"timing","timmings":"timing",
    "fee":"fees","fess":"fees","scolarship":"scholarship",
    "scholership":"scholarship","hostle":"hostel","hostell":"hostel",
    "exams":"exam","admision":"admission","addmission":"admission",
    "timetabel":"timetable","libary":"library","libray":"library",
    "buss":"bus","wify":"wifi","wi-fi":"wifi",
}

def preprocess(query: str) -> list:
    query = query.lower().translate(str.maketrans("","",string.punctuation))
    tokens = [t for t in query.split() if t not in STOPWORDS]
    return [MISSPELLINGS.get(t, t) for t in tokens]

# ─────────────────────────────────────────────────────────
# WEEK 3 — Synonym Dictionary
# ─────────────────────────────────────────────────────────
SYNONYM_DICT = {
    "fees":       ["fees","fee","tuition","payment","cost","charge","amount","price"],
    "timing":     ["timing","time","hours","open","close","timings"],
    "contact":    ["contact","phone","email","number","reach","call","address"],
    "admission":  ["admission","apply","application","enroll","enrollment","join"],
    "hostel":     ["hostel","dorm","dormitory","accommodation","stay","residence"],
    "exam":       ["exam","examination","test","assessment","paper","evaluation"],
    "timetable":  ["timetable","schedule","class","lecture","period","slot"],
    "library":    ["library","books","reading","study","resource"],
    "scholarship":["scholarship","grant","stipend","aid","financial","merit"],
    "canteen":    ["canteen","food","cafeteria","lunch","meal","eat","dining"],
    "principal":  ["principal","director","head","management","chancellor"],
    "result":     ["result","marks","grade","score","performance","report"],
    "holiday":    ["holiday","vacation","break","leave","calendar","off"],
    "bus":        ["bus","transport","travel","route","vehicle","commute"],
    "wifi":       ["wifi","internet","network","connection","online"],
}

TOKEN_TO_KEY = {syn: key for key, syns in SYNONYM_DICT.items() for syn in syns}

# ─────────────────────────────────────────────────────────
# WEEK 4 — TF-IDF
# ─────────────────────────────────────────────────────────
def build_tfidf(documents):
    tokenized = [preprocess(d) for d in documents]
    N = len(tokenized)
    df = defaultdict(int)
    for toks in tokenized:
        for t in set(toks): df[t] += 1
    idf = {t: math.log((N+1)/(f+1))+1 for t,f in df.items()}
    matrix = []
    for toks in tokenized:
        tf = Counter(toks); total = max(len(toks),1)
        matrix.append({t: (c/total)*idf.get(t,0) for t,c in tf.items()})
    return matrix, idf

def cosine(a, b):
    common = set(a) & set(b)
    if not common: return 0.0
    dot = sum(a[t]*b[t] for t in common)
    ma  = math.sqrt(sum(v**2 for v in a.values()))
    mb  = math.sqrt(sum(v**2 for v in b.values()))
    return dot/(ma*mb) if ma and mb else 0.0

faq_docs    = list(EXTENDED_FAQ.keys())
faq_answers = list(EXTENDED_FAQ.values())
TFIDF_MATRIX, GLOBAL_IDF = build_tfidf(faq_docs)

def query_vector(tokens):
    tf = Counter(tokens); total = max(len(tokens),1)
    return {t: (c/total)*GLOBAL_IDF.get(t,0) for t,c in tf.items()}

def tfidf_retrieve(query, threshold=0.1):
    tokens = preprocess(query)
    if not tokens: return None, 0.0
    qvec = query_vector(tokens)
    scores = [cosine(qvec, dv) for dv in TFIDF_MATRIX]
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_i] >= threshold:
        return faq_answers[best_i], scores[best_i]
    return None, scores[best_i]

# ─────────────────────────────────────────────────────────
# WEEK 5 — Intent Classification
# ─────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "admissions":  ["admission","apply","application","enroll","register","join","intake"],
    "exams":       ["exam","examination","test","paper","assessment","marks","result","grade"],
    "timetable":   ["timetable","schedule","class","lecture","slot","period"],
    "hostel":      ["hostel","dorm","accommodation","room","stay","residence","warden"],
    "scholarships":["scholarship","grant","stipend","financial","merit","free","waiver"],
    "fees":        ["fee","fees","tuition","payment","cost","amount","charge","pay"],
}

def classify_intent(query):
    tokens = preprocess(query)
    scores = defaultdict(int)
    for t in tokens:
        for intent, kws in INTENT_PATTERNS.items():
            if t in kws: scores[intent] += 1
    return max(scores, key=scores.get) if scores else "general"

# ─────────────────────────────────────────────────────────
# WEEK 6 — Entity Extraction
# ─────────────────────────────────────────────────────────
COURSE_CODES = ["CS","IT","EC","ME","CE","EE","AI","DS"]

def extract_entities(query):
    entities = {"courses":[], "semesters":[], "years":[]}
    for code in COURSE_CODES:
        if re.search(rf'\b{code}\b', query, re.IGNORECASE):
            entities["courses"].append(code.upper())
    m = re.search(r'\b(sem|semester)\s*(\d{1,2})\b', query, re.IGNORECASE)
    if m: entities["semesters"].append(f"Semester {m.group(2)}")
    m2 = re.search(r'\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+year\b', query, re.IGNORECASE)
    if m2: entities["years"].append(m2.group(0).title())
    return entities

# ─────────────────────────────────────────────────────────
# WEEK 7 — Conversation Context
# ─────────────────────────────────────────────────────────
class ConversationContext:
    def __init__(self):
        self.history = []
        self.last_intent = None
        self.clarification_counter = 0

    def resolve_followup(self, query):
        tokens = preprocess(query)
        followup_triggers = {"when","where","how","what","which","who","more","details","else"}
        is_short = len(tokens) <= 3
        has_trigger = bool(set(tokens) & followup_triggers)
        if (is_short or has_trigger) and self.last_intent not in (None,"general"):
            return f"{query} {self.last_intent}"
        return query

    def update(self, query, response, intent):
        self.history.append({"role":"user","text":query})
        self.history.append({"role":"bot","text":response})
        self.last_intent = intent

# ─────────────────────────────────────────────────────────
# WEEK 8 — Fallback & Handover
# ─────────────────────────────────────────────────────────
HANDOVER_CONTACTS = {
    "general":     "📧 admin@institute.edu | ☎ +91-9000000000",
    "admissions":  "📧 admissions@institute.edu | ☎ +91-9000000001",
    "hostel":      "📧 hostel@institute.edu | ☎ +91-9000000002",
    "scholarships":"📧 scholarships@institute.edu | ☎ +91-9000000003",
    "exams":       "📧 exams@institute.edu | ☎ +91-9000000004",
}

CLARIFICATION_PROMPTS = [
    "Could you please rephrase your question?",
    "I'm not sure I understood. Are you asking about fees, exams, hostel, or something else?",
]

LOW_CONF = 0.15

# ─────────────────────────────────────────────────────────
# WEEK 10 — Analytics
# ─────────────────────────────────────────────────────────
class ChatAnalytics:
    def __init__(self):
        self.logs = []
        self.intent_counts = Counter()
        self.unresolved = []

    def log(self, query, response, intent, confidence):
        self.logs.append({"query":query,"response":response,"intent":intent,"confidence":confidence})
        self.intent_counts[intent] += 1
        if confidence < LOW_CONF:
            self.unresolved.append(query)

    def report(self):
        return {
            "total":      len(self.logs),
            "unresolved": len(self.unresolved),
            "intents":    dict(self.intent_counts.most_common(7)),
            "suggestions": self._suggestions(),
        }

    def _suggestions(self):
        freq = Counter()
        for q in self.unresolved:
            freq.update(preprocess(q))
        return [t for t,c in freq.most_common(5) if c >= 2]

# ─────────────────────────────────────────────────────────
# Session store (in-memory; swap for Redis/DB in production)
# ─────────────────────────────────────────────────────────
sessions:  dict[str, ConversationContext] = {}
analytics = ChatAnalytics()

def get_session(sid: str) -> ConversationContext:
    if sid not in sessions:
        sessions[sid] = ConversationContext()
    return sessions[sid]

# ─────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────
def process_query(query: str, ctx: ConversationContext):
    tokens = preprocess(query)
    if not tokens:
        return {"text":"Please type a valid question.","intent":"general","confidence":0,"entities":{}}

    intent   = classify_intent(query)
    entities = extract_entities(query)

    # Synonym shortcut (Week 3)
    for t in tokens:
        if t in TOKEN_TO_KEY:
            key = TOKEN_TO_KEY[t]
            if key in FAQ_DB:
                ctx.update(query, FAQ_DB[key], intent)
                analytics.log(query, FAQ_DB[key], intent, 0.9)
                return {"text":FAQ_DB[key],"intent":intent,"confidence":0.9,"entities":entities}

    # TF-IDF retrieval (Week 4)
    resolved = ctx.resolve_followup(query)
    answer, confidence = tfidf_retrieve(resolved)

    if answer:
        ctx.update(query, answer, intent)
        analytics.log(query, answer, intent, confidence)
        return {"text":answer,"intent":intent,"confidence":round(confidence,3),"entities":entities}

    # Fallback (Week 8)
    ctx.clarification_counter += 1
    if ctx.clarification_counter >= 2:
        ctx.clarification_counter = 0
        contact = HANDOVER_CONTACTS.get(intent, HANDOVER_CONTACTS["general"])
        text = f"I'm having trouble with that. Please reach out to a human advisor:\n\n{contact}"
        analytics.log(query, text, intent, 0)
        return {"text":text,"intent":intent,"confidence":0,"entities":entities,"isHandover":True}

    prompt = CLARIFICATION_PROMPTS[(ctx.clarification_counter-1) % len(CLARIFICATION_PROMPTS)]
    analytics.log(query, prompt, intent, 0)
    return {"text":prompt,"intent":intent,"confidence":0,"entities":entities,"isClarification":True}

# ─────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status":"ok","sessions":len(sessions)})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    sid   = data.get("session_id") or str(uuid.uuid4())
    if not query:
        return jsonify({"error":"query is required"}), 400
    ctx    = get_session(sid)
    result = process_query(query, ctx)
    result["session_id"] = sid
    return jsonify(result)

@app.route("/api/analytics")
def get_analytics():
    return jsonify(analytics.report())

@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True)
    sid  = data.get("session_id","")
    if sid in sessions:
        del sessions[sid]
    return jsonify({"reset":True,"session_id":sid})

if __name__ == "__main__":
    print("🎓 InstituteBot backend running on http://localhost:5000")
    app.run(debug=True, port=5000)
