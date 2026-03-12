"""
InstituteBot — Flask Backend (Weeks 1–10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Install:  pip install flask flask-cors
Run:      python app.py
API:      http://localhost:5000
"""

import re, math, string, uuid
from collections import defaultdict, Counter
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Week 1: FAQ Database ────────────────────────────────────────────────────
FAQ_DB = {
    "timing":      "College timings are 9 AM to 5 PM, Monday to Saturday.",
    "fees":        "The annual fee is ₹85,000 for general category students.",
    "contact":     "Contact the admin office at admin@institute.edu or +91-9000000000.",
    "admission":   "Admissions open in June every year. Visit the admissions office or our website.",
    "hostel":      "Hostel facility is available. Contact the warden at hostel@institute.edu.",
    "exam":        "Exams are held in November (sem 1) and April (sem 2).",
    "timetable":   "Timetables are posted on the notice board and the student portal.",
    "library":     "The library is open 8 AM – 8 PM on weekdays.",
    "scholarship": "Merit-based scholarships are available. Apply via the portal by August.",
    "canteen":     "The canteen is open 8 AM – 6 PM with veg and non-veg options.",
    "principal":   "The principal's office is on the 2nd floor of the main building.",
    "result":      "Results are declared within 3 weeks after exams on the student portal.",
    "holiday":     "The holiday calendar is available on the institute website.",
    "bus":         "College buses run on 12 routes. Email transport@institute.edu for details.",
    "wifi":        "Free Wi-Fi is in all academic buildings. Log in with your student ID.",
}

EXTENDED_FAQ = {
    "What are the college timings?":             FAQ_DB["timing"],
    "How much are the fees?":                    FAQ_DB["fees"],
    "How can I contact the administration?":     FAQ_DB["contact"],
    "When do admissions open?":                  FAQ_DB["admission"],
    "Is hostel facility available?":             FAQ_DB["hostel"],
    "When are the exams scheduled?":             FAQ_DB["exam"],
    "Where can I find the timetable?":           FAQ_DB["timetable"],
    "What are the library hours?":               FAQ_DB["library"],
    "How can I apply for a scholarship?":        FAQ_DB["scholarship"],
    "What are the canteen timings?":             FAQ_DB["canteen"],
    "Where is the principal office?":            FAQ_DB["principal"],
    "When will results be declared?":            FAQ_DB["result"],
    "Where can I find the holiday list?":        FAQ_DB["holiday"],
    "How do I get info about the college bus?":  FAQ_DB["bus"],
    "How do I connect to WiFi?":                 FAQ_DB["wifi"],
}

# ─── Week 2: Preprocessing ───────────────────────────────────────────────────
STOPWORDS = {
    "is","the","a","an","of","in","for","to","what","when","where","how",
    "are","can","i","me","my","do","does","please","tell","know","want",
    "get","about","and","or","on","at","with","this","that","it","its",
    "was","be","have","has","will","would"
}
MISSPELLINGS = {
    "timeing":"timing","timings":"timing","fee":"fees","scolarship":"scholarship",
    "scholership":"scholarship","hostle":"hostel","hostell":"hostel","exams":"exam",
    "admision":"admission","addmission":"admission","timetabel":"timetable",
    "libary":"library","libray":"library","buss":"bus","wify":"wifi","wi-fi":"wifi"
}

def preprocess(q):
    q = q.lower().translate(str.maketrans("","",string.punctuation))
    return [MISSPELLINGS.get(t,t) for t in q.split() if t not in STOPWORDS]

# ─── Week 3: Synonym Dictionary ──────────────────────────────────────────────
SYNONYMS = {
    "fees":        ["fees","fee","tuition","payment","cost","charge","amount","price"],
    "timing":      ["timing","time","hours","open","close","timings"],
    "contact":     ["contact","phone","email","number","reach","call","address"],
    "admission":   ["admission","apply","application","enroll","enrollment","join"],
    "hostel":      ["hostel","dorm","dormitory","accommodation","stay","residence"],
    "exam":        ["exam","examination","test","assessment","paper","evaluation"],
    "timetable":   ["timetable","schedule","class","lecture","period","slot"],
    "library":     ["library","books","reading","study","resource"],
    "scholarship": ["scholarship","grant","stipend","aid","financial","merit"],
    "canteen":     ["canteen","food","cafeteria","lunch","meal","eat","dining"],
    "principal":   ["principal","director","head","management","chancellor"],
    "result":      ["result","marks","grade","score","performance","report"],
    "holiday":     ["holiday","vacation","break","leave","calendar","off"],
    "bus":         ["bus","transport","travel","route","vehicle","commute"],
    "wifi":        ["wifi","internet","network","connection","online"],
}
TOKEN_TO_KEY = {s:k for k,syns in SYNONYMS.items() for s in syns}

# ─── Week 4: TF-IDF ──────────────────────────────────────────────────────────
def build_tfidf(docs):
    tok = [preprocess(d) for d in docs]
    N = len(tok)
    df = defaultdict(int)
    for t in tok:
        for w in set(t): df[w] += 1
    idf = {w: math.log((N+1)/(f+1))+1 for w,f in df.items()}
    mat = []
    for t in tok:
        c = Counter(t); n = max(len(t),1)
        mat.append({w:(v/n)*idf.get(w,0) for w,v in c.items()})
    return mat, idf

def cosine(a,b):
    common = set(a)&set(b)
    if not common: return 0.0
    dot = sum(a[k]*b[k] for k in common)
    ma = math.sqrt(sum(v*v for v in a.values()))
    mb = math.sqrt(sum(v*v for v in b.values()))
    return dot/(ma*mb) if ma and mb else 0.0

FAQ_DOCS    = list(EXTENDED_FAQ.keys())
FAQ_ANSWERS = list(EXTENDED_FAQ.values())
MATRIX, IDF = build_tfidf(FAQ_DOCS)

def retrieve(query, threshold=0.1):
    toks = preprocess(query)
    if not toks: return None, 0.0
    c = Counter(toks); n = max(len(toks),1)
    v = {w:(x/n)*IDF.get(w,0) for w,x in c.items()}
    scores = [cosine(v,dv) for dv in MATRIX]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return (FAQ_ANSWERS[best], round(scores[best],3)) if scores[best]>=threshold else (None, round(scores[best],3))

# ─── Week 5: Intent Classification ───────────────────────────────────────────
INTENT_KW = {
    "admissions":   ["admission","apply","application","enroll","register","join","intake"],
    "exams":        ["exam","examination","test","paper","assessment","marks","result","grade"],
    "timetable":    ["timetable","schedule","class","lecture","slot","period"],
    "hostel":       ["hostel","dorm","accommodation","room","stay","residence","warden"],
    "scholarships": ["scholarship","grant","stipend","financial","merit","waiver"],
    "fees":         ["fee","fees","tuition","payment","cost","amount","charge","pay"],
}

def classify_intent(query):
    toks = preprocess(query)
    sc = defaultdict(int)
    for t in toks:
        for intent,kws in INTENT_KW.items():
            if t in kws: sc[intent] += 1
    return max(sc,key=sc.get) if sc else "general"

# ─── Week 6: Entity Extraction ───────────────────────────────────────────────
COURSE_CODES = ["CS","IT","EC","ME","CE","EE","AI","DS"]

def extract_entities(query):
    ent = {"courses":[],"semesters":[],"years":[]}
    for c in COURSE_CODES:
        if re.search(rf'\b{c}\b', query, re.IGNORECASE):
            ent["courses"].append(c)
    m = re.search(r'\b(sem|semester)\s*(\d{1,2})\b', query, re.IGNORECASE)
    if m: ent["semesters"].append(f"Semester {m.group(2)}")
    m2 = re.search(r'\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+year\b', query, re.IGNORECASE)
    if m2: ent["years"].append(m2.group(0).title())
    return ent

# ─── Week 8: Fallback & Handover ─────────────────────────────────────────────
HANDOVER = {
    "general":      "admin@institute.edu  |  +91-9000000000",
    "admissions":   "admissions@institute.edu  |  +91-9000000001",
    "hostel":       "hostel@institute.edu  |  +91-9000000002",
    "scholarships": "scholarships@institute.edu  |  +91-9000000003",
    "exams":        "exams@institute.edu  |  +91-9000000004",
}

# ─── Session + Analytics store ───────────────────────────────────────────────
SESSIONS  = {}
ANALYTICS = {"total":0, "unresolved":0, "intents":Counter()}

def get_session(sid):
    if sid not in SESSIONS:
        SESSIONS[sid] = {"last_intent":None,"clarif":0}
    return SESSIONS[sid]

# ─── Core pipeline ────────────────────────────────────────────────────────────
def process(query, sid):
    ctx    = get_session(sid)
    toks   = preprocess(query)
    intent = classify_intent(query)
    ent    = extract_entities(query)

    ANALYTICS["total"] += 1
    ANALYTICS["intents"][intent] += 1

    # Synonym shortcut (Week 3)
    for t in toks:
        if t in TOKEN_TO_KEY:
            key = TOKEN_TO_KEY[t]
            if key in FAQ_DB:
                ctx["clarif"] = 0; ctx["last_intent"] = intent
                return {"text":FAQ_DB[key],"intent":intent,"confidence":0.9,"entities":ent}

    # Follow-up resolution (Week 7)
    resolved = query
    triggers = {"when","where","how","what","which","more","details"}
    if len(toks)<=3 and set(toks)&triggers and ctx["last_intent"]:
        resolved = f"{query} {ctx['last_intent']}"

    # TF-IDF retrieval (Week 4)
    answer, conf = retrieve(resolved)
    if answer:
        ctx["clarif"] = 0; ctx["last_intent"] = intent
        return {"text":answer,"intent":intent,"confidence":conf,"entities":ent}

    # Fallback / handover (Week 8)
    ANALYTICS["unresolved"] += 1
    ctx["clarif"] += 1
    if ctx["clarif"] >= 2:
        ctx["clarif"] = 0
        contact = HANDOVER.get(intent, HANDOVER["general"])
        return {"text":f"I'm having trouble with that.\nPlease contact a human advisor:\n📧 {contact}",
                "intent":intent,"confidence":0,"entities":ent,"isHandover":True}

    return {"text":"Could you rephrase that? Try asking about fees, exams, hostel, or admissions.",
            "intent":intent,"confidence":0,"entities":ent,"isClarification":True}

# ─── API Routes ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status":"ok","sessions":len(SESSIONS)})

@app.route("/api/chat", methods=["POST"])
def chat():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    sid   = data.get("session_id") or str(uuid.uuid4())
    if not query:
        return jsonify({"error":"query required"}), 400
    result = process(query, sid)
    result["session_id"] = sid
    return jsonify(result)

@app.route("/api/analytics")
def analytics():
    t = ANALYTICS["total"]
    return jsonify({
        "total":      t,
        "unresolved": ANALYTICS["unresolved"],
        "resolution": round((1 - ANALYTICS["unresolved"]/t)*100) if t else 100,
        "intents":    dict(ANALYTICS["intents"].most_common(7)),
    })

@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True)
    sid  = data.get("session_id","")
    if sid in SESSIONS: del SESSIONS[sid]
    return jsonify({"reset":True})

if __name__ == "__main__":
    print("\n" + "═"*48)
    print("  🎓  InstituteBot Backend")
    print("  ▶   http://localhost:5000")
    print("═"*48 + "\n")
    app.run(debug=True, port=5000)
