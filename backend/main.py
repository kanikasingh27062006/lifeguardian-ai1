"""
LifeGuardian AI — Backend API v3
=================================
Prompt 3 additions:
  - /api/scenarios/run      — run a scenario, record event, return prediction + response
  - /api/notifications      — GET/POST/PATCH(read)/DELETE notifications
  - /api/events extended    — category filter, risk filter, clear with confirm token
  - /api/qlearn/train       — run Q-learning, return Q-table + reward history
  - /api/qlearn/demo        — show best action for a given state
  - /api/stats/summary      — extended with notification count, latest scenario

PROTOTYPE: Simulated data only. No real sensors, devices, or emergency services.
"""

import datetime, json, math, os, random, sys, uuid
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "..", "data")
EVENTS_FILE   = os.path.join(DATA_DIR, "events.json")
NOTIFS_FILE   = os.path.join(DATA_DIR, "notifications.json")
QTABLE_FILE   = os.path.join(DATA_DIR, "qtable.json")
ML_DIR        = os.path.join(BASE_DIR, "ml")
FRONTEND_DIR  = os.path.join(BASE_DIR, "..", "frontend")

os.makedirs(DATA_DIR, exist_ok=True)

if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

# ── ML imports ─────────────────────────────────────────────────────────
try:
    from predict  import predict as ml_predict, is_model_ready, get_metrics, get_load_error
    from assistant import build_response
    _ml_available = True
except Exception as _ml_err:
    _ml_available = False
    _ml_err_msg   = str(_ml_err)
    def ml_predict(**_):
        return {"prediction":"unknown","probabilities":{},"model_ready":False,"error":_ml_err_msg}
    def is_model_ready(): return False
    def get_metrics(): return None
    def get_load_error(): return _ml_err_msg
    def build_response(*a, **kw):
        return {"headline":"ML unavailable","explanation":_ml_err_msg,
                "suggestion":"","influencing_factors":[],"risk_level":"unknown","disclaimer":""}

# ── app ────────────────────────────────────────────────────────────────
app = FastAPI(title="LifeGuardian AI API", description="Prototype backend — Simulated data only.", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── JSON store helpers ─────────────────────────────────────────────────
def _load(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

def _load_events():    return _load(EVENTS_FILE, [])
def _save_events(ev):  _save(EVENTS_FILE, ev)
def _load_notifs():    return _load(NOTIFS_FILE, [])
def _save_notifs(ns):  _save(NOTIFS_FILE, ns)

# ── heuristic fallback ─────────────────────────────────────────────────
def _heuristic_risk(activity_state, inactivity_duration, movement_intensity, scenario="normal"):
    if scenario == "possible_fall":       return "high"
    if scenario == "unusual_inactivity":  return "medium"
    if scenario == "high_risk_demo":      return "high"
    score = 0
    if activity_state == "inactive":     score += 40
    elif activity_state == "lying_down": score += 15
    if inactivity_duration >= 60:        score += 40
    elif inactivity_duration >= 30:      score += 20
    elif inactivity_duration >= 15:      score += 10
    if movement_intensity < 5:           score += 20
    elif movement_intensity < 15:        score += 10
    if score >= 60: return "high"
    if score >= 25: return "medium"
    return "low"

# ── Pydantic models ────────────────────────────────────────────────────
class ActivityInput(BaseModel):
    activity_state:      str   = Field(..., description="walking|sitting|lying_down|inactive")
    inactivity_duration: float = Field(0.0, ge=0, le=480)
    movement_intensity:  float = Field(50.0, ge=0, le=100)
    scenario:            Optional[str] = Field("normal")
    hour_of_day:         Optional[int] = Field(None, ge=0, le=23)
    notes:               Optional[str] = Field(None)
    event_category:      Optional[str] = Field("activity_analysis")

class PredictInput(BaseModel):
    activity_state:      str   = Field(...)
    inactivity_duration: float = Field(0.0)
    movement_intensity:  float = Field(50.0)
    hour_of_day:         Optional[int] = Field(None, ge=0, le=23)

class AssistantInput(BaseModel):
    activity_state:      str   = Field(...)
    inactivity_duration: float = Field(0.0)
    movement_intensity:  float = Field(50.0)
    ml_prediction:       str   = Field("low")
    hour_of_day:         Optional[int] = Field(None)
    lang:                Optional[str] = Field("en")

class ScenarioRunInput(BaseModel):
    scenario_id: str = Field(...)
    lang:        Optional[str] = Field("en")

class NotifReadInput(BaseModel):
    notif_id: str = Field(...)

class QLearningInput(BaseModel):
    episodes: Optional[int] = Field(500, ge=50, le=2000)
    seed:     Optional[int] = Field(42)

class QDemoInput(BaseModel):
    state_id: Optional[int] = Field(0, ge=0, le=5)

# ══════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ══════════════════════════════════════════════════════════════════════
SCENARIOS = {
    "normal": {
        "id": "normal",
        "name_en": "Normal Activity",
        "name_hi": "सामान्य गतिविधि",
        "description_en": "Typical daily routine — regular movement and engagement",
        "description_hi": "सामान्य दैनिक दिनचर्या — नियमित हलचल और सक्रियता",
        "activity_state": "walking",
        "inactivity_duration": 5,
        "movement_intensity": 65,
        "expected_risk": "low",
        "icon": "activity",
        "steps_en": ["Simulating regular morning routine","Monitoring step count and movement","Movement pattern consistent with normal activity","No unusual patterns detected"],
        "steps_hi": ["सामान्य सुबह की दिनचर्या का अनुकरण","कदमों और हलचल की निगरानी","हलचल का पैटर्न सामान्य गतिविधि के अनुरूप","कोई असामान्य पैटर्न नहीं मिला"],
    },
    "unusual_inactivity": {
        "id": "unusual_inactivity",
        "name_en": "Unusual Inactivity",
        "name_hi": "असामान्य निष्क्रियता",
        "description_en": "Extended period without movement — may warrant attention",
        "description_hi": "लंबे समय तक कोई हलचल नहीं — ध्यान देने की आवश्यकता हो सकती है",
        "activity_state": "inactive",
        "inactivity_duration": 45,
        "movement_intensity": 5,
        "expected_risk": "medium",
        "icon": "clock",
        "steps_en": ["Inactivity threshold reached (45 min)","No movement detected in simulated zone","Generating reminder notification","Suggesting check-in within 15 minutes"],
        "steps_hi": ["निष्क्रियता सीमा पहुँची (45 मिनट)","अनुकरणीय क्षेत्र में कोई हलचल नहीं मिली","अनुस्मारक सूचना तैयार की जा रही है","15 मिनट में जाँच का सुझाव"],
    },
    "possible_fall": {
        "id": "possible_fall",
        "name_en": "Possible Fall Scenario",
        "name_hi": "संभावित गिरावट परिदृश्य",
        "description_en": "Sudden movement spike followed by prolonged immobility",
        "description_hi": "अचानक तेज हलचल के बाद लंबी स्थिरता",
        "activity_state": "lying_down",
        "inactivity_duration": 20,
        "movement_intensity": 8,
        "expected_risk": "high",
        "icon": "alert",
        "steps_en": ["Spike in accelerometer-like signal detected","Subject now stationary for extended period","Pattern consistent with possible fall — illustrative","High-risk alert generated for demonstration"],
        "steps_hi": ["एक्सेलेरोमीटर जैसे संकेत में उछाल मिला","विषय अब लंबे समय से स्थिर","पैटर्न संभावित गिरावट से मिलता जुलता — प्रदर्शनात्मक","प्रदर्शन के लिए उच्च-जोखिम अलर्ट तैयार"],
        "warning_en": "This is an illustrative simulation. No confirmed fall has occurred.",
        "warning_hi": "यह एक प्रदर्शनात्मक अनुकरण है। कोई पुष्टि की हुई गिरावट नहीं हुई।",
    },
    "high_risk_demo": {
        "id": "high_risk_demo",
        "name_en": "High-Risk Demonstration",
        "name_hi": "उच्च-जोखिम प्रदर्शन",
        "description_en": "Combined factors: prolonged inactivity, night-time, minimal movement",
        "description_hi": "संयुक्त कारक: लंबी निष्क्रियता, रात का समय, न्यूनतम हलचल",
        "activity_state": "inactive",
        "inactivity_duration": 90,
        "movement_intensity": 3,
        "expected_risk": "high",
        "icon": "zap",
        "steps_en": ["Simulating extended night-time inactivity","Multiple risk factors combined","No movement detected for 90 simulated minutes","Maximum-concern state — demonstration only"],
        "steps_hi": ["रात के समय विस्तारित निष्क्रियता का अनुकरण","कई जोखिम कारक एक साथ","90 अनुकरणीय मिनट तक कोई हलचल नहीं","अधिकतम चिंता की स्थिति — केवल प्रदर्शन"],
    },
}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status":"ok","service":"LifeGuardian AI","version":"3.0.0","prototype":True,"model_ready":is_model_ready()}

# ── Activity analyze ────────────────────────────────────────────────────
@app.post("/api/activity/analyze")
async def analyze_activity(activity: ActivityInput):
    hour = activity.hour_of_day if activity.hour_of_day is not None else datetime.datetime.now().hour
    if is_model_ready():
        result = ml_predict(activity_state=activity.activity_state, inactivity_duration_minutes=activity.inactivity_duration,
                            movement_intensity=activity.movement_intensity, hour_of_day=hour)
        risk   = result["prediction"] if result["prediction"] != "unknown" else _heuristic_risk(activity.activity_state, activity.inactivity_duration, activity.movement_intensity, activity.scenario)
        source = "ml_model" if result["prediction"] != "unknown" else "heuristic_fallback"
        proba  = result.get("probabilities", {})
    else:
        risk   = _heuristic_risk(activity.activity_state, activity.inactivity_duration, activity.movement_intensity, activity.scenario)
        source = "heuristic_rule_based"
        proba  = {}

    risk_colors = {"low":"#22c55e","medium":"#f59e0b","high":"#ef4444"}
    risk_labels = {"low":"Low Risk","medium":"Medium Risk","high":"High Risk"}
    event_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    record = {
        "id": event_id, "timestamp": timestamp, "category": activity.event_category or "activity_analysis",
        "activity_state": activity.activity_state, "inactivity_duration": activity.inactivity_duration,
        "movement_intensity": activity.movement_intensity, "scenario": activity.scenario or "normal",
        "simulated_risk": risk, "prediction_source": source, "notes": activity.notes,
    }
    events = _load_events(); events.insert(0, record); _save_events(events[:500])
    return {
        "event_id": event_id, "timestamp": timestamp, "simulated_risk": risk,
        "risk_label": risk_labels.get(risk, risk), "risk_color": risk_colors.get(risk, "#94a3b8"),
        "probabilities": proba, "prediction_source": source, "model_ready": is_model_ready(),
        "prototype_disclaimer": "Illustrative assessment only. Not clinically validated.",
        "activity_state": activity.activity_state, "inactivity_duration": activity.inactivity_duration,
        "movement_intensity": activity.movement_intensity, "scenario": activity.scenario, "hour_of_day": hour,
    }

# ── ML predict (clean) ──────────────────────────────────────────────────
@app.post("/api/predict")
async def predict_risk(inp: PredictInput):
    hour = inp.hour_of_day if inp.hour_of_day is not None else datetime.datetime.now().hour
    result = ml_predict(activity_state=inp.activity_state, inactivity_duration_minutes=inp.inactivity_duration,
                        movement_intensity=inp.movement_intensity, hour_of_day=hour)
    return {**result, "prototype_disclaimer": "Illustrative only — trained on synthetic data."}

# ── Model info ──────────────────────────────────────────────────────────
@app.get("/api/model/info")
async def model_info():
    metrics = get_metrics()
    return {
        "model_ready": is_model_ready(), "load_error": get_load_error(), "metrics": metrics,
        "model_type": "Random Forest (pure-Python/NumPy, no scipy)",
        "features": ["activity_state","inactivity_duration_minutes","movement_intensity","hour_of_day"],
        "label_order": ["low","medium","high"], "training_data": "3000 synthetic records — demonstration labels only",
        "disclaimer": "This model is trained on synthetic demonstration data. High accuracy reflects learned labelling rules, not validated medical prediction.",
    }

# ── AI Response Assistant ───────────────────────────────────────────────
@app.post("/api/assistant")
async def ai_assistant(inp: AssistantInput):
    hour = inp.hour_of_day if inp.hour_of_day is not None else datetime.datetime.now().hour
    response = build_response(activity_state=inp.activity_state, inactivity_duration=inp.inactivity_duration,
                              movement_intensity=inp.movement_intensity, ml_prediction=inp.ml_prediction,
                              hour_of_day=hour, lang=inp.lang or "en")
    return {**response, "assistant_type": "rule_based", "no_llm": True}

# ── Scenarios list ──────────────────────────────────────────────────────
@app.get("/api/scenarios")
async def get_scenarios():
    out = []
    for sc in SCENARIOS.values():
        out.append({k: v for k, v in sc.items() if k not in ("steps_en","steps_hi")})
    return {"scenarios": out}

# ── Run scenario ────────────────────────────────────────────────────────
@app.post("/api/scenarios/run")
async def run_scenario(inp: ScenarioRunInput):
    sc = SCENARIOS.get(inp.scenario_id)
    if not sc:
        raise HTTPException(status_code=404, detail=f"Scenario '{inp.scenario_id}' not found")

    activity  = sc["activity_state"]
    duration  = sc["inactivity_duration"]
    intensity = sc["movement_intensity"]
    hour      = datetime.datetime.now().hour

    # ML prediction
    if is_model_ready():
        result = ml_predict(activity_state=activity, inactivity_duration_minutes=duration,
                            movement_intensity=intensity, hour_of_day=hour)
        risk   = result["prediction"] if result["prediction"] != "unknown" else sc["expected_risk"]
        source = "ml_model" if result["prediction"] != "unknown" else "heuristic_fallback"
        proba  = result.get("probabilities", {})
    else:
        risk   = _heuristic_risk(activity, duration, intensity, inp.scenario_id)
        source = "heuristic_rule_based"
        proba  = {}

    # Assistant response
    asst = build_response(activity_state=activity, inactivity_duration=duration, movement_intensity=intensity,
                          ml_prediction=risk, hour_of_day=hour, lang=inp.lang or "en")

    timestamp = datetime.datetime.now().isoformat()
    event_id  = str(uuid.uuid4())

    # Notification
    notif_id = str(uuid.uuid4())
    notif_msgs = {
        "en": {
            "normal":             ("Normal Activity Detected", "System is monitoring regular movement patterns."),
            "unusual_inactivity": ("Inactivity Reminder", "No movement detected for 45 minutes. Consider checking in."),
            "possible_fall":      ("Possible Fall — Please Check", "Unusual pattern detected. Manual check recommended."),
            "high_risk_demo":     ("High-Risk Alert (Demo)", "Multiple risk factors present. Immediate attention recommended."),
        },
        "hi": {
            "normal":             ("सामान्य गतिविधि पहचानी गई", "सिस्टम नियमित हलचल की निगरानी कर रहा है।"),
            "unusual_inactivity": ("निष्क्रियता स्मरण", "45 मिनट से कोई हलचल नहीं। जाँच करने पर विचार करें।"),
            "possible_fall":      ("संभावित गिरावट — जाँच करें", "असामान्य पैटर्न मिला। मैन्युअल जाँच की सिफारिश।"),
            "high_risk_demo":     ("उच्च-जोखिम अलर्ट (प्रदर्शन)", "कई जोखिम कारक मौजूद। तत्काल ध्यान की सिफारिश।"),
        },
    }
    lang_code = inp.lang or "en"
    nm = notif_msgs.get(lang_code, notif_msgs["en"]).get(inp.scenario_id, ("Alert", "Scenario completed."))
    notif = {
        "id": notif_id, "timestamp": timestamp, "title": nm[0], "body": nm[1],
        "risk": risk, "scenario_id": inp.scenario_id, "read": False,
        "simulated": True,
        "disclaimer_en": "Simulated notification — no real alert was sent.",
        "disclaimer_hi": "अनुकरणीय सूचना — कोई वास्तविक अलर्ट नहीं भेजा गया।",
    }
    notifs = _load_notifs(); notifs.insert(0, notif); _save_notifs(notifs[:100])

    # Event record
    event_record = {
        "id": event_id, "timestamp": timestamp, "category": "scenario_run",
        "activity_state": activity, "inactivity_duration": duration,
        "movement_intensity": intensity, "scenario": inp.scenario_id,
        "simulated_risk": risk, "prediction_source": source,
        "notes": f"Scenario: {sc['name_en']}",
    }
    events = _load_events(); events.insert(0, event_record); _save_events(events[:500])

    lang_key = "hi" if lang_code == "hi" else "en"
    return {
        "scenario": sc,
        "steps": sc.get(f"steps_{lang_key}", sc.get("steps_en", [])),
        "warning": sc.get(f"warning_{lang_key}"),
        "simulated_risk": risk,
        "probabilities": proba,
        "prediction_source": source,
        "assistant_response": asst,
        "notification_id": notif_id,
        "event_id": event_id,
        "timestamp": timestamp,
        "model_ready": is_model_ready(),
        "prototype_disclaimer": "Simulated scenario only. No real emergency services contacted.",
    }

# ── Events ─────────────────────────────────────────────────────────────
@app.get("/api/events")
async def get_events(limit: int = 50, offset: int = 0, category: str = "", risk: str = ""):
    events = _load_events()
    if category:
        events = [e for e in events if e.get("category","") == category]
    if risk:
        events = [e for e in events if e.get("simulated_risk","") == risk]
    return {"total": len(events), "events": events[offset: offset + limit]}

@app.delete("/api/events")
async def clear_events():
    _save_events([])
    return {"cleared": True}

# ── Notifications ───────────────────────────────────────────────────────
@app.get("/api/notifications")
async def get_notifications():
    notifs = _load_notifs()
    unread = sum(1 for n in notifs if not n.get("read", False))
    return {"total": len(notifs), "unread": unread, "notifications": notifs}

@app.patch("/api/notifications/{notif_id}/read")
async def mark_read(notif_id: str):
    notifs = _load_notifs()
    for n in notifs:
        if n["id"] == notif_id:
            n["read"] = True
            _save_notifs(notifs)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Notification not found")

@app.patch("/api/notifications/read-all")
async def mark_all_read():
    notifs = _load_notifs()
    for n in notifs:
        n["read"] = True
    _save_notifs(notifs)
    return {"ok": True, "marked": len(notifs)}

@app.delete("/api/notifications/{notif_id}")
async def delete_notification(notif_id: str):
    notifs = _load_notifs()
    new_notifs = [n for n in notifs if n["id"] != notif_id]
    if len(new_notifs) == len(notifs):
        raise HTTPException(status_code=404, detail="Notification not found")
    _save_notifs(new_notifs)
    return {"ok": True}

@app.delete("/api/notifications")
async def clear_notifications():
    _save_notifs([])
    return {"cleared": True}

# ── Stats / dashboard (extended) ───────────────────────────────────────
@app.get("/api/stats/summary")
async def stats_summary():
    events = _load_events()
    notifs = _load_notifs()
    total  = len(events)
    risk_counts     = {"low":0,"medium":0,"high":0}
    activity_counts = {"walking":0,"sitting":0,"lying_down":0,"inactive":0}
    category_counts = {}
    for e in events:
        r = e.get("simulated_risk","low")
        if r in risk_counts: risk_counts[r] += 1
        a = e.get("activity_state","sitting")
        if a in activity_counts: activity_counts[a] += 1
        cat = e.get("category","activity_analysis")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    recent = events[:5]
    now    = datetime.datetime.now()
    hours  = []
    for i in range(23, -1, -1):
        h = now - datetime.timedelta(hours=i)
        intensity = random.randint(10, 80) if 6 <= h.hour <= 22 else random.randint(0, 20)
        hours.append({"hour": h.strftime("%H:00"), "intensity": intensity})

    current_risk  = events[0]["simulated_risk"]  if events else "low"
    current_state = events[0]["activity_state"]  if events else "sitting"
    last_scenario = next((e for e in events if e.get("category") == "scenario_run"), None)
    unread_notifs = sum(1 for n in notifs if not n.get("read", False))

    return {
        "total_events": total, "risk_counts": risk_counts, "activity_counts": activity_counts,
        "category_counts": category_counts, "recent_events": recent, "activity_timeline": hours,
        "current_status": {"state": current_state, "risk": current_risk, "last_updated": now.isoformat()},
        "model_ready": is_model_ready(), "unread_notifications": unread_notifs,
        "last_scenario": last_scenario,
    }

# ══════════════════════════════════════════════════════════════════════
# Q-LEARNING (Educational Simulation)
# ══════════════════════════════════════════════════════════════════════
#
# States  (6): Normal, Slightly Inactive, Inactive, Very Inactive, Lying Still, Possible Fall
# Actions (3): 0=Continue Monitoring, 1=Show Reminder, 2=Recommend Check-In
# Rewards: correct action = +1.0 .. +2.0; wrong = -0.5 .. -1.0
#

Q_STATES = [
    {"id":0,"name_en":"Normal Activity",   "name_hi":"सामान्य गतिविधि",     "risk":"low",    "ideal_action":0},
    {"id":1,"name_en":"Slightly Inactive", "name_hi":"थोड़ा निष्क्रिय",      "risk":"low",    "ideal_action":0},
    {"id":2,"name_en":"Inactive (30 min)","name_hi":"निष्क्रिय (30 मिनट)", "risk":"medium", "ideal_action":1},
    {"id":3,"name_en":"Very Inactive",     "name_hi":"अत्यधिक निष्क्रिय",   "risk":"medium", "ideal_action":1},
    {"id":4,"name_en":"Lying Still",       "name_hi":"शांत लेटे हुए",       "risk":"high",   "ideal_action":2},
    {"id":5,"name_en":"Possible Fall",     "name_hi":"संभावित गिरावट",      "risk":"high",   "ideal_action":2},
]
Q_ACTIONS = [
    {"id":0,"name_en":"Continue Monitoring","name_hi":"निगरानी जारी रखें"},
    {"id":1,"name_en":"Show Reminder",      "name_hi":"अनुस्मारक दिखाएँ"},
    {"id":2,"name_en":"Recommend Check-In", "name_hi":"जाँच की सलाह दें"},
]

def _reward(state_id: int, action_id: int) -> float:
    ideal = Q_STATES[state_id]["ideal_action"]
    diff  = abs(action_id - ideal)
    if diff == 0:   return 2.0 if Q_STATES[state_id]["risk"] == "high" else 1.0
    if diff == 1:   return -0.5
    return -1.0

def _train_q(episodes=500, seed=42, alpha=0.2, gamma=0.85, eps_start=1.0, eps_end=0.05):
    rng = random.Random(seed)
    n_states  = len(Q_STATES)
    n_actions = len(Q_ACTIONS)
    Q = [[0.0]*n_actions for _ in range(n_states)]
    rewards_history = []
    eps_decay = (eps_start - eps_end) / episodes

    for ep in range(episodes):
        eps = max(eps_end, eps_start - ep * eps_decay)
        state = rng.randint(0, n_states - 1)
        ep_reward = 0.0
        for _ in range(6):   # short episode
            if rng.random() < eps:
                action = rng.randint(0, n_actions - 1)
            else:
                action = Q[state].index(max(Q[state]))
            r = _reward(state, action)
            next_state = rng.randint(0, n_states - 1)
            Q[state][action] += alpha * (r + gamma * max(Q[next_state]) - Q[state][action])
            state = next_state
            ep_reward += r
        rewards_history.append(round(ep_reward, 3))

    return Q, rewards_history

@app.post("/api/qlearn/train")
async def qlearn_train(inp: QLearningInput):
    Q, reward_hist = _train_q(episodes=inp.episodes, seed=inp.seed)
    best_actions   = [row.index(max(row)) for row in Q]

    # Smooth reward history to ~50 points for chart
    n = len(reward_hist)
    bucket = max(1, n // 50)
    smoothed = []
    for i in range(0, n, bucket):
        chunk = reward_hist[i:i+bucket]
        smoothed.append({"episode": i+bucket, "reward": round(sum(chunk)/len(chunk), 3)})

    # Save Q-table
    _save(QTABLE_FILE, {"Q": Q, "episodes": inp.episodes, "seed": inp.seed,
                        "best_actions": best_actions, "timestamp": datetime.datetime.now().isoformat()})

    # Event record
    events = _load_events()
    events.insert(0, {
        "id": str(uuid.uuid4()), "timestamp": datetime.datetime.now().isoformat(),
        "category": "qlearn_train", "activity_state": "sitting",
        "inactivity_duration": 0, "movement_intensity": 0, "scenario": "qlearn",
        "simulated_risk": "low", "prediction_source": "q_learning",
        "notes": f"Q-learning trained: {inp.episodes} episodes, seed={inp.seed}",
    })
    _save_events(events[:500])

    return {
        "states":           Q_STATES,
        "actions":          Q_ACTIONS,
        "q_table":          Q,
        "best_actions":     best_actions,
        "reward_history":   smoothed,
        "final_avg_reward": round(sum(reward_hist[-50:])/50, 3) if len(reward_hist) >= 50 else round(sum(reward_hist)/len(reward_hist), 3),
        "episodes":         inp.episodes,
        "seed":             inp.seed,
        "hyperparams":      {"alpha": 0.2, "gamma": 0.85, "epsilon_start": 1.0, "epsilon_end": 0.05},
    }

@app.get("/api/qlearn/info")
async def qlearn_info():
    saved = _load(QTABLE_FILE, None)
    return {
        "states":  Q_STATES,
        "actions": Q_ACTIONS,
        "saved":   saved is not None,
        "saved_episodes": saved.get("episodes") if saved else None,
        "saved_timestamp": saved.get("timestamp") if saved else None,
    }

# ── Serve SPA ──────────────────────────────────────────────────────────
if os.path.isdir(FRONTEND_DIR):
    _assets_dir = os.path.join(FRONTEND_DIR, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/", response_class=FileResponse)
    async def serve_root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/{full_path:path}", response_class=FileResponse)
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RENDER") is None   # disable reload on Render
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
