# LifeGuardian AI — Intelligent Emergency Detection and Response System for Elderly People

**College Minor Project · Prompt 3 of 3 Complete**

> **PROTOTYPE** — All activity data is simulated. No real sensors, medical devices, cameras, wearables, or emergency services are connected. Risk predictions are illustrative and NOT clinically validated.

---

## Feature Overview

| Feature | Status |
|---|---|
| Premium bilingual UI (English + Hindi) | ✅ Complete |
| Far-left header language selector | ✅ Complete |
| Dashboard with real state (events, risk, unread alerts) | ✅ Complete |
| Activity Monitor with ML risk prediction | ✅ Complete |
| Risk Analysis page with model metrics & confusion matrix | ✅ Complete |
| AI Response Assistant (rule-based, no LLM) | ✅ Complete |
| Emergency Simulation — 4 interactive scenarios | ✅ Complete |
| In-app Notification system (read/unread/dismiss) | ✅ Complete |
| Event History with filters + confirmation before clear | ✅ Complete |
| Reinforcement Learning Learning Lab (Q-learning) | ✅ Complete |
| Persistent local storage (JSON files) | ✅ Complete |

---

## Architecture

```
lifeguardian/
├── backend/
│   ├── main.py                # FastAPI v3 — all API endpoints
│   ├── requirements.txt
│   └── ml/
│       ├── generate_data.py   # Synthetic dataset generator (run once)
│       ├── train.py           # RF training script (run once, ~60s)
│       ├── predict.py         # Prediction module
│       ├── assistant.py       # Rule-based bilingual response engine
│       ├── pipeline.py        # Pickle-safe encoder classes
│       └── pure_rf.py         # Pure-Python/NumPy Random Forest
├── frontend/
│   └── index.html             # Complete single-file SPA (vanilla JS + CSS)
├── data/
│   ├── events.json            # Persisted event history
│   ├── notifications.json     # Persisted notifications
│   ├── qtable.json            # Saved Q-learning table
│   └── synthetic_activity_data.csv
├── models/
│   ├── pipeline.pkl           # Trained RF model
│   ├── metrics.json           # Test-set metrics
│   └── label_classes.json
└── README.md
```

---

## Installation

```bash
cd lifeguardian/backend
pip install -r requirements.txt
```

**`requirements.txt`** contents: `fastapi uvicorn[standard] pydantic python-multipart aiofiles numpy`

> **Note:** `scipy` is not required. The project uses a pure-Python/NumPy Random Forest implementation to avoid scipy DLL issues on Windows.

---

## Running the Application

### Start the backend (serves both API and frontend):
```bash
cd lifeguardian/backend
python main.py
```

Open **http://localhost:8000** in your browser.

The backend automatically serves `frontend/index.html` as a SPA.

---

## Training the ML Model (already trained — skip unless starting fresh)

```bash
# Step 1: Generate synthetic data
cd lifeguardian/backend
python ml/generate_data.py

# Step 2: Train the Random Forest model (~60 seconds)
python ml/train.py
```

Model saves to `models/pipeline.pkl`. Restart the backend after training.

---

## Using the Application

### Language Selector
The **Language | English | हिन्दी** selector is visible in the dark header bar at the top of every page. Click either button to switch the entire interface.

### Activity Monitor
1. Select an activity state (Walking / Sitting / Lying Down / Inactive)
2. Adjust inactivity duration and movement intensity sliders
3. Click **Analyze with ML** to get a risk prediction
4. Or click a **Quick Scenario** to load preset values

### Emergency Simulation
1. Navigate to **Emergency Sim.**
2. Select one of 4 scenarios: Normal Activity, Unusual Inactivity, Possible Fall, High-Risk Demo
3. Click **Run Scenario**
4. View ML prediction, simulated system steps, AI response, and generated notification

### Notifications
- Generated automatically when scenarios are run
- View in **Event History** page (notification panel at top)
- Mark read / dismiss individual notifications
- "Mark All Read" clears the unread badge count

### Event History
- All activity analyses and scenario runs are recorded
- Filter by event type or risk level
- Click **Clear History** for a confirmation dialog before deleting

### Learning Lab (Q-Learning)
1. Navigate to **Learning Lab**
2. Choose training episodes (100–2000) and a random seed
3. Click **Train Agent** (completes in seconds)
4. View reward history chart, Q-table with best actions
5. Use the **Agent Action Demo** dropdown to see the agent's decision for any state

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Service health + model status |
| POST | `/api/activity/analyze` | Analyze activity, get ML risk, save event |
| POST | `/api/predict` | Pure ML prediction (no event written) |
| GET | `/api/model/info` | Model type, features, metrics |
| POST | `/api/assistant` | Rule-based bilingual AI response |
| GET | `/api/scenarios` | List all 4 scenarios |
| POST | `/api/scenarios/run` | Run scenario → prediction + notification + event |
| GET | `/api/notifications` | All notifications with unread count |
| PATCH | `/api/notifications/{id}/read` | Mark one notification read |
| PATCH | `/api/notifications/read-all` | Mark all read |
| DELETE | `/api/notifications/{id}` | Dismiss one notification |
| DELETE | `/api/notifications` | Clear all notifications |
| GET | `/api/events` | Event history (supports `?category=&risk=&limit=&offset=`) |
| DELETE | `/api/events` | Clear all events |
| GET | `/api/stats/summary` | Dashboard summary incl. unread count |
| POST | `/api/qlearn/train` | Train Q-learning agent, return Q-table |
| GET | `/api/qlearn/info` | Saved Q-table info |

---

## Prototype Limitations

- All activity data is simulated — no real sensors, cameras, or wearables are connected
- The ML model is trained on synthetic demonstration data; high accuracy means the model learned the labelling rules, not real health patterns
- No real emergency alerts, SMS messages, phone calls, emails, or notifications are sent to anyone
- Risk predictions are illustrative and must not be used for real medical decisions
- The Q-learning environment and rewards are educational approximations

---

## Technology Stack

- **Backend:** Python 3.14, FastAPI 0.115+, Uvicorn
- **ML:** Pure-Python/NumPy Random Forest (no scipy dependency)
- **Frontend:** Vanilla JavaScript, inline CSS — no build step required
- **Storage:** Local JSON files (`data/` directory)
- **Languages supported:** English, Hindi (Devanagari)

---

## Free Deployment (Optional)

For local demonstration only — open `http://localhost:8000` on your machine.

To share on a network: run `python main.py` and access via your local IP (`http://192.168.x.x:8000`).

For public deployment: the app can be deployed to [Railway](https://railway.app), [Render](https://render.com), or [Fly.io](https://fly.io) free tiers using a `Procfile: web: python main.py` — no paid services required.
