import sys, os, ast, csv, json, pickle

sys.stdout.reconfigure(encoding='utf-8')

checks = []
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 1. Syntax checks ──────────────────────────────────────────────────
files = [
    'lifeguardian/backend/main.py',
    'lifeguardian/backend/ml/train.py',
    'lifeguardian/backend/ml/predict.py',
    'lifeguardian/backend/ml/assistant.py',
    'lifeguardian/backend/ml/pure_rf.py',
    'lifeguardian/backend/ml/pipeline.py',
    'lifeguardian/backend/ml/generate_data.py',
]
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            src = fh.read()
        ast.parse(src)
        checks.append((f'syntax {os.path.basename(f)}', 'PASS'))
    except Exception as e:
        checks.append((f'syntax {os.path.basename(f)}', f'FAIL: {e}'))

# ── 2. Dataset ─────────────────────────────────────────────────────────
csv_path = 'lifeguardian/data/synthetic_activity_data.csv'
rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
checks.append(('dataset rows == 3000', 'PASS' if len(rows) == 3000 else f'FAIL: {len(rows)}'))
classes = set(r['risk_category'] for r in rows)
checks.append(('dataset classes', 'PASS' if classes == {'low','medium','high'} else f'FAIL: {classes}'))
checks.append(('no scenario leakage', 'PASS' if 'scenario' not in rows[0] else 'FAIL: scenario present'))

# ── 3. Model files ─────────────────────────────────────────────────────
for f in ['lifeguardian/models/pipeline.pkl', 'lifeguardian/models/metrics.json']:
    exists = os.path.exists(f)
    checks.append((f'model file {os.path.basename(f)}', 'PASS' if exists else 'FAIL: missing'))

# ── 4. Metrics ─────────────────────────────────────────────────────────
m = json.load(open('lifeguardian/models/metrics.json'))
checks.append(('accuracy > 0.85', 'PASS' if m['accuracy'] > 0.85 else f'FAIL: {m["accuracy"]}'))
checks.append(('f1 > 0.85',       'PASS' if m['f1_weighted'] > 0.85 else f'FAIL: {m["f1_weighted"]}'))
checks.append(('has confusion_matrix', 'PASS' if m.get('confusion_matrix') else 'FAIL'))
checks.append(('has per_class',        'PASS' if m.get('per_class') else 'FAIL'))
checks.append(('has disclaimer',       'PASS' if m.get('disclaimer') else 'FAIL'))

# ── 5. Prediction module ───────────────────────────────────────────────
sys.path.insert(0, 'lifeguardian/backend/ml')
from predict import predict, is_model_ready
checks.append(('model_ready', 'PASS' if is_model_ready() else 'FAIL'))

r = predict('inactive', 45, 5, 14)
checks.append(('predict inactive→high', 'PASS' if r['prediction'] == 'high' else f'FAIL: {r["prediction"]}'))

r2 = predict('walking', 3, 70, 10)
checks.append(('predict walking→low',   'PASS' if r2['prediction'] == 'low' else f'FAIL: {r2["prediction"]}'))

r3 = predict('bad_state', 0, 50, 12)
checks.append(('invalid input graceful', 'PASS' if r3['error'] else 'FAIL: no error returned'))

checks.append(('probabilities returned', 'PASS' if 'low' in r['probabilities'] else 'FAIL'))

# ── 6. Assistant ───────────────────────────────────────────────────────
from assistant import build_response

a = build_response('inactive', 70, 4, 'high', 14, 'en')
checks.append(('assistant EN headline', 'PASS' if a.get('headline') else 'FAIL'))
checks.append(('assistant EN suggestion', 'PASS' if a.get('suggestion') else 'FAIL'))
checks.append(('assistant EN factors > 0', 'PASS' if len(a.get('influencing_factors', [])) > 0 else 'FAIL'))

ah = build_response('inactive', 70, 4, 'high', 14, 'hi')
checks.append(('assistant HI headline', 'PASS' if ah.get('headline') else 'FAIL'))

a_low = build_response('walking', 3, 65, 'low', 10, 'en')
checks.append(('assistant low branch', 'PASS' if a_low.get('risk_level') == 'low' else 'FAIL'))

# ── 7. Frontend ────────────────────────────────────────────────────────
with open('lifeguardian/frontend/index.html', 'r', encoding='utf-8') as fh:
    fe = fh.read()

frontend_checks = [
    ('no CDN scripts',              'unpkg.com' not in fe and 'cdn.tailwindcss' not in fe),
    ('has ML predict call',         'api/predict' in fe),
    ('has risk page function',      'pageRisk' in fe),
    ('has AI page function',        'pageAI' in fe),
    ('has probability bars',        'probabilities' in fe),
    ('has Hindi text',              'जोखिम' in fe),
    ('has confusion matrix CSS',    'cm-table' in fe),
    ('has model source badge',      'model-source' in fe),
    ('has assistant rendering',     'renderAssistant' in fe),
    ('html closes correctly',       fe.strip().endswith('</html>')),
]
for name, ok in frontend_checks:
    checks.append((f'frontend {name}', 'PASS' if ok else 'FAIL'))

# ── Summary ────────────────────────────────────────────────────────────
passed = sum(1 for _, v in checks if v == 'PASS')
failed = [(n, v) for n, v in checks if v != 'PASS']

for name, result in checks:
    print(f'  [{result}] {name}')

print()
print(f'Result: {passed}/{len(checks)} checks passed')
if failed:
    print('FAILURES:')
    for n, v in failed:
        print(f'  {n}: {v}')
else:
    print('All checks passed!')
