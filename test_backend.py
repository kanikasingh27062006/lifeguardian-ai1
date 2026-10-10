import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'lifeguardian/backend')
from main import app, _compute_simulated_risk, ActivityInput

tests = [
    ('walking',    5,  65, 'normal',             'low'),
    ('inactive',   45,  5, 'unusual_inactivity', 'medium'),
    ('lying_down', 20,  8, 'possible_fall',      'high'),
    ('inactive',   90,  2, 'normal',             'high'),
    ('sitting',     5, 50, 'normal',             'low'),
    ('walking',    10, 70, 'normal',             'low'),
]

all_pass = True
for state, dur, intensity, scenario, expected in tests:
    a = ActivityInput(activity_state=state, inactivity_duration=dur, movement_intensity=intensity, scenario=scenario)
    got = _compute_simulated_risk(a)
    status = 'PASS' if got == expected else 'FAIL'
    if got != expected:
        all_pass = False
    print(f'[{status}] {scenario} / {state} -> expected={expected} got={got}')

print()

# Check route count
routes = [r.path for r in app.routes]
print(f'[PASS] API routes: {routes}')
print()
print('All risk tests passed!' if all_pass else 'SOME TESTS FAILED')
