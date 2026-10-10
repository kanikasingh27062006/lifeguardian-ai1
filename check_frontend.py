import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('lifeguardian/frontend/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

checks = [
    ('DOCTYPE html', '<!DOCTYPE html>' in content),
    ('React CDN', 'react@18' in content),
    ('ReactDOM CDN', 'react-dom@18' in content),
    ('Tailwind CDN', 'cdn.tailwindcss.com' in content),
    ('Recharts CDN', 'recharts' in content),
    ('Babel standalone', 'babel' in content.lower()),
    ('Translation EN dict', 'translations' in content and "en:" in content),
    ('Translation HI dict', "hi:" in content),
    ('LangContext', 'LangContext' in content),
    ('NavContext', 'NavContext' in content),
    ('ActivityContext', 'ActivityContext' in content),
    ('DashboardPage', 'DashboardPage' in content),
    ('ActivityMonitorPage', 'ActivityMonitorPage' in content),
    ('ComingSoonPage', 'ComingSoonPage' in content),
    ('AboutPage', 'AboutPage' in content),
    ('Sidebar component', 'function Sidebar' in content),
    ('Hindi Devanagari text', '\u0921\u0948\u0936\u092c\u094b\u0930\u094d\u0921' in content),
    ('API fetch', 'apiFetch' in content),
    ('ReactDOM.createRoot', 'createRoot' in content),
    ('HTML closes correctly', content.strip().endswith('</html>')),
    ('Script closes', '</script>' in content),
]

all_pass = True
for name, result in checks:
    status = 'PASS' if result else 'FAIL'
    if not result: all_pass = False
    print(f'  [{status}] {name}')

print()
print(f'File size: {len(content)} chars, {content.count(chr(10))} lines')
print('All checks passed!' if all_pass else 'SOME CHECKS FAILED')
