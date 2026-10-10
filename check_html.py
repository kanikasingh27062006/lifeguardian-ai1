content = open('lifeguardian/frontend/index.html', encoding='utf-8').read()

checks = [
  # ── Core fading bug ──
  ('overlay bug removed (no display:block!important)',
    '#overlay{display:block!important}' not in content),

  # ── Banner ──
  ('banner height clamped to 28px',
    'height:28px;min-height:28px' in content),
  ('banner text truncates (no wrap)',
    'white-space:nowrap' in content and 'proto-banner-text' in content),
  ('banner .hidden uses display:none!important',
    'display:none!important' in content),
  ('proto-close button wired to closeBanner()',
    'id="proto-close"' in content and 'onclick="closeBanner()"' in content),
  ('proto-banner-text is a span element',
    'id="proto-banner-text"' in content),

  # ── Mobile stacking (banner=28, lang=32, topbar=42) ──
  ('mob-topbar starts at top:60px on mobile',
    'top:60px' in content),
  ('main padding-top:102px with banner',
    'padding-top:102px' in content),
  ('main padding-top:74px without banner',
    'padding-top:74px' in content),
  ('mob-topbar.banner-hidden top:32px mobile',
    'mob-topbar.banner-hidden{top:32px}' in content),
  ('mob-topbar.banner-hidden top:36px desktop',
    'mob-topbar.banner-hidden{top:36px}' in content),
  ('lang-bar height:32px on mobile',
    'height:32px' in content),

  # ── closeBanner() applies to all fixed elements ──
  ('closeBanner targets #mob-topbar',
    "'#mob-topbar'" in content and 'closeBanner' in content),
  ('closeBanner targets #main',
    "'#main'" in content and 'closeBanner' in content),
  ('closeBanner targets #lang-bar',
    "'#lang-bar'" in content and 'closeBanner' in content),

  # ── Language persistence ──
  ('language saved to sessionStorage',
    "sessionStorage.setItem('lg_lang'" in content),
  ('language restored on load',
    "sessionStorage.getItem('lg_lang')" in content),
  ('banner-dismissed stored in session',
    "sessionStorage.getItem('bannerDismissed')" in content),

  # ── CSS hygiene ──
  ('no duplicate #main{margin-left:0} rule',
    content.count('#main{margin-left:0') == 1),
  ('overlay display:none by default',
    '#overlay{display:none;' in content),
  ('main has overflow-x:hidden',
    'overflow-x:hidden' in content),
  ('mob-nav labels truncate properly',
    'mob-nav-btn span' in content and 'text-overflow:ellipsis' in content),

  # ── Functionality ──
  ('setRiskActivity updates act-btn visual state',
    'classList.toggle' in content and 'setRiskActivity' in content),
]

all_ok = True
for name, result in checks:
    status = 'OK' if result else 'FAIL'
    if not result: all_ok = False
    print(f'  [{status}] {name}')

print()
if all_ok:
    print('All checks passed!')
else:
    print('Some checks FAILED - review FAIL lines above.')
lines = content.count('\n') + 1
print(f'Total lines: {lines}')
