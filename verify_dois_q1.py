"""Verify DOI cho 4 references MỚI của bài Q1 (+ spot-check 3 ref lõi cũ).
Nguồn sự thật: CrossRef API. Chạy: python verify_dois_q1.py"""
import io, json, sys, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DOIS = {
    # 4 ref MỚI cần verify cho bài Q1
    'Vovk2005_ALRW_book':        '10.1007/b106715',
    'Papadopoulos2002_ICM_ECML': '10.1007/3-540-36755-1_29',
    'Angelopoulos2023_FnTML':    '10.1561/2200000101',
    'Wang2019_NegTransfer_CVPR': '10.1109/CVPR.2019.01155',
    # spot-check 3 ref lõi kế thừa từ JST (đã verify trước — xác nhận lại)
    'Faiman2008':                '10.1002/pip.813',
    'Keddouda2024':              '10.1016/j.apenergy.2024.123064',
    'Kladas2023':                '10.1051/epjpv/2023021',
}

ok = 0
for name, doi in DOIS.items():
    try:
        req = urllib.request.urlopen(
            'https://api.crossref.org/works/' + doi, timeout=15)
        msg = json.loads(req.read())['message']
        title = (msg.get('title') or ['?'])[0][:75]
        year = (msg.get('issued', {}).get('date-parts', [[0]])[0] or [0])[0]
        auth = (msg.get('author') or [{}])[0].get('family', '?')
        print(f'[OK]   {name}: {auth} ({year}) — {title}')
        ok += 1
    except Exception as e:
        print(f'[FAIL] {name} ({doi}): {type(e).__name__}: {e}')

print(f'\n{ok}/{len(DOIS)} verified')
