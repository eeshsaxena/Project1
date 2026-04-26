import urllib.request, json
try:
    with urllib.request.urlopen('http://127.0.0.1:5000/api/corpora', timeout=10) as r:
        data = json.loads(r.read())
        print(f'OK - {len(data)} corpora found')
        for c in data:
            print(f'  {c["name"]} ({c["docs"]} docs, {c["queries"]} queries)')
except Exception as e:
    print(f'FAILED: {e}')
