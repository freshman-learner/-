"""Download the official PyPI wheel in resumable chunks and verify its SHA256."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request

out = Path(__file__).resolve().parent / '_downloads'
out.mkdir(exist_ok=True)
with urllib.request.urlopen('https://pypi.org/pypi/gymnasium-robotics/1.4.2/json', timeout=30) as r:
    data = json.load(r)
item = next(x for x in data['urls'] if x['filename'].endswith('py3-none-any.whl'))
target = out / item['filename']
partial = target.with_suffix('.partial')
size = item['size']
with partial.open('ab') as stream:
    while stream.tell() < size:
        start = stream.tell()
        end = min(size-1, start+1024*1024-1)
        for retry in range(5):
            try:
                request = urllib.request.Request(item['url'], headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206:
                        raise RuntimeError('Server did not honor range request')
                    chunk = response.read()
                if len(chunk) != end-start+1:
                    raise RuntimeError('Incomplete chunk')
                stream.write(chunk)
                stream.flush()
                print(f'{stream.tell()}/{size}', flush=True)
                break
            except Exception:
                if retry == 4:
                    raise
                time.sleep(1)
assert hashlib.sha256(partial.read_bytes()).hexdigest() == item['digests']['sha256']
partial.replace(target)
print('Verified:', target)
