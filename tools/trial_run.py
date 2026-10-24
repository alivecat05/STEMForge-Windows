"""Submit a generated audio clip to the local server for a real inference check."""
import array
import io
import json
import math
import pathlib
import time
import urllib.request
import wave

root = pathlib.Path(__file__).resolve().parents[1]
rate = 44100
samples = array.array('h')
for i in range(rate * 8):
    t = i / rate
    beat = math.exp(-18 * (t % .5))
    value = .14 * math.sin(2 * math.pi * 220 * t) + .1 * math.sin(2 * math.pi * 440 * t) + .2 * beat * math.sin(2 * math.pi * 65 * t)
    samples.extend([int(value * 32767), int(value * 32767)])
buffer = io.BytesIO()
with wave.open(buffer, 'wb') as wav:
    wav.setnchannels(2)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    wav.writeframes(samples.tobytes())
clip = root / 'data' / 'trial-8seconds.wav'
clip.write_bytes(buffer.getvalue())
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
request = urllib.request.Request('http://127.0.0.1:8765/api/separate', data=buffer.getvalue(), headers={
    'Content-Type': 'application/octet-stream', 'X-Filename': clip.name,
    'X-Model': 'demucs4', 'X-Device': 'gpu', 'X-Format': 'WAV',
})
with opener.open(request, timeout=30) as response:
    task = json.load(response)
(root / 'runtime' / 'trial-job.json').write_text(json.dumps(task), encoding='utf-8')
print(json.dumps(task), flush=True)
last = ''
for _ in range(360):
    with opener.open(f'http://127.0.0.1:8765/api/jobs/{task["id"]}', timeout=30) as response:
        state = json.load(response)
    log = '\n'.join(state['job']['logs'][-4:])
    if log != last:
        print(log, flush=True)
        last = log
    if state['job']['status'] in ('completed', 'failed'):
        (root / 'runtime' / 'trial-result.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(state, ensure_ascii=False), flush=True)
        raise SystemExit(0 if state['job']['status'] == 'completed' and len(state['files']) == 4 else 1)
    time.sleep(5)
raise TimeoutError('Trial did not finish in 30 minutes')
