"""Real audio integration checks against a running local STEMForge server."""
import argparse
import io
import json
import pathlib
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runtime' / 'test-results'
REPORT.mkdir(parents=True, exist_ok=True)
BASE = 'http://127.0.0.1:8765'
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
EXPECTED = {'demucs4': 4, 'demucs6': 6, 'bsroformer': 2, 'melroformer': 2,
            'vocalblend': 2, 'mdxfast': 2, 'drumsplit': 10}

def fetch(path, data=None, headers=None):
    with HTTP.open(urllib.request.Request(BASE + path, data=data, headers=headers or {}), timeout=120) as response:
        return response.status, dict(response.headers), response.read()

def probe(path):
    result = subprocess.run([str(ROOT / 'runtime/ffmpeg/ffprobe.exe'), '-v', 'error',
        '-show_entries', 'format=duration:stream=codec_type,sample_rate,channels', '-of', 'json', str(path)],
        capture_output=True, check=True)
    return json.loads(result.stdout)

def run(model, source, fmt, device, low_vram):
    started = time.monotonic()
    _, _, body = fetch('/api/separate', source.read_bytes(), {
        'Content-Type': 'application/octet-stream', 'X-Filename': urllib.parse.quote(source.name),
        'X-Model': model, 'X-Device': device, 'X-Format': fmt, 'X-Low-Vram': str(int(low_vram))})
    job_id = json.loads(body)['id']
    tag = f'{model}-{fmt}-{device}' + ('-low' if low_vram else '')
    record = {'model': model, 'format': fmt, 'device': device, 'low_vram': low_vram,
              'source': str(source), 'id': job_id, 'status': 'running'}
    report_path = REPORT / f'{tag}.json'
    report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'START {tag} {job_id}', flush=True)
    last_log = ''
    while True:
        _, _, body = fetch(f'/api/jobs/{job_id}')
        state = json.loads(body)
        job = state['job']
        if job['logs'] and job['logs'][-1] != last_log:
            last_log = job['logs'][-1]
            print(f'{tag}: {last_log[-400:]}', flush=True)
        if job['status'] in ('completed', 'failed'):
            break
        time.sleep(10)
    record.update(status=job['status'], seconds=round(time.monotonic()-started, 2), job=job, files=state['files'])
    errors = []
    checks = []
    if job['status'] == 'completed':
        if len(state['files']) != EXPECTED[model]:
            errors.append(f'Expected {EXPECTED[model]} tracks, got {len(state["files"])}')
        duration = float(probe(source)['format']['duration'])
        for track in state['files']:
            path = ROOT / 'data/jobs' / job_id / 'output' / track['name']
            details = probe(path)
            actual_duration = float(details['format']['duration'])
            if abs(actual_duration-duration) > .2:
                errors.append(f'{track["name"]}: duration mismatch {actual_duration} / {duration}')
            audio = [s for s in details['streams'] if s['codec_type']=='audio']
            if not audio or audio[0]['channels'] != 2:
                errors.append(f'{track["name"]}: missing stereo audio')
            decoded = subprocess.run([str(ROOT/'runtime/ffmpeg/ffmpeg.exe'), '-v','error','-i',str(path),'-f','null','-'],capture_output=True)
            if decoded.returncode or decoded.stderr:
                errors.append(f'{track["name"]}: decode error {decoded.stderr.decode(errors="replace")}')
            signal, _ = sf.read(path, always_2d=True)
            if not np.isfinite(signal).all():
                errors.append(f'{track["name"]}: non-finite samples')
            peak = float(np.max(np.abs(signal)))
            rms = float(np.sqrt(np.mean(signal ** 2)))
            if peak == 0:
                errors.append(f'{track["name"]}: silent track')
            url = urllib.parse.quote(track['url'], safe='/')
            code, headers, piece = fetch(url, headers={'Range': 'bytes=0-255'})
            with path.open('rb') as f:
                expected = f.read(256)
            if code != 206 or piece != expected:
                errors.append(f'{track["name"]}: playback range mismatch')
            checks.append({'name':track['name'], 'seconds':actual_duration, 'bytes':path.stat().st_size,
                           'audio':audio, 'decodes':decoded.returncode == 0, 'range_status':code,
                           'peak':peak, 'rms':rms})
        _, _, archive = fetch(f'/api/jobs/{job_id}/archive.zip')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            if z.testzip() is not None or set(z.namelist()) != {t['name'] for t in state['files']}:
                errors.append('ZIP content mismatch or CRC failure')
        record['zip_bytes'] = len(archive)
    else:
        errors.append(job.get('error') or 'Separation failed')
    logfile = ROOT / 'data/jobs' / job_id / 'engine.log'
    logs = logfile.read_text(encoding='utf-8', errors='replace') if logfile.exists() else '\n'.join(job['logs'])
    record.update(checks=checks, errors=errors, passed=not errors,
                  cuda_logged='setting Torch device to CUDA' in logs,
                  cpu_fallback='GPU 推理失败' in '\n'.join(job['logs']))
    report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'END {tag} passed={record["passed"]} tracks={len(state["files"])} seconds={record["seconds"]}', flush=True)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=pathlib.Path)
    parser.add_argument('--models', nargs='+', default=list(EXPECTED), choices=list(EXPECTED))
    parser.add_argument('--format', default='WAV', choices=['WAV','FLAC','MP3'])
    parser.add_argument('--device', default='gpu', choices=['gpu','cpu','auto'])
    parser.add_argument('--low-vram', action='store_true')
    args = parser.parse_args()
    results = [run(model,args.source,args.format,args.device,args.low_vram) for model in args.models]
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)
