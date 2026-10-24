"""Format, CPU, automatic-device, low-memory and bad-input integration checks."""
import argparse
import json
import math
import pathlib
import subprocess
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=pathlib.Path, help='Existing audio file used to create a 10-second test clip')
    parser.add_argument('--start', type=float, default=35,
                        help='Clip start time in seconds (default: 35; must be nonnegative)')
    args = parser.parse_args()
    original = args.source.expanduser().resolve()
    if not original.is_file():
        parser.error(f'Source audio file does not exist: {original}')
    if not math.isfinite(args.start) or args.start < 0:
        parser.error('--start must be a finite, nonnegative number')

    # Delay engine dependencies until after argument validation so --help works
    # without installing inference libraries or contacting a running server.
    from test_all_models import ROOT, REPORT, EXPECTED, HTTP, BASE, fetch, run, probe
    ffmpeg = ROOT / 'runtime/ffmpeg/ffmpeg.exe'
    if not ffmpeg.is_file():
        parser.error(f'Project FFmpeg is missing; run install.cmd first: {ffmpeg}')
    duration = float(probe(original)['format']['duration'])
    if not math.isfinite(duration) or duration < args.start + 10:
        parser.error(f'Source is {duration:.3f}s long; --start {args.start:g} requires at least {args.start + 10:g}s')

    # Keep generated sources for later audio audits. A unique project-local folder
    # prevents overwriting the supplied file or input clips from earlier runs.
    clip_dir = pathlib.Path(tempfile.mkdtemp(prefix='extra-mode-inputs-', dir=REPORT))
    source = clip_dir / 'test-song-10s.wav'
    subprocess.run([str(ffmpeg), '-v', 'error', '-ss', str(args.start), '-i', str(original),
                    '-t', '10', '-vn', '-ac', '2', '-ar', '44100', '-c:a', 'pcm_s16le',
                    '-y', str(source)], check=True)
    clip_duration = float(probe(source)['format']['duration'])
    if abs(clip_duration - 10) > 0.02:
        parser.error(f'Expected a 10-second clip, but FFmpeg produced {clip_duration:.3f}s')
    print(f'Test clip: {source} (10 seconds, starting at {args.start:g}s)', flush=True)

    results = []
    for fmt in ('FLAC', 'MP3'):
        for model in EXPECTED:
            results.append(run(model, source, fmt, 'gpu', False))
    results.append(run('bsroformer', source, 'WAV', 'gpu', True))
    results.append(run('mdxfast', source, 'WAV', 'cpu', False))
    for model, suffix, codec in [('demucs6', '.m4a', 'aac'), ('mdxfast', '.flac', 'flac')]:
        converted = source.with_suffix(suffix)
        subprocess.run([str(ffmpeg), '-v', 'error', '-i', str(source), '-c:a', codec,
                        '-y', str(converted)], check=True)
        results.append(run(model, converted, 'WAV', 'auto', False))

    # A valid-looking extension with invalid content must fail without breaking the worker.
    _, _, body = fetch('/api/separate', b'not an audio file', {'Content-Type': 'application/octet-stream',
        'X-Filename': 'invalid-test.mp3', 'X-Model': 'mdxfast', 'X-Device': 'gpu', 'X-Format': 'WAV'})
    job_id = json.loads(body)['id']
    for _ in range(120):
        _, _, body = fetch(f'/api/jobs/{job_id}')
        state = json.loads(body)
        if state['job']['status'] in ('completed', 'failed'):
            break
        time.sleep(2)
    assert state['job']['status'] == 'failed', state
    assert not state['files'], state['files']
    _, _, body = fetch('/api/info')
    assert len(json.loads(body)['models']) == 7
    (REPORT / 'invalid-input.json').write_text(json.dumps({'passed': True, 'job_id': job_id, 'state': state},
                                                        ensure_ascii=False, indent=2), encoding='utf-8')
    with HTTP.open(urllib.request.Request(BASE + f'/api/jobs/{job_id}', method='DELETE'), timeout=30) as response:
        assert response.status == 200
    # Verify the queue still accepts and completes a good task after that failure.
    results.append(run('demucs4', source, 'WAV', 'auto', False))
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
