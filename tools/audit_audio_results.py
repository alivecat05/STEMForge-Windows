"""Audit completed integration-test audio without downloading or running models.

Usage: runtime\venv\Scripts\python.exe tools\audit_audio_results.py
Reads test-results/*.json; writes audio-audit.json and audio-audit.txt there.
Signal integrity checks do not measure subjective separation quality or SDR.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / 'runtime' / 'test-results'
FOUR_STEMS = {'bass', 'drums', 'other', 'vocals'}
VOCAL_STEMS = {'instrumental', 'vocals'}
EXPECTED = {
    'demucs4': FOUR_STEMS,
    'demucs6': FOUR_STEMS | {'guitar', 'piano'},
    'bsroformer': VOCAL_STEMS,
    'melroformer': VOCAL_STEMS,
    'vocalblend': VOCAL_STEMS,
    'mdxfast': VOCAL_STEMS,
    'drumsplit': {f'main/{s}' for s in FOUR_STEMS}
                 | {f'drum_details/{s}' for s in ['kick', 'snare', 'toms', 'hh', 'ride', 'crash']},
}
ALIASES = {
    'vocal': 'vocals', 'vocals': 'vocals', 'instrumental': 'instrumental',
    'bass': 'bass', 'drum': 'drums', 'drums': 'drums', 'other': 'other',
    'guitar': 'guitar', 'piano': 'piano', 'kick': 'kick', 'snare': 'snare',
    'tom': 'toms', 'toms': 'toms', 'hh': 'hh', 'hihat': 'hh', 'hihats': 'hh',
    'ride': 'ride', 'crash': 'crash',
}


def canonical_stem(name: str, model: str) -> str | None:
    """Use the last stem marker: stage-two names can also contain '(Drums)'."""
    path = PurePosixPath(name.replace('\\', '/'))
    candidates = re.findall(r'\(([^()]+)\)', path.stem)
    normalized = [re.sub(r'[^a-z]', '', part.casefold()) for part in candidates]
    found = [ALIASES[part] for part in normalized if part in ALIASES]
    if not found:
        # Ensemble versions may use plain '_Vocals_' instead of parentheses.
        words = re.split(r'[_\s.\-]+', path.stem.casefold())
        found = [ALIASES[part] for part in words if part in ALIASES]
    if not found:
        return None
    stem = found[-1]
    if model == 'drumsplit':
        return f'{path.parts[0]}/{stem}' if len(path.parts) > 1 else stem
    return stem


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [str(ROOT / 'runtime/ffmpeg/ffprobe.exe'), '-v', 'error',
         '-show_entries', 'format=duration', '-of', 'json', str(path)],
        capture_output=True, check=True, timeout=60,
    )
    duration = float(json.loads(completed.stdout)['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid source duration: {duration}')
    return duration


def signal_metrics(path: Path) -> dict:
    """Stream every decoded sample using libsndfile (WAV/FLAC/MP3 supported)."""
    with sf.SoundFile(str(path)) as audio:
        sample_rate, channels, reported_frames = audio.samplerate, audio.channels, audio.frames
        subtype = audio.subtype
        # PCM full-scale includes its maximum positive quantized value. MP3 can
        # overshoot full-scale after decoding; this statistic is diagnostic only.
        pcm_bits = {'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32, 'PCM_U8': 8, 'PCM_S8': 8}.get(subtype)
        clipping_threshold = 1.0 - 2.0 ** (1 - pcm_bits) if pcm_bits else 1.0
        peak = np.zeros(channels, dtype=np.float64)
        square_sum = np.zeros(channels, dtype=np.float64)
        nonfinite = np.zeros(channels, dtype=np.int64)
        clipped = np.zeros(channels, dtype=np.int64)
        nonzero = np.zeros(channels, dtype=np.int64)
        frames = 0
        for block in audio.blocks(blocksize=65536, dtype='float64', always_2d=True):
            frames += len(block)
            finite = np.isfinite(block)
            nonfinite += np.count_nonzero(~finite, axis=0)
            valid = np.where(finite, block, 0.0)
            magnitude = np.abs(valid)
            peak = np.maximum(peak, np.max(magnitude, axis=0))
            square_sum += np.sum(valid * valid, axis=0)
            nonzero += np.count_nonzero(magnitude > 0, axis=0)
            clipped += np.count_nonzero(magnitude >= clipping_threshold, axis=0)
    samples = frames * channels
    rms = math.sqrt(float(square_sum.sum()) / samples) if samples else 0.0
    return {
        'sample_rate': sample_rate, 'channels': channels, 'subtype': subtype,
        'frames': frames, 'reported_frames': reported_frames, 'samples': samples,
        'seconds': frames / sample_rate, 'finite': not bool(nonfinite.sum()),
        'nonfinite_samples': int(nonfinite.sum()), 'nonzero_signal': bool(nonzero.sum()),
        'nonzero_samples': int(nonzero.sum()), 'peak': float(peak.max(initial=0)),
        'rms': rms, 'rms_dbfs': 20 * math.log10(rms) if rms > 0 else None,
        'clipping_threshold': clipping_threshold,
        'clipped_samples': int(clipped.sum()),
        'clipping_ratio': float(clipped.sum()) / samples if samples else 0.0,
        'per_channel': [
            {'peak': float(peak[c]), 'rms': math.sqrt(float(square_sum[c]) / frames) if frames else 0.0,
             'nonfinite_samples': int(nonfinite[c]), 'nonzero_samples': int(nonzero[c]),
             'clipping_ratio': float(clipped[c]) / frames if frames else 0.0}
            for c in range(channels)
        ],
    }


def audit_report(report: dict, report_path: Path, tolerance: float, duration_cache: dict) -> dict:
    model = report['model']
    errors, warnings, tracks = [], [], []
    result = {key: report.get(key) for key in ['id', 'model', 'format', 'device', 'low_vram', 'source']}
    result['report'] = report_path.name
    result['upstream_report_passed'] = report.get('passed')
    source = Path(report['source'])
    try:
        if source not in duration_cache:
            duration_cache[source] = probe_duration(source)
        source_duration = duration_cache[source]
    except Exception as error:
        source_duration = None
        errors.append(f'Could not probe source: {error}')
    result['source_seconds'] = source_duration
    if not re.fullmatch(r'[0-9a-f]{12}', str(report['id'])):
        result.update(passed=False, errors=['Invalid job id'], warnings=[], tracks=[])
        return result
    output = (ROOT / 'data/jobs' / report['id'] / 'output').resolve()
    declared = report.get('files', [])
    actual_stems = []
    declared_names = []
    for track in declared:
        name = track['name']
        declared_names.append(name.replace('\\', '/'))
        stem = canonical_stem(name, model)
        actual_stems.append(stem)
        entry = {'name': name, 'stem': stem, 'errors': [], 'warnings': []}
        path = (output / name).resolve()
        try:
            if not path.is_relative_to(output):
                raise ValueError('Track path escapes output directory')
            if any(part.startswith('.') for part in path.relative_to(output).parts):
                raise ValueError('Temporary/hidden track included in report')
            if path.suffix.casefold() != '.' + report['format'].casefold():
                entry['errors'].append('Output extension does not match requested format')
            entry.update(signal_metrics(path))
            entry['bytes'] = path.stat().st_size
            if not entry['finite']:
                entry['errors'].append('Decoded samples contain NaN or infinity')
            if not entry['samples'] or not entry['nonzero_signal']:
                entry['errors'].append('Audio is empty or entirely silent')
            if entry['frames'] != entry['reported_frames']:
                entry['errors'].append('Decoded frame count differs from file metadata')
            if entry['channels'] != 2:
                entry['errors'].append('Expected stereo output')
            if source_duration is not None:
                entry['duration_delta_seconds'] = entry['seconds'] - source_duration
                if abs(entry['duration_delta_seconds']) > tolerance:
                    entry['errors'].append(f'Duration differs from source by more than {tolerance}s')
            if entry['clipped_samples']:
                entry['warnings'].append('Samples at/above full scale; clipping ratio is diagnostic, not a perceptual verdict')
            if any(channel['nonzero_samples'] == 0 for channel in entry['per_channel']):
                entry['warnings'].append('At least one channel is entirely silent')
        except Exception as error:
            entry['errors'].append(f'Audio audit failed: {type(error).__name__}: {error}')
        entry['passed'] = not entry['errors']
        errors.extend(f'{name}: {error}' for error in entry['errors'])
        warnings.extend(f'{name}: {warning}' for warning in entry['warnings'])
        tracks.append(entry)
    expected = EXPECTED[model]
    if Counter(actual_stems) != Counter(expected):
        errors.append(f'Stem set/count mismatch: expected {sorted(expected)}, got {actual_stems}')
    if len(declared_names) != len(set(declared_names)):
        errors.append('Report contains duplicate output filenames')
    on_disk = sorted(p.relative_to(output).as_posix() for p in output.rglob('*')
                     if p.is_file() and p.suffix.lower() in {'.wav', '.flac', '.mp3', '.m4a'}
                     and not any(part.startswith('.') for part in p.relative_to(output).parts))
    if sorted(declared_names) != on_disk:
        errors.append('Report filenames do not match all visible audio files on disk')
    result.update(expected_stems=sorted(expected), actual_stems=actual_stems,
                  track_count=len(tracks), tracks=tracks, errors=errors, warnings=warnings,
                  passed=not errors)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=REPORT_DIR)
    parser.add_argument('--duration-tolerance', type=float, default=0.2)
    args = parser.parse_args()
    if not math.isfinite(args.duration_tolerance) or args.duration_tolerance < 0:
        parser.error('--duration-tolerance must be finite and nonnegative')
    audited, skipped, failed_runs, scan_errors = [], [], [], []
    duration_cache = {}
    for path in sorted(args.reports.glob('*.json')):
        if path.name == 'audio-audit.json':
            continue
        try:
            report = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(report, dict) or report.get('model') not in EXPECTED or not report.get('source'):
                continue
            if report.get('status') == 'failed':
                failed_runs.append({
                    'report': path.name, 'id': report.get('id'), 'model': report['model'],
                    'source': report['source'], 'status': 'failed', 'passed': False,
                    'errors': report.get('errors') or [report.get('job', {}).get('error') or 'Model run failed'],
                })
                continue
            if report.get('status') != 'completed':
                skipped.append({'report': path.name, 'status': report.get('status')})
                continue
            result = audit_report(report, path, args.duration_tolerance, duration_cache)
            audited.append(result)
            print(f'{path.name}: {"PASS" if result["passed"] else "FAIL"} ({len(result["tracks"])} tracks)', flush=True)
        except Exception as error:
            scan_errors.append({'report': path.name, 'error': f'{type(error).__name__}: {error}'})
    completed_passed = bool(audited) and all(r['passed'] for r in audited)
    if failed_runs or scan_errors or any(not r['passed'] for r in audited):
        overall = 'FAIL'
    elif skipped or not audited:
        overall = 'INCOMPLETE'
    else:
        overall = 'PASS'
    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Decoded-signal integrity; not subjective audio quality, source leakage, or SDR measurement.',
        'clipping_note': 'Fraction of decoded samples at/above PCM full scale (or 1.0 for compressed/float audio). MP3 decode overshoot is possible. Reported as a warning, not an automatic failure.',
        'duration_tolerance_seconds': args.duration_tolerance,
        'coverage_note': 'Counts cover discovered single-run reports only; completeness of the planned model/format matrix is checked separately.',
        'total_reports': len(audited) + len(skipped) + len(failed_runs) + len(scan_errors),
        'completed_reports': len(audited), 'tracks_audited': sum(len(r['tracks']) for r in audited),
        'passed_reports': sum(r['passed'] for r in audited),
        'completed_audio_passed': completed_passed,
        'failed_run_count': len(failed_runs), 'incomplete_report_count': len(skipped),
        'overall': overall, 'passed': overall == 'PASS',
        'failed_runs': failed_runs, 'skipped_incomplete': skipped, 'scan_errors': scan_errors, 'results': audited,
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    json_path = args.reports / 'audio-audit.json'
    temporary = json_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(json_path)
    lines = [f'Audio integrity audit overall: {overall}',
             f'Completed audio reports: {summary["passed_reports"]}/{summary["completed_reports"]} passed; tracks: {summary["tracks_audited"]}',
             f'Total reports: {summary["total_reports"]}; failed runs: {len(failed_runs)}; incomplete: {len(skipped)}; scan errors: {len(scan_errors)}',
             f'Duration tolerance: {args.duration_tolerance:.3f}s; source duration from FFprobe.',
             summary['coverage_note'], summary['scope'], summary['clipping_note'], '']
    for result in audited:
        lines.append(f'{result["report"]}: {"PASS" if result["passed"] else "FAIL"}; tracks={len(result["tracks"])}')
        for track in result['tracks']:
            if 'peak' in track:
                lines.append(f'  {track["stem"]}: {track["seconds"]:.3f}s, peak={track["peak"]:.6f}, '
                             f'RMS={track["rms"]:.6f}, clip={track["clipping_ratio"]:.6%}, finite={track["finite"]}')
        lines.extend(f'  ERROR: {error}' for error in result['errors'])
    lines.extend(f'FAILED RUN {item["report"]}: {item["errors"]}' for item in failed_runs)
    lines.extend(f'INCOMPLETE {item["report"]}: {item["status"]}' for item in skipped)
    lines.extend(f'ERROR reading {item["report"]}: {item["error"]}' for item in scan_errors)
    (args.reports / 'audio-audit.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Audited {summary["completed_reports"]}/{summary["total_reports"]} reports / {summary["tracks_audited"]} tracks; overall={overall}', flush=True)
    return {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[overall]


if __name__ == '__main__':
    raise SystemExit(main())
