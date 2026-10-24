"""STEMForge: local-only audio stem separation server (Python standard library only).

Backend separation runs in a separate process using the audio-separator 0.47.0 CLI.
Never expose this service to public interfaces: it is designed for 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
import zipfile
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'app'
DATA = ROOT / 'data'
JOBS_DIR = DATA / 'jobs'
MODELS_DIR = DATA / 'models'
for folder in (JOBS_DIR, MODELS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MODEL_CHOICES = {
    'demucs4': {'label': 'Demucs FT · 4 STEMS', 'files': ['htdemucs_ft.yaml'], 'tracks': 'Vocal / Drums / Bass / Other', 'kind': '4-stem'},
    'demucs6': {'label': 'Demucs 6S · 6 STEMS', 'files': ['htdemucs_6s.yaml'], 'tracks': 'Vocal / Drums / Bass / Guitar / Piano / Other', 'kind': '6-stem'},
    'bsroformer': {'label': 'BS-RoFormer · Vocal', 'files': ['model_bs_roformer_ep_317_sdr_12.9755.ckpt'], 'tracks': 'Vocals / Instrumental', 'kind': '2-stem'},
    'melroformer': {'label': 'MelBand RoFormer · Vocal', 'files': ['model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt'], 'tracks': 'Vocals / Instrumental', 'kind': '2-stem'},
    'vocalblend': {'label': 'Dual RoFormer · Ensemble', 'files': ['model_bs_roformer_ep_317_sdr_12.9755.ckpt', 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt'], 'tracks': 'Vocals / Instrumental · 两模型融合', 'kind': 'ensemble'},
    'mdxfast': {'label': 'MDX-Net HQ · Fast', 'files': ['UVR-MDX-NET-Inst_HQ_5.onnx'], 'tracks': 'Vocals / Instrumental', 'kind': '2-stem'},
    'drumsplit': {'label': 'Drum Split · 2 Stages', 'files': ['htdemucs_ft.yaml', 'MDX23C-DrumSep-aufr33-jarredou.ckpt'], 'tracks': '4 STEMS + kick/snare/toms/hi-hat/ride/crash', 'kind': 'two-stage'},
}
VALID_AUDIO = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.aiff', '.aif', '.wma', '.opus', '.mp4'}
FORMATS = {'WAV', 'FLAC', 'MP3'}
MAX_UPLOAD = 600 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.RLock()
TASKS = queue.Queue()
ACTIVE = {'process': None, 'id': None}
CLI_CODE = 'from audio_separator.utils.cli import main; main()'


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def append_log(job: dict, message: str):
    message = str(message).rstrip()
    if not message:
        return
    with LOCK:
        job['logs'].append(message[-1600:])
        job['logs'] = job['logs'][-100:]
        job['updated'] = now()


def safe_filename(value: str) -> str:
    base = Path(str(value).replace('\\', '/')).name
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', base)
    base = base.strip(' .')[:120]
    return base or 'audio.wav'


def job_public(job: dict) -> dict:
    return {key: val for key, val in job.items() if key not in ('input_path', 'logs') } | {'logs': job['logs'][-45:]}


def save_job(job: dict):
    """Persist finished projects so tracks remain visible after server restart."""
    folder = JOBS_DIR / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in job.items() if k != 'input_path'}
    temp = folder / 'job.json.tmp'
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(folder / 'job.json')


def restore_jobs():
    for file in sorted(JOBS_DIR.glob('*/job.json'), key=lambda p: p.stat().st_mtime)[-50:]:
        try:
            data = json.loads(file.read_text(encoding='utf-8'))
            if data.get('id') != file.parent.name or data.get('model') not in MODEL_CHOICES:
                continue
            if data.get('status') in ('running', 'queued'):
                data['status'] = 'failed'
                data['step'] = '上次服务关闭导致任务中断'
                data['error'] = '请重新提交音频以继续分离。'
            data.setdefault('logs', [])
            suffix = Path(data.get('filename') or 'audio.wav').suffix.lower()
            data['input_path'] = str(file.parent / ('input' + suffix))
            JOBS[data['id']] = data
        except Exception:
            logging.exception('Could not restore job from %s', file)


def files_for(job: dict) -> list[dict]:
    output = JOBS_DIR / job['id'] / 'output'
    if not output.is_dir():
        return []
    tracks = []
    for p in sorted(output.rglob('*')):
        if p.is_file() and not any(part.startswith('.') for part in p.relative_to(output).parts) and p.suffix.lower() in {'.wav', '.flac', '.mp3', '.m4a'}:
            rel = p.relative_to(output).as_posix()
            tracks.append({'name': rel, 'size': p.stat().st_size, 'url': f'/api/audio/{job["id"]}/{rel}'})
    return tracks


def make_env(device: str):
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['AUDIO_SEPARATOR_MODEL_DIR'] = str(MODELS_DIR)
    env['HF_HOME'] = str(DATA / 'hf_cache')
    env['TORCH_HOME'] = str(DATA / 'torch_cache')
    env['XDG_CACHE_HOME'] = str(DATA / 'cache')
    env['OMP_NUM_THREADS'] = str(max(1, min(8, os.cpu_count() or 4)))
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    return env


def launch_cli(job: dict, model: str, source: Path | None, output: Path, device: str, download_only=False) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    names = list(model) if isinstance(model, (list, tuple)) else [model]
    cli_code = CLI_CODE
    if os.name == 'nt' and device == 'cpu' and names[0].endswith('.onnx') and not download_only:
        cli_code = 'from app.cpu_mdx_cli import main; main()'
    cmd = [sys.executable, '-u', '-c', cli_code, '--model_filename', names[0],
           '--model_file_dir', str(MODELS_DIR), '--output_dir', str(output)]
    if len(names) > 1 and not download_only:
        cmd += ['--extra_models', *names[1:], '--ensemble_algorithm', 'avg_fft']
    if download_only:
        cmd += ['--download_model_only']
    else:
        cmd += ['--output_format', job['format']]
        if job['format'] == 'MP3':
            cmd += ['--output_bitrate', '320k']
        cmd += ['--demucs_shifts', '1', '--mdxc_batch_size', '1', '--mdx_batch_size', '1']
        if job.get('low_vram'):
            cmd += ['--mdxc_segment_size', '128', '--mdxc_override_model_segment_size']
        if model == 'htdemucs_ft.yaml' or model == 'htdemucs_6s.yaml':
            cmd += ['--demucs_segment_size', '7']
        cmd += [str(source)]
    append_log(job, f'[执行] {" + ".join(names)} | {device.upper()} | {"download" if download_only else "separate"}')
    env = make_env(device)
    # Avoid shell=True and only accept model names from the fixed allowlist.
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors='replace',
                            bufsize=1, creationflags=creationflags)
    with LOCK:
        ACTIVE.update(process=proc, id=job['id'])
    try:
        with (JOBS_DIR / job['id'] / 'engine.log').open('a', encoding='utf-8') as full_log:
            for line in iter(proc.stdout.readline, ''):
                full_log.write(line)
                full_log.flush()
                append_log(job, line)
        proc.wait()
        append_log(job, f'[返回码] {proc.returncode}')
        return proc.returncode == 0
    finally:
        if proc.stdout:
            proc.stdout.close()
        with LOCK:
            if ACTIVE['id'] == job['id']:
                ACTIVE.update(process=None, id=None)


def model_stage(job, model, source, output):
    device = job['device']
    try:
        ok = launch_cli(job, model, source, output, device, job['type'] == 'download')
        if ok:
            return
        if device == 'auto' or device == 'gpu':
            logs = '\n'.join(job['logs'][-25:]).lower()
            if ('out of memory' in logs or 'cuda error' in logs or 'cuda is not available' in logs or 'no cuda' in logs or 'cudnn' in logs):
                append_log(job, 'GPU 推理失败，尝试自动回退到 CPU。')
                if output.is_dir():
                    for file in output.iterdir():
                        if file.is_file() and file.suffix.lower() in {'.wav', '.mp3', '.flac'}:
                            file.unlink(missing_ok=True)
                if launch_cli(job, model, source, output, 'cpu', job['type'] == 'download'):
                    return
        raise RuntimeError(f'模型 {model} 执行失败，请打开运行日志查看原因。')
    except Exception:
        raise


def find_drums(folder: Path) -> Path:
    for file in folder.iterdir():
        if file.is_file() and 'drums' in file.stem.casefold() and file.suffix.lower() in {'.wav', '.flac', '.mp3'}:
            return file
    raise FileNotFoundError('第一阶段未找到 Drums 轨道，无法开始鼓组细分。')


def perform(job: dict):
    dest = JOBS_DIR / job['id']
    output = dest / 'output'
    output.mkdir(exist_ok=True)
    files = MODEL_CHOICES[job['model']]['files']
    if job['type'] == 'download':
        for idx, model in enumerate(files):
            with LOCK:
                job['step'] = f'正在下载 {idx+1}/{len(files)} 个模型'
            model_stage(job, model, None, output)
        return
    source = Path(job['input_path'])
    if job['model'] == 'vocalblend':
        with LOCK:
            job['step'] = '两模型高质量人声融合'
        model_stage(job, files, source, output)
    elif job['model'] == 'drumsplit':
        with LOCK:
            job['step'] = '阶段 1/2 · 拆分 4 轨'
        model_stage(job, files[0], source, output / 'main')
        drums = find_drums(output / 'main')
        with LOCK:
            job['step'] = '阶段 2/2 · 鼓组细分'
        model_stage(job, files[1], drums, output / 'drum_details')
    else:
        with LOCK:
            job['step'] = '正在分离音轨'
        model_stage(job, files[0], source, output)
    if not files_for(job):
        raise RuntimeError('程序未生成任何音轨，请查看日志。')


def worker():
    while True:
        job = TASKS.get()
        try:
            with LOCK:
                job['status'] = 'running'
                job['updated'] = now()
                job['step'] = '准备执行'
            save_job(job)
            perform(job)
            with LOCK:
                job['status'] = 'completed'
                job['step'] = '模型缓存完成' if job['type'] == 'download' else '分离完成'
        except Exception as error:
            append_log(job, traceback.format_exc()[-2600:])
            with LOCK:
                job['status'] = 'failed'
                job['error'] = str(error)
                job['step'] = '失败'
        finally:
            with LOCK:
                job['updated'] = now()
            save_job(job)
            TASKS.task_done()


def new_job(model: str, fmt: str, device: str, type_: str, filename='', low_vram=False):
    if not isinstance(model, str) or model not in MODEL_CHOICES:
        raise ValueError('未知模型')
    if fmt not in FORMATS:
        raise ValueError('不支持的导出格式')
    if device not in {'auto', 'gpu', 'cpu'}:
        raise ValueError('不支持的设备类型')
    job_id = uuid.uuid4().hex[:12]
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True, exist_ok=False)
    job = {'id': job_id, 'model': model, 'format': fmt, 'device': device,
           'type': type_, 'filename': filename, 'low_vram': bool(low_vram),
           'status': 'queued', 'step': '队列中', 'created': now(), 'updated': now(),
           'logs': ['已加入处理队列，模型缺失时会自动下载。'], 'error': None,
           'input_path': None}
    with LOCK:
        JOBS[job_id] = job
    return job, folder


class Handler(BaseHTTPRequestHandler):
    server_version = 'STEMForge/1.0'

    def log_message(self, fmt, *args):
        logging.info('%s %s', self.address_string(), fmt % args)

    def json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def fail(self, message, code=400):
        self.json({'error': message}, code)

    def path_parts(self):
        return [unquote(s) for s in urlsplit(self.path).path.split('/') if s]

    def get_job(self, job_id):
        with LOCK:
            return JOBS.get(job_id)

    def do_GET(self):
        p = self.path_parts()
        try:
            if not p or p == ['index.html']:
                return self.send_file(APP / 'index.html', mimetypes.types_map.get('.html', 'text/html'), as_attachment=False)
            if p == ['api', 'info']:
                models = []
                for id_, info in MODEL_CHOICES.items():
                    entry = dict(info, id=id_)
                    entry['cache'] = [f'{filename}: ' + ('存在索引文件' if (MODELS_DIR / filename).exists() else '待首次下载') for filename in info['files']]
                    models.append(entry)
                return self.json({'models': models, 'jobs': [job_public(j) for j in list(JOBS.values())[-50:][::-1]],
                                  'root': str(ROOT), 'max_upload_mb': MAX_UPLOAD // 1024 // 1024})
            if len(p) == 3 and p[:2] == ['api', 'jobs']:
                job = self.get_job(p[2])
                if not job:
                    return self.fail('没有找到任务', 404)
                return self.json({'job': job_public(job), 'files': files_for(job)})
            if len(p) == 4 and p[:2] == ['api', 'jobs'] and p[3] == 'archive.zip':
                job = self.get_job(p[2])
                if not job or job['type'] != 'separate':
                    return self.fail('没有找到音轨任务', 404)
                if job['status'] not in {'completed', 'failed'}:
                    return self.fail('任务还未完成', 409)
                output = JOBS_DIR / job['id'] / 'output'
                tracks = files_for(job)
                if not tracks:
                    return self.fail('没有可打包的音轨', 404)
                archive = JOBS_DIR / job['id'] / f'STEMForge_{job["id"]}_STEMS.zip'
                if not archive.exists():
                    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
                        for track in tracks:
                            z.write(output / track['name'], track['name'])
                return self.send_file(archive, 'application/zip', as_attachment=True)
            if len(p) >= 4 and p[:2] == ['api', 'audio']:
                job_id = p[2]
                if not re.fullmatch('[0-9a-f]{12}', job_id) or not self.get_job(job_id):
                    return self.fail('未找到任务', 404)
                rel = Path(*p[3:])
                root = (JOBS_DIR / job_id / 'output').resolve()
                file = (root / rel).resolve()
                if not file.is_relative_to(root) or file.suffix.lower() not in {'.wav', '.flac', '.mp3', '.m4a'}:
                    return self.fail('非法文件路径', 403)
                return self.send_file(file, mimetypes.guess_type(file.name)[0] or 'application/octet-stream', as_attachment=False)
            return self.fail('没有此页面', 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            logging.exception('GET failed')
            self.fail(str(error), 500)

    def read_json(self):
        n = int(self.headers.get('Content-Length') or '0')
        if n < 2 or n > 16 * 1024:
            raise ValueError('请求内容过大或为空')
        data = json.loads(self.rfile.read(n))
        if not isinstance(data, dict):
            raise ValueError('请求内容必须是 JSON 对象')
        return data

    def do_POST(self):
        p = self.path_parts()
        try:
            if p == ['api', 'download']:
                data = self.read_json()
                model = data.get('model')
                job, _ = new_job(model, 'WAV', 'auto', 'download')
                save_job(job)
                TASKS.put(job)
                return self.json({'id': job['id']}, 202)
            if p == ['api', 'separate']:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/octet-stream':
                    return self.fail('上传必须使用二进制音频流', 415)
                origin = self.headers.get('Origin', '')
                if origin and not re.fullmatch(r'https?://(?:127\.0\.0\.1|localhost)(?::\d+)?', origin):
                    return self.fail('禁止跨站调用本地接口', 403)
                model = self.headers.get('X-Model', 'demucs4')
                fmt = self.headers.get('X-Format', 'WAV').upper()
                device = self.headers.get('X-Device', 'auto')
                low_vram = self.headers.get('X-Low-Vram', '0') == '1'
                filename = safe_filename(unquote(self.headers.get('X-Filename', 'audio.wav')))
                if Path(filename).suffix.lower() not in VALID_AUDIO:
                    return self.fail('不支持该音频格式，请上传 WAV/MP3/FLAC/M4A 等文件')
                n = int(self.headers.get('Content-Length') or '0')
                if n <= 0 or n > MAX_UPLOAD:
                    return self.fail('文件为空或超过 600MB 限制', 413)
                job, folder = new_job(model, fmt, device, 'separate', filename, low_vram)
                source = folder / ('input' + Path(filename).suffix.lower())
                job['input_path'] = str(source)
                remaining = n
                with source.open('wb') as f:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                if remaining:
                    with LOCK:
                        JOBS.pop(job['id'], None)
                    shutil.rmtree(folder, ignore_errors=True)
                    return self.fail('文件传输中断', 400)
                save_job(job)
                TASKS.put(job)
                return self.json({'id': job['id']}, 202)
            return self.fail('没有此接口', 404)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.fail(str(error))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            logging.exception('POST failed')
            self.fail(f'内部错误: {error}', 500)

    def do_DELETE(self):
        p = self.path_parts()
        if len(p) == 3 and p[:2] == ['api', 'jobs']:
            job = self.get_job(p[2])
            if not job:
                return self.fail('任务不存在', 404)
            if job['status'] in {'running', 'queued'}:
                return self.fail('正在执行的任务不可删除，请运行完成后再删除。', 409)
            with LOCK:
                JOBS.pop(p[2], None)
            shutil.rmtree(JOBS_DIR / p[2], ignore_errors=True)
            return self.json({'deleted': True})
        self.fail('接口不存在', 404)

    def send_file(self, file: Path, mime='application/octet-stream', as_attachment=False):
        if not file.is_file():
            return self.fail('文件不存在', 404)
        total = file.stat().st_size
        start, end = 0, total-1
        range_header = self.headers.get('Range', '')
        if not range_header.startswith('bytes='):
            range_header = ''
        if range_header.startswith('bytes='):
            match = re.fullmatch(r'(\d*)-(\d*)', range_header[6:])
            if not match or not any(match.groups()):
                return self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            if match.group(1):
                start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), total-1)
            elif match.group(2):
                start = max(0, total - int(match.group(2)))
            if start >= total or end < start:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{total}')
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(206 if range_header else 200)
        self.send_header('Content-Type', mime + ('; charset=utf-8' if mime == 'text/html' else ''))
        self.send_header('Content-Length', str(length))
        self.send_header('Accept-Ranges', 'bytes')
        if range_header:
            self.send_header('Content-Range', f'bytes {start}-{end}/{total}')
        if as_attachment:
            self.send_header('Content-Disposition', f'attachment; filename="{file.name}"')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        with file.open('rb') as f:
            f.seek(start)
            while length > 0:
                chunk = f.read(min(1024 * 256, length))
                if not chunk:
                    break
                self.wfile.write(chunk)
                length -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description='STEMForge local web interface')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    restore_jobs()
    threading.Thread(target=worker, name='stem-worker', daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    srv.daemon_threads = True
    url = f'http://127.0.0.1:{args.port}'
    print(f'STEMForge 正在运行：{url}', flush=True)
    if not args.no_browser:
        threading.Timer(1.1, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('STEMForge 已退出。')
    finally:
        srv.server_close()


if __name__ == '__main__':
    main()
