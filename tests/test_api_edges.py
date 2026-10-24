"""Isolated API/worker contract tests; no model downloads or real inference."""
import http.client
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import server


class ApiEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='stemforge-api-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        isolation = patch.multiple(server, DATA=self.root, JOBS_DIR=self.root / 'jobs',
                                  MODELS_DIR=self.root / 'models', JOBS={}, TASKS=queue.Queue())
        isolation.start()
        self.addCleanup(isolation.stop)
        server.JOBS_DIR.mkdir()
        server.MODELS_DIR.mkdir()

    def fetch(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.http.server_port, timeout=5)
        try:
            conn.request(method, path, body, headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def upload(self, headers=None, body=b'RIFF-test-audio'):
        return self.fetch('POST', '/api/separate', body,
                          {'Content-Type': 'application/octet-stream', 'X-Filename': 'song.wav', **(headers or {})})

    def output_job(self, status='completed', type_='separate'):
        job, folder = server.new_job('demucs4', 'WAV', 'cpu', type_, 'song.wav')
        job['status'] = status
        output = folder / 'output'
        output.mkdir()
        track = output / 'voice.wav'
        track.write_bytes(bytes(range(256)))
        return job, track

    def test_metadata_lists_all_models_without_private_input_path(self):
        job, _ = self.output_job()
        job['input_path'] = 'private/input.wav'
        code, headers, body = self.fetch('GET', '/api/info')
        data = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual({m['id'] for m in data['models']}, set(server.MODEL_CHOICES))
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertNotIn('input_path', data['jobs'][0])

    def test_unknown_routes_and_missing_jobs(self):
        for method, url in [('GET', '/not-found'), ('POST', '/not-found'),
                            ('DELETE', '/not-found'), ('GET', '/api/jobs/000000000000'),
                            ('DELETE', '/api/jobs/000000000000'),
                            ('GET', '/api/jobs/000000000000/archive.zip'),
                            ('GET', '/api/audio/bad-id/voice.wav')]:
            with self.subTest(method=method, url=url):
                code, _, body = self.fetch(method, url)
                self.assertEqual(code, 404)
                self.assertIn('error', json.loads(body))

    def test_invalid_upload_options_create_no_jobs(self):
        for headers, expected in [({'X-Model': 'unknown'}, 400),
                                   ({'X-Format': 'OGG'}, 400),
                                   ({'X-Device': 'tpu'}, 400),
                                   ({'X-Filename': 'file.exe'}, 400),
                                   ({'Content-Type': 'text/plain'}, 415),
                                   ({'Origin': 'https://example.invalid'}, 403)]:
            with self.subTest(headers=headers):
                self.assertEqual(self.upload(headers)[0], expected)
                self.assertEqual(server.JOBS, {})
                self.assertEqual(list(server.JOBS_DIR.iterdir()), [])

    def test_empty_oversized_and_invalid_content_length(self):
        self.assertEqual(self.upload(body=b'')[0], 413)
        self.assertEqual(self.upload({'Content-Length': str(server.MAX_UPLOAD + 1)}, body=b'')[0], 413)
        self.assertEqual(self.upload({'Content-Length': '-1'}, body=b'')[0], 413)
        self.assertEqual(self.upload({'Content-Length': 'abc'}, body=b'')[0], 400)
        self.assertEqual(server.JOBS, {})

    def test_upload_defaults_and_filename_normalization(self):
        code, _, body = self.upload({'X-Filename': '..%2F..%2F%E4%B8%AD%E6%96%87%20song.MP3',
                                     'X-Low-Vram': '1', 'Origin': 'http://localhost:8765'})
        self.assertEqual(code, 202)
        job = server.JOBS[json.loads(body)['id']]
        self.assertEqual(job['filename'], '中文 song.MP3')
        self.assertEqual((job['model'], job['format'], job['device']), ('demucs4', 'WAV', 'auto'))
        self.assertTrue(job['low_vram'])
        self.assertTrue(Path(job['input_path']).is_relative_to(server.JOBS_DIR))
        self.assertEqual(Path(job['input_path']).read_bytes(), b'RIFF-test-audio')
        self.assertEqual(server.TASKS.qsize(), 1)
        self.assertTrue((server.JOBS_DIR / job['id'] / 'job.json').is_file())

    def test_download_invalid_json(self):
        for body in [b'', b'{', b'{}', b'{"model":"bad"}', b'x' * 16385]:
            with self.subTest(body=body[:30]):
                self.assertEqual(self.fetch('POST', '/api/download', body)[0], 400)
        self.assertEqual(server.JOBS, {})

    def test_download_nonobject_json_is_client_error(self):
        for body in [b'[]', b'null', b'42', b'"text"']:
            with self.subTest(body=body):
                self.assertEqual(self.fetch('POST', '/api/download', body)[0], 400)

    def test_download_nonstr_model_is_client_error(self):
        for model in [[], {}, 42, True]:
            with self.subTest(model=model):
                self.assertEqual(self.fetch('POST', '/api/download', json.dumps({'model': model}))[0], 400)

    def test_audio_full_and_range_variants(self):
        job, track = self.output_job()
        url = f'/api/audio/{job["id"]}/voice.wav'
        raw = track.read_bytes()
        for range_, start, stop in [(None, 0, 256), ('bytes=0-9', 0, 10),
                                   ('bytes=250-', 250, 256), ('bytes=-10', 246, 256),
                                   ('bytes=250-999', 250, 256), ('bytes=-999', 0, 256)]:
            with self.subTest(range=range_):
                code, headers, body = self.fetch('GET', url, headers={'Range': range_} if range_ else {})
                self.assertEqual(code, 206 if range_ else 200)
                self.assertEqual(body, raw[start:stop])
                self.assertEqual(int(headers['Content-Length']), stop-start)
                if range_:
                    self.assertEqual(headers['Content-Range'], f'bytes {start}-{stop-1}/256')

    def test_audio_unsatisfiable_ranges(self):
        job, _ = self.output_job()
        url = f'/api/audio/{job["id"]}/voice.wav'
        for range_ in ['bytes=256-', 'bytes=10-2', 'bytes=-0', 'bytes=0-1,4-5', 'bytes=abc']:
            with self.subTest(range=range_):
                self.assertEqual(self.fetch('GET', url, headers={'Range': range_})[0], 416)

    def test_empty_byte_range_rejected(self):
        job, _ = self.output_job()
        self.assertEqual(self.fetch('GET', f'/api/audio/{job["id"]}/voice.wav', headers={'Range': 'bytes=-'})[0], 416)

    def test_unknown_range_unit_ignored(self):
        job, _ = self.output_job()
        self.assertEqual(self.fetch('GET', f'/api/audio/{job["id"]}/voice.wav', headers={'Range': 'items=0-1'})[0], 200)

    def test_paths_cannot_escape_output_directory(self):
        job, track = self.output_job()
        (track.parent.parent / 'secret.wav').write_bytes(b'secret')
        for path in ['../secret.wav', '%2e%2e/secret.wav', '%2e%2e%5csecret.wav',
                     '%2e%2e%2fsecret.wav', 'engine.log']:
            with self.subTest(path=path):
                self.assertEqual(self.fetch('GET', f'/api/audio/{job["id"]}/{path}')[0], 403)
        self.assertEqual(self.fetch('GET', f'/api/audio/{job["id"]}/missing.wav')[0], 404)

    def test_nested_unicode_track_and_archive_contents(self):
        job, track = self.output_job()
        folder = track.parent / 'drum_details'
        folder.mkdir()
        (folder / '鼓.wav').write_bytes(b'drum data')
        (folder / 'secret.txt').write_text('not audio')
        code, _, body = self.fetch('GET', f'/api/audio/{job["id"]}/drum_details/%E9%BC%93.wav')
        self.assertEqual((code, body), (200, b'drum data'))
        code, headers, body = self.fetch('GET', f'/api/jobs/{job["id"]}/archive.zip')
        self.assertEqual(code, 200)
        self.assertIn('attachment;', headers['Content-Disposition'])
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(set(z.namelist()), {'voice.wav', 'drum_details/鼓.wav'})
            self.assertEqual(z.read('drum_details/鼓.wav'), b'drum data')

    def test_archive_and_delete_lifecycle(self):
        job, track = self.output_job(status='queued')
        url = f'/api/jobs/{job["id"]}'
        for state in ['queued', 'running']:
            job['status'] = state
            self.assertEqual(self.fetch('GET', url + '/archive.zip')[0], 409)
            self.assertEqual(self.fetch('DELETE', url)[0], 409)
        job['status'] = 'failed'
        self.assertEqual(self.fetch('GET', url + '/archive.zip')[0], 200)
        self.assertEqual(self.fetch('DELETE', url)[0], 200)
        self.assertFalse(track.parent.parent.exists())
        self.assertEqual(self.fetch('GET', url)[0], 404)

    def test_archive_empty_or_download_task(self):
        job, track = self.output_job()
        track.unlink()
        self.assertEqual(self.fetch('GET', f'/api/jobs/{job["id"]}/archive.zip')[0], 404)
        job, _ = self.output_job(type_='download')
        self.assertEqual(self.fetch('GET', f'/api/jobs/{job["id"]}/archive.zip')[0], 404)

    def test_temporary_audio_excluded_from_file_listing_and_archive(self):
        job, track = self.output_job()
        (track.parent / '.input_(Other)_htdemucs_6s.wav.cfrnvtmp.wav').write_bytes(b'incomplete')
        hidden = track.parent / '.staging'
        hidden.mkdir()
        (hidden / 'voice.wav').write_bytes(b'incomplete')
        self.assertEqual([f['name'] for f in server.files_for(job)], ['voice.wav'])
        code, _, body = self.fetch('GET', f'/api/jobs/{job["id"]}/archive.zip')
        self.assertEqual(code, 200)
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            self.assertEqual(z.namelist(), ['voice.wav'])

    def test_persistence_recovery_and_interrupted_job(self):
        job, _ = self.output_job(status='running')
        server.save_job(job)
        server.JOBS.clear()
        server.restore_jobs()
        restored = server.JOBS[job['id']]
        self.assertEqual(restored['status'], 'failed')
        self.assertIn('中断', restored['step'])
        self.assertTrue(Path(restored['input_path']).is_relative_to(server.JOBS_DIR))
        self.assertEqual(len(server.files_for(restored)), 1)

    def test_missing_audio_output_fails(self):
        job, _ = server.new_job('mdxfast', 'WAV', 'cpu', 'separate')
        job['input_path'] = 'dummy.wav'
        with patch.object(server, 'launch_cli', return_value=True):
            with self.assertRaisesRegex(RuntimeError, '未生成任何音轨'):
                server.perform(job)

    def test_gpu_failure_retries_cpu_and_cleans_partial_audio(self):
        job, track = self.output_job()
        job.update(device='gpu', logs=['CUDA out of memory'])
        with patch.object(server, 'launch_cli', side_effect=[False, True]) as launch:
            server.model_stage(job, 'htdemucs_ft.yaml', Path('source.wav'), track.parent)
        self.assertEqual([call.args[4] for call in launch.call_args_list], ['gpu', 'cpu'])
        self.assertFalse(track.exists())

    def test_nongpu_error_does_not_retry(self):
        job, track = self.output_job()
        job.update(device='auto', logs=['invalid model file'])
        with patch.object(server, 'launch_cli', return_value=False) as launch:
            with self.assertRaises(RuntimeError):
                server.model_stage(job, 'htdemucs_ft.yaml', None, track.parent)
        self.assertEqual(launch.call_count, 1)

    def test_missing_drums_blocks_second_stage(self):
        job, _ = server.new_job('drumsplit', 'WAV', 'cpu', 'separate')
        job['input_path'] = 'dummy.wav'
        def fake_launch(job, model, source, output, device, download_only=False):
            output.mkdir(parents=True)
            (output / 'Vocals.wav').write_bytes(b'fake')
            return True
        with patch.object(server, 'launch_cli', side_effect=fake_launch) as launch:
            with self.assertRaisesRegex(FileNotFoundError, 'Drums'):
                server.perform(job)
        self.assertEqual(launch.call_count, 1)

    def test_each_model_downloads_all_declared_files(self):
        for model, info in server.MODEL_CHOICES.items():
            with self.subTest(model=model):
                job, _ = server.new_job(model, 'WAV', 'auto', 'download')
                with patch.object(server, 'launch_cli', return_value=True) as launch:
                    server.perform(job)
                self.assertEqual([call.args[1] for call in launch.call_args_list], info['files'])
                self.assertTrue(all(call.args[2] is None and call.args[5] for call in launch.call_args_list))

    def test_ensemble_passes_both_models_together(self):
        job, folder = server.new_job('vocalblend', 'WAV', 'cpu', 'separate')
        job['input_path'] = str(folder / 'input.wav')
        def fake_launch(job, model, source, output, device, download_only=False):
            (output / 'Vocals.wav').write_bytes(b'fake')
            return True
        with patch.object(server, 'launch_cli', side_effect=fake_launch) as launch:
            server.perform(job)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[1], server.MODEL_CHOICES['vocalblend']['files'])

    def test_drum_second_stage_uses_first_stage_drums(self):
        job, folder = server.new_job('drumsplit', 'FLAC', 'cpu', 'separate')
        job['input_path'] = str(folder / 'input.wav')
        def fake_launch(job, model, source, output, device, download_only=False):
            output.mkdir(parents=True, exist_ok=True)
            (output / ('Drums.flac' if output.name == 'main' else 'Kick.flac')).write_bytes(b'fake')
            return True
        with patch.object(server, 'launch_cli', side_effect=fake_launch) as launch:
            server.perform(job)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(launch.call_args_list[1].args[2], folder / 'output/main/Drums.flac')
        self.assertEqual({t['name'] for t in server.files_for(job)}, {'main/Drums.flac', 'drum_details/Kick.flac'})

    def test_cli_formats_low_vram_ensemble_and_environment(self):
        for fmt in ['WAV', 'FLAC', 'MP3']:
            with self.subTest(format=fmt):
                job, folder = server.new_job('vocalblend', fmt, 'cpu', 'separate', low_vram=True)
                proc = SimpleNamespace(stdout=io.StringIO('engine initialized\n'), returncode=0, wait=lambda: 0)
                with patch.object(server.subprocess, 'Popen', return_value=proc) as popen:
                    self.assertTrue(server.launch_cli(job, server.MODEL_CHOICES['vocalblend']['files'],
                                                     folder / 'input.wav', folder / 'output', 'cpu'))
                args, kwargs = popen.call_args
                command = args[0]
                self.assertEqual(command[command.index('--output_format') + 1], fmt)
                self.assertEqual(command[command.index('--model_file_dir') + 1], str(server.MODELS_DIR))
                self.assertEqual(command[command.index('--ensemble_algorithm') + 1], 'avg_fft')
                self.assertIn('--extra_models', command)
                self.assertIn('--mdxc_override_model_segment_size', command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '-1')
                self.assertFalse(kwargs.get('shell', False))
                self.assertEqual((folder / 'engine.log').read_text(), 'engine initialized\n')
                self.assertEqual('--output_bitrate' in command, fmt == 'MP3')
                if fmt == 'MP3':
                    self.assertEqual(command[command.index('--output_bitrate') + 1], '320k')


class CacheLocation(unittest.TestCase):
    def test_model_and_library_caches_are_project_local(self):
        env = server.make_env('cpu')
        for key in ['AUDIO_SEPARATOR_MODEL_DIR', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME']:
            with self.subTest(variable=key):
                self.assertTrue(Path(env[key]).is_relative_to(server.ROOT), env[key])
                self.assertEqual(Path(env[key]).drive.lower(), server.ROOT.drive.lower())
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '-1')


if __name__ == '__main__':
    unittest.main(verbosity=2)
