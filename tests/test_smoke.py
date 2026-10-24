"""Offline smoke tests for the UI/API (model inference is mocked)."""
import http.client
import io
import json
import pathlib
import queue
import sys
import tempfile
import threading
import time
import unittest
import wave
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'app'))
import server


class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='stemforge-smoke-')
        cls.root = pathlib.Path(cls.temp.name)
        cls.isolation = patch.multiple(server, JOBS_DIR=cls.root / 'jobs',
                                       MODELS_DIR=cls.root / 'models', JOBS={}, TASKS=queue.Queue())
        cls.isolation.start()
        server.JOBS_DIR.mkdir()
        server.MODELS_DIR.mkdir()
        cls.http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        cls.port = cls.http.server_address[1]
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.worker = threading.Thread(target=server.worker, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        server.TASKS.join()
        cls.isolation.stop()
        cls.temp.cleanup()

    def fetch(self, method, path, data=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=6)
        conn.request(method, path, body=data, headers=headers or {})
        res = conn.getresponse()
        out = res.read()
        hdr = dict(res.getheaders())
        code = res.status
        conn.close()
        return code, hdr, out

    def test_index_and_info(self):
        code, headers, body = self.fetch('GET', '/')
        self.assertEqual(code, 200)
        self.assertIn(b'STEMForge', body)
        self.assertIn('text/html', headers['Content-Type'])
        code, _, body = self.fetch('GET', '/api/info')
        self.assertEqual(code, 200)
        self.assertEqual(len(json.loads(body)['models']), 7)

    def test_upload_zip_audio_range_delete(self):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b'\0\0' * 800)
        wav = buffer.getvalue()

        def fake_run(job, model, source, output, device, download_only=False):
            output.mkdir(parents=True, exist_ok=True)
            (output / 'song_(Vocals).wav').write_bytes(wav)
            return True

        with patch.object(server, 'launch_cli', new=fake_run):
            code, _, body = self.fetch('POST', '/api/separate', wav, {
                'X-Filename': 'song.wav', 'X-Model': 'demucs4', 'X-Device': 'cpu',
                'X-Format': 'WAV', 'Content-Type': 'application/octet-stream'
            })
            self.assertEqual(code, 202, body)
            id_ = json.loads(body)['id']
            for _ in range(120):
                code, _, body = self.fetch('GET', f'/api/jobs/{id_}')
                job = json.loads(body)
                if job['job']['status'] in ('completed', 'failed'):
                    break
                time.sleep(.03)
            self.assertEqual(job['job']['status'], 'completed', job['job']['logs'])
            self.assertEqual(len(job['files']), 1)
            self.assertTrue((server.JOBS_DIR / id_ / 'job.json').is_file())
            code, _, output = self.fetch('GET', f'/api/audio/{id_}/song_(Vocals).wav', headers={'Range':'bytes=0-9'})
            self.assertEqual(code, 206)
            self.assertEqual(output, wav[:10])
            code, _, z = self.fetch('GET', f'/api/jobs/{id_}/archive.zip')
            self.assertEqual(code, 200)
            with zipfile.ZipFile(io.BytesIO(z)) as file:
                self.assertEqual(file.namelist(), ['song_(Vocals).wav'])
            code, _, _ = self.fetch('DELETE', f'/api/jobs/{id_}')
            self.assertEqual(code, 200)
            code, _, _ = self.fetch('GET', f'/api/jobs/{id_}')
            self.assertEqual(code, 404)

    def test_upload_validations(self):
        code, _, _ = self.fetch('POST', '/api/separate', b'abc', {'Content-Type':'application/octet-stream','X-Model':'not-allowed', 'X-Filename':'bad.wav'})
        self.assertEqual(code, 400)
        code, _, _ = self.fetch('POST', '/api/separate', b'abc', {'Content-Type':'application/octet-stream','X-Filename':'malware.exe'})
        self.assertEqual(code, 400)

    def test_download_job(self):
        with patch.object(server, 'launch_cli', return_value=True):
            code, _, data = self.fetch('POST', '/api/download', b'{"model":"bsroformer"}', {'Content-Type':'application/json'})
            self.assertEqual(code, 202)
            id_ = json.loads(data)['id']
            for _ in range(100):
                code, _, data = self.fetch('GET', f'/api/jobs/{id_}')
                if json.loads(data)['job']['status'] in ('completed','failed'):
                    break
                time.sleep(.03)
            self.assertEqual(json.loads(data)['job']['status'], 'completed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
