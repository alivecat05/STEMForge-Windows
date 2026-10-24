"""Write the final test matrix and links from recorded evidence."""
import datetime
import importlib.metadata as metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'runtime/test-results'
MODELS = {'demucs4':'Demucs FT 四轨','demucs6':'Demucs 6S 六轨','bsroformer':'BS-RoFormer',
          'melroformer':'MelBand RoFormer','vocalblend':'双 RoFormer 融合','mdxfast':'MDX-Net HQ',
          'drumsplit':'Drum Split 两阶段'}
EXPECTED = [f'{m}-{fmt}-gpu' for m in MODELS for fmt in ['WAV','FLAC','MP3']] + [
    'bsroformer-WAV-gpu-low','mdxfast-WAV-cpu','demucs6-WAV-auto','mdxfast-WAV-auto','demucs4-WAV-auto']
records = {}
for key in EXPECTED:
    records[key] = json.loads((REPORTS / (key+'.json')).read_text(encoding='utf-8'))
audio = json.loads((REPORTS/'audio-audit.json').read_text(encoding='utf-8'))
invalid = json.loads((REPORTS/'invalid-input.json').read_text(encoding='utf-8'))
cpu_invalid = json.loads((REPORTS/'cpu-invalid-input.json').read_text(encoding='utf-8'))
assert all(r.get('passed') and r['status']=='completed' for r in records.values())
assert audio['overall']=='PASS' and audio['completed_reports']==len(EXPECTED)
assert invalid['passed']
assert cpu_invalid['passed']
api_log = (REPORTS/'api-tests.txt').read_text(encoding='utf-8-sig')
assert 'Ran 35 tests' in api_log and api_log.rstrip().endswith('OK')
lines = [
    'STEMForge 安装与全面测试报告',
    '日期：'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00'),
    '结果：全部计划测试通过。', '',
    '测试输入：'+records['demucs4-WAV-gpu']['source'],
    '完整歌曲：174.325 秒；7种模式均使用完整歌曲输出WAV。',
    '格式及专项补测：从同一首歌35秒处截取10秒，逐模式测试FLAC/MP3。',
    '本机环境：Windows；NVIDIA GeForce RTX 4060 Laptop GPU（8GB），驱动566.07。',
    'Python 3.11.17；CUDA 12.6；FFmpeg/FFprobe 9.0.2。',
    '依赖：'+', '.join(f'{p} {metadata.version(p)}' for p in ['audio-separator','torch','onnxruntime-gpu']), '',
    '模型存储：'+str(ROOT/'data/models'),
    '模型及配置约2.36GiB；已配置模型缓存、输入副本和最终输出均在E盘项目目录中。',
    '融合推理的临时音轨使用系统临时目录，完成后清理；模型文件始终保存在上述E盘路径。',
    '原始MP3文件未修改。', '',
    '模型测试矩阵（WAV=完整歌曲；FLAC/MP3=10秒片段）',
]
for model, label in MODELS.items():
    full = records[model+'-WAV-gpu']
    lines.append(f'{label}: WAV通过 / FLAC通过 / MP3通过；每次{len(full["files"])}轨')
    lines.append(f'  全曲输出：{ROOT / "data/jobs" / full["id"] / "output"}')
    lines.append(f'  全曲ZIP：{ROOT / "data/jobs" / full["id"] / ("STEMForge_"+full["id"]+"_STEMS.zip")}')
    lines.append(f'  首次测试耗时：{full["seconds"]:.1f}秒（含启动、可能的下载和状态轮询，不是纯推理性能基准）。')
lines += ['', f'实际分离：{len(records)}次成功；共{sum(len(r["files"]) for r in records.values())}条输出。',
    '全曲7种模式均确认使用CUDA，没有CPU回退。',
    '专项通过：BS-RoFormer低显存；MDX CPU；M4A输入+自动设备；FLAC输入+自动设备。',
    '错误恢复通过：损坏MP3被拒绝，队列随后成功处理正常音频。',
    'CPU错误保留通过：损坏MP3返回失败及非零退出码，未被退出兼容修复误判为成功。',
    '35项基础与边界测试通过：上传参数、JSON、Range、路径、ZIP CRC、生命周期、持久化、回退、模型编排及CPU退出异常保留等。',
    '每条音轨检查：预期名称/数量、时长误差≤0.2秒、完整解码、有限且非全零样本、峰值/RMS/削波比例。',
    '每次输出检查：浏览器试听Range接口、ZIP音轨清单及CRC。',
    '浏览器实测：整曲人声音轨连续播放36秒，后台轮询未重置播放器。',
    '重复安装复用检查通过；服务重启后完成任务和音轨可恢复。', '',
    '本次修复',
    '- 使用具体Python路径，解决Windows版本链接创建失败。',
    '- 安装器使用系统代理并放宽下载超时；固定CUDA12.6与匹配的ONNX版本；检查现有环境后复用。',
    '- 保留音轨名称；后台刷新保留播放器；隐藏未完成输出与临时音频。',
    '- 修复异常JSON及Range边界响应；保存完整推理日志；显示最近50条任务。',
    '- Windows CPU/ONNX任务曾在输出后以0xC0000005退出；现在仅在CLI正常返回后刷新日志并结束独立工作进程，真实推理异常仍保留。MDX CPU实测返回码0且音轨检查通过。具体原生故障来源尚未确认。',
    '- 修正鼓组输出说明：四轨 + kick/snare/toms/hi-hat/ride/crash，共10轨。', '',
    '范围说明：本报告验证这台机器和指定音频的运行、导出与文件完整性；没有干净参考分轨，因此不作SDR或主观音质排名。',
    '详细数据：'+str(REPORTS),
    '启动：双击项目start.cmd或桌面STEMForge快捷方式；本地页面 http://127.0.0.1:8765/',
]
(ROOT/'测试报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
print('PASS: 26 successful separations / 100 audio tracks; report saved to', ROOT/'测试报告.txt')
