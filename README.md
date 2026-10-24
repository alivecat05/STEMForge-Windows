STEMForge · 本地 AI STEMS 分离器 v1.0
========================================

系统要求：Windows 10/11 x64。建议 RTX 4060 8GB（安装 CUDA 相关依赖），也支持 CPU。

使用步骤：
  1. 把压缩包完整解压到有足够空间的文件夹（本机位于 E:\STEMForge-Windows），不要在 ZIP 内直接运行。
  2. 首次双击 install.cmd：程序自动下载专用 Python 运行时、推理环境和 FFmpeg。
     不需要电脑预装 Python / Conda / CUDA Toolkit；需要可用网络。
     安装过程中会看到命令行下载进度；可能需要下载几 GB 的依赖。
  3. 双击 start.cmd，浏览器自动打开 http://127.0.0.1:8765。
  4. 拖入音频、选择模型、选择 GPU 或 CPU、选择 WAV/FLAC/MP3，点击开始。
  5. 缺失的模型权重会在首次使用时自动下载；以后从 data\models 复用。
  6. 在结果页面试听各分轨，单独保存或下载全部音轨 ZIP。
  7. 关闭运行窗口即可停止服务。

模型与模式：
  - Demucs FT 4 轨：vocals, drums, bass, other（建议默认）
  - Demucs 6S：上述四轨 + guitar, piano
  - BS-RoFormer：人声 / 伴奏（优先推荐高质量人声）
  - MelBand RoFormer：人声 / 伴奏
  - Dual RoFormer Ensemble：BS-RoFormer + MelBand RoFormer 人声融合；比单模型耗时长、显存压力更高
  - MDX-Net HQ：人声 / 伴奏，资源开销相对低
  - Drum Split：先 Demucs 4 轨，再 MDX23C 细分鼓组

模型推理不需要互联网，但首次安装依赖和首次下载每个模型需要互联网。

目录说明：
  app\       运行代码与页面，正常使用无需修改
  runtime\   安装后自动生成独立 Python、依赖库、uv 和 FFmpeg
  data\models\ 保存模型权重缓存（模型文件不在初始 ZIP 内）
  data\jobs\   保存上传的源音频、输出音轨和打包 ZIP
  runtime\install.log  记录安装日志

注意事项：
  - 交付物是“联网一键安装器 + 应用源码”，不是自带数 GB 环境的离线 EXE。
  - 本机 Windows / RTX 4060 8GB 已完成实际 GPU 分离测试；详细结果见项目中的测试报告。
  - 建议至少保留 20GB 可用空间用于运行环境、下载缓存和模型，音轨输出会额外占用空间。
  - 显卡要安装支持 CUDA 的 NVIDIA 驱动；只用 CPU 不需要 NVIDIA 驱动。
  - 8GB 显存使用 RoFormer / 双模型融合时，如果 OOM，可尝试“低显存尝试”模式或 CPU。
  - 部分 RoFormer 模型在 CPU 上比较慢；歌曲越长，使用内存越多。
  - 模型的“缓存”状态仅是文件存在性提示；引擎会下载缺失权重/配置。
  - Win Defender / 公司代理 / 防火墙可能阻止 GitHub、PyPI、FFmpeg、模型镜像下载。
  - 输出 MP3 为有损编码，不适合需要最高质量后期处理的场景，建议 WAV。
  - 安装器只下载官方托管的 uv、Python、PyPI 依赖、FFmpeg 和模型资源。
  - 处理的音乐需确保拥有合法使用权，遵守各模型权重的单独许可证。
  - 软件本地服务绑定 127.0.0.1，没有云上传接口，不要自行开放公网端口。

遇到问题：
  - 双击 install.cmd 重新尝试安装，会复用已缓存组件。
  - install.cmd 失败查看 runtime\install.log；安装卡住常见原因是网络连接/代理。
  - 页面打不开：确认 start.cmd 窗口没有关闭，网址为 http://127.0.0.1:8765。
  - 端口占用：关闭其他 STEMForge 实例。
  - GPU 不生效：更新 NVIDIA 驱动，运行系统 nvidia-smi 验证 GPU。
  - 下载权重失败：检查网络与 data\models 文件夹权限，打开任务日志。
  - 第一次模型下载有时会很慢；任务日志会显示下载失败原因。

项目依赖：
  - audio-separator 0.47.0（MIT，https://github.com/nomadkaraoke/python-audio-separator）
  - uv 0.13.3（MIT/Apache-2.0，https://github.com/astral-sh/uv）
  - FFmpeg（LGPL/GPL，取决于构建，https://ffmpeg.org/）
  - 各 AI 模型的许可可能与应用和依赖的开源许可不同，商用需独立确认。

本机测试结果不代表所有 Windows 系统和所有输入音频均已测试。
