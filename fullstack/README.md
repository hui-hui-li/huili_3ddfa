# 3DDFA Fullstack Reconstruction + Attention Analysis System

这个目录是基于 `3DDFA_V2` 核心算法搭建的完整前后端系统，包含：

- 用户管理：注册、登录、鉴权、编辑个人资料、修改密码
- 管理员能力：用户列表、封禁/解封、重置密码、授予/取消管理员
- 管理员审计：记录管理员关键操作日志并支持查询
- 媒体管理：用户照片/视频上传、列表、删除、下载
- 重建任务：每个媒体可发起 3D 人脸重建任务
- 结果管理：关键帧 OBJ、关键帧预览、逐帧 OBJ 序列 ZIP、动画 MP4、逐帧在线预览
- 头姿态与注意力分析：课堂/考试/驾驶三场景评分、时序平滑、预警与CSV导出
- 实时分析：单帧接口支持单人评分与多人抬头率统计（可用于摄像头流逐帧调用）
- 任务筛选与曲线可视化：支持按状态/场景筛选任务，支持注意力曲线查看
- 并发能力：支持多任务重建并发，重建进行中仍可继续使用其他接口

## 注意力分析能力

- 单人实时注意力评分：`single` 模式
- 多人课堂抬头率统计：`multi` 模式
- 驾驶员分心/疲劳预警：输出风险分和预警标签
- 时序平滑：视频分析阶段默认启用 EMA 平滑，降低抖动影响

## 目录结构

- `backend/app`: FastAPI 后端
- `frontend`: 前端页面（由后端静态托管）
- `backend/storage`: SQLite 数据库和上传/输出文件

## 在 conda 3ddfa 环境启动

```bash
cd /home/huili/3ddfa/3DDFA_V2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3ddfa
pip install -r requirements.txt
pip install -r fullstack/backend/requirements.txt
cd fullstack/backend
RECONSTRUCTION_MAX_WORKERS=2 INFERENCE_THREADS_PER_WORKER=1 uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问：`http://127.0.0.1:8000`

说明：系统默认将“首个注册用户”设为管理员，用于初始化平台管理权限。
说明：如果机器 CPU 核数更多，可适当提高 `RECONSTRUCTION_MAX_WORKERS`；如果希望在重建时界面更流畅，可优先保持 `INFERENCE_THREADS_PER_WORKER=1`。

## 核心 API

认证与用户：

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `PATCH /api/auth/profile`
- `POST /api/auth/change-password`

管理员：

- `GET /api/admin/users`
- `POST /api/admin/users/{user_id}/ban`
- `POST /api/admin/users/{user_id}/unban`
- `POST /api/admin/users/{user_id}/set-admin`
- `POST /api/admin/users/{user_id}/reset-password`
- `GET /api/admin/audit-logs`

媒体：

- `POST /api/media/photos`
- `POST /api/media/videos`
- `GET /api/media`
- `DELETE /api/media/{media_id}`
- `GET /api/media/{media_id}/download`

说明：`/api/media/photos` 和 `/api/media/videos` 在 `auto_reconstruct=true` 时支持
`attention_scenario=classroom|exam|driving`，用于指定自动重建的注意力场景阈值。

重建：

- `POST /api/reconstructions`
- `GET /api/reconstructions`
- `GET /api/reconstructions/{job_id}/download` （关键帧OBJ）
- `GET /api/reconstructions/{job_id}/preview` （关键帧预览）
- `GET /api/reconstructions/{job_id}/sequence` （逐帧序列ZIP）
- `GET /api/reconstructions/{job_id}/animation` （动画MP4）
- `GET /api/reconstructions/{job_id}/metadata` （序列元数据JSON）
- `GET /api/reconstructions/{job_id}/frames` （逐帧在线预览列表）
- `GET /api/reconstructions/{job_id}/frames/{frame_index}` （指定帧预览图）
- `GET /api/reconstructions/{job_id}/attention` （注意力分析摘要）
- `GET /api/reconstructions/{job_id}/attention-timeline` （注意力时序）
- `GET /api/reconstructions/{job_id}/attention-csv` （注意力CSV导出）
- `GET /api/reconstructions/{job_id}/attention-curve` （注意力曲线采样数据）
- `GET /api/reconstructions/{job_id}/attention-metadata` （注意力原始JSON）
- `GET /api/reconstructions/{job_id}/events` （任务实时进度 SSE）
- `POST /api/reconstructions/{job_id}/cancel` （终止重建任务，支持 queued/running）

说明：`POST /api/reconstructions` 请求体支持：

- `media_id`: 媒体ID
- `attention_scenario`: `classroom|exam|driving`（可选，默认 `classroom`）

说明：`GET /api/reconstructions` 支持筛选参数：

- `status`: `queued|running|completed|failed|cancelled`
- `attention_scenario`: `classroom|exam|driving`

实时注意力：

- `POST /api/attention/frame` （单帧头姿态+注意力评分，支持 `single|multi`）

请求参数示例：

- `scenario`: `classroom | exam | driving`
- `mode`: `single | multi`
- `session_id`（可选）：用于单人模式跨帧平滑
- `smoothing_alpha`（可选）：平滑系数 `0~0.95`
