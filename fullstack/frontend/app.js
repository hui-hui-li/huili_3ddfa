const API_PREFIX = '/api';

let token = localStorage.getItem('access_token') || '';
let pollTimer = null;
let currentUser = null;
let currentAttentionJobId = null;
let framePlaybackTimer = null;
let lastAttentionCurveData = { points: [] };
const selectedMediaIds = new Set();
const selectedJobIds = new Set();
const jobEventStreams = new Map();

const framePreviewState = {
  jobId: null,
  page: 1,
  pageSize: 60,
  total: 0,
  entries: [],
  currentIndex: 0,
};

const authPanel = document.getElementById('auth-panel');
const appPanel = document.getElementById('app-panel');
const adminPanel = document.getElementById('admin-panel');
const adminView = document.getElementById('view-admin');
const adminNavBtn = document.getElementById('admin-nav-btn');
const appNavButtons = Array.from(document.querySelectorAll('.app-nav-btn'));
const appViews = Array.from(document.querySelectorAll('.app-view'));

const authMsg = document.getElementById('auth-msg');
const appMsg = document.getElementById('app-msg');
const userInfo = document.getElementById('user-info');

const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const tabLogin = document.getElementById('tab-login');
const tabRegister = document.getElementById('tab-register');
const profileForm = document.getElementById('profile-form');
const profileUsernameInput = document.getElementById('profile-username');
const profileEmailInput = document.getElementById('profile-email');
const jobsFilterForm = document.getElementById('jobs-filter-form');
const jobsSearchFilter = document.getElementById('jobs-search-filter');
const jobsSearchClear = document.getElementById('jobs-search-clear');
const deleteAccountForm = document.getElementById('delete-account-form');
const jobsCreatedFrom = document.getElementById('jobs-created-from');
const jobsCreatedTo = document.getElementById('jobs-created-to');

const mediaBody = document.getElementById('media-body');
const jobsBody = document.getElementById('jobs-body');
const usersBody = document.getElementById('users-body');
const auditBody = document.getElementById('audit-body');
const mediaSelectAll = document.getElementById('media-select-all');
const jobsSelectAll = document.getElementById('jobs-select-all');
const batchRebuildMediaBtn = document.getElementById('batch-rebuild-media');
const batchDeleteMediaBtn = document.getElementById('batch-delete-media');
const batchCancelJobsBtn = document.getElementById('batch-cancel-jobs');
const batchDeleteJobsBtn = document.getElementById('batch-delete-jobs');

const frameModal = document.getElementById('frame-preview-modal');
const framePreviewImage = document.getElementById('frame-preview-image');
const framePreviewMeta = document.getElementById('frame-preview-meta');
const framePreviewStatus = document.getElementById('frame-preview-status');
const framePreviewDetail = document.getElementById('frame-preview-detail');
const pageInfo = document.getElementById('page-info');
const framePlaybackFps = document.getElementById('frame-playback-fps');
const playFramesBtn = document.getElementById('play-frames');
const imagePreviewModal = document.getElementById('image-preview-modal');
const imagePreviewTitle = document.getElementById('image-preview-title');
const imagePreviewMeta = document.getElementById('image-preview-meta');
const imagePreviewImage = document.getElementById('image-preview-image');

const realtimeAttentionSummary = document.getElementById('realtime-attention-summary');
const realtimeAttentionBody = document.getElementById('realtime-attention-body');
const attentionSummaryText = document.getElementById('attention-summary-text');
const attentionTimelineBody = document.getElementById('attention-timeline-body');
const attentionJobSelect = document.getElementById('attention-job-select');
const downloadAttentionCsvBtn = document.getElementById('download-attention-csv');
const photoUploadScenarioSelect = document.getElementById('photo-upload-scenario');
const videoUploadScenarioSelect = document.getElementById('video-upload-scenario');
const jobsStatusFilter = document.getElementById('jobs-status-filter');
const jobsScenarioFilter = document.getElementById('jobs-scenario-filter');
const attentionCurveCanvas = document.getElementById('attention-curve-canvas');
const RELOAD_VIEW_KEY = 'app_reload_view';
const RELOAD_MSG_KEY = 'app_reload_msg';
const RELOAD_TYPE_KEY = 'app_reload_type';
const STATUS_LABELS = {
  queued: '排队中',
  running: '进行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
};
const SCENARIO_LABELS = {
  classroom: '课堂',
  exam: '考试',
  driving: '驾驶',
};
const MODE_LABELS = {
  single: '单人',
  multi: '多人',
};
const MEDIA_TYPE_LABELS = {
  photo: '照片',
  video: '视频',
};
const WARNING_LABELS = {
  'no-face-detected': '未检测到人脸',
  'frequent-low-attention': '频繁低专注',
  'head-down-too-long': '低头时间过长',
  'long-side-view': '长时间侧视',
  'frequent-head-turn': '频繁转头',
  'driving-distraction-risk': '存在驾驶分心风险',
};
const STAGE_LABELS = {
  queued: '排队中',
  running: '进行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
  cancel_requested: '已请求终止',
  photo_load: '加载图片',
  photo_detect: '检测人脸',
  photo_mesh: '生成三维网格',
  photo_done: '照片重建完成',
  video_prepare: '准备视频',
  video_processing: '处理视频帧',
  video_finalize: '生成关键帧',
  video_packaging: '打包模型序列',
  video_metadata: '写入分析元数据',
  video_done: '视频重建完成',
};
const MESSAGE_LABELS = {
  'job queued': '任务已进入队列',
  'job started': '任务已开始执行',
  'job completed': '任务已完成',
  'loading image': '正在加载图片',
  'detecting face': '正在检测人脸',
  'generating mesh': '正在生成三维网格',
  'photo reconstruction completed': '照片重建完成',
  'video opened': '视频已打开',
  'processing video frames': '正在处理视频帧',
  'finalizing keyframe': '正在生成关键帧',
  'packaging frame models': '正在打包逐帧模型',
  'writing video metadata': '正在写入分析元数据',
  'video reconstruction completed': '视频重建完成',
  'cancel requested by user': '用户已请求终止',
  'cancelled by user before start': '任务开始前已取消',
  'cancelled by user': '用户已取消任务',
};

function setMsg(el, text, type = '') {
  el.textContent = text || '';
  el.classList.remove('error', 'ok');
  if (type) el.classList.add(type);
}

async function api(path, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const resp = await fetch(`${API_PREFIX}${path}`, {
    ...options,
    headers,
  });

  if (!resp.ok) {
    if (resp.status === 401 && token) {
      setToken('');
      currentUser = null;
      closeAllJobEventStreams();
      closeImagePreviewModal();
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      authPanel.classList.remove('hidden');
      appPanel.classList.add('hidden');
      setMsg(authMsg, '登录状态已失效，请重新登录', 'error');
    }
    let detail = `请求失败: ${resp.status}`;
    try {
      const data = await resp.json();
      detail = data.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }

  const contentType = resp.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return resp.json();
  }
  return resp;
}

function switchAuthTab(mode) {
  const isLogin = mode === 'login';
  tabLogin.classList.toggle('active', isLogin);
  tabRegister.classList.toggle('active', !isLogin);
  loginForm.classList.toggle('hidden', !isLogin);
  registerForm.classList.toggle('hidden', isLogin);
  setMsg(authMsg, '');
}

function switchAppView(viewId) {
  const hasTarget = appViews.some((view) => view.id === viewId);
  const targetViewId = hasTarget ? viewId : 'view-media';
  appViews.forEach((view) => {
    view.classList.toggle('hidden', view.id !== targetViewId);
  });
  appNavButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === targetViewId);
  });
}

function getCurrentViewId() {
  const active = appViews.find((view) => !view.classList.contains('hidden'));
  return active ? active.id : 'view-media';
}

function storeReloadState(viewId, message, type = 'ok') {
  sessionStorage.setItem(RELOAD_VIEW_KEY, viewId || 'view-media');
  if (message) {
    sessionStorage.setItem(RELOAD_MSG_KEY, message);
    sessionStorage.setItem(RELOAD_TYPE_KEY, type || 'ok');
  } else {
    sessionStorage.removeItem(RELOAD_MSG_KEY);
    sessionStorage.removeItem(RELOAD_TYPE_KEY);
  }
}

function consumeReloadState() {
  const viewId = sessionStorage.getItem(RELOAD_VIEW_KEY);
  const message = sessionStorage.getItem(RELOAD_MSG_KEY);
  const type = sessionStorage.getItem(RELOAD_TYPE_KEY);
  sessionStorage.removeItem(RELOAD_VIEW_KEY);
  sessionStorage.removeItem(RELOAD_MSG_KEY);
  sessionStorage.removeItem(RELOAD_TYPE_KEY);
  return {
    viewId: viewId || '',
    message: message || '',
    type: type || '',
  };
}

function reloadPagePreservingSession(viewId, message, type = 'ok') {
  storeReloadState(viewId || getCurrentViewId(), message, type);
  closeAllJobEventStreams();
  window.location.reload();
}

async function fetchMe() {
  return api('/auth/me');
}

function setToken(newToken) {
  token = newToken;
  if (token) {
    localStorage.setItem('access_token', token);
  } else {
    localStorage.removeItem('access_token');
  }
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(ts) {
  if (!ts) return '-';
  const d = new Date(ts);
  return d.toLocaleString();
}

function displayStatus(value) {
  const key = String(value ?? '').trim().toLowerCase();
  return STATUS_LABELS[key] || value || '-';
}

function displayScenario(value) {
  const key = String(value ?? '').trim().toLowerCase();
  return SCENARIO_LABELS[key] || value || '-';
}

function displayMode(value) {
  const key = String(value ?? '').trim().toLowerCase();
  return MODE_LABELS[key] || value || '-';
}

function displayMediaType(value) {
  const key = String(value ?? '').trim().toLowerCase();
  return MEDIA_TYPE_LABELS[key] || value || '-';
}

function displayWarning(value) {
  const key = String(value ?? '').trim().toLowerCase();
  return WARNING_LABELS[key] || value || '-';
}

function displayStage(value) {
  const key = String(value ?? '').trim().toLowerCase();
  return STAGE_LABELS[key] || value || '-';
}

function displayMessage(value) {
  const key = String(value ?? '').trim().toLowerCase();
  return MESSAGE_LABELS[key] || value || '';
}

function shortJobId(value) {
  const text = String(value ?? '');
  if (text.length <= 18) return text;
  return `${text.slice(0, 8)}...${text.slice(-6)}`;
}

async function copyText(text) {
  const value = String(text ?? '');
  if (!value) return;
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const temp = document.createElement('textarea');
  temp.value = value;
  temp.setAttribute('readonly', 'readonly');
  temp.style.position = 'fixed';
  temp.style.top = '-9999px';
  document.body.appendChild(temp);
  temp.select();
  document.execCommand('copy');
  document.body.removeChild(temp);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function openWithToken(path) {
  if (!token) return;
  const url = `${API_PREFIX}${path}${path.includes('?') ? '&' : '?'}access_token=${encodeURIComponent(token)}`;
  window.open(url, '_blank');
}

function userLabel(user) {
  const roles = [user.is_admin ? '管理员' : '普通用户'];
  if (user.is_banned) roles.push('已封禁');
  if (user.must_reset_password) roles.push('需重置密码');
  return roles.join(' / ');
}

function renderUserInfo() {
  if (!currentUser) {
    userInfo.textContent = '';
    return;
  }
  const labels = [userLabel(currentUser)];
  if (currentUser.must_reset_password) labels.push('请尽快修改密码');
  userInfo.textContent = `${currentUser.username} (${currentUser.email}) | ${labels.join(' | ')}`;
}

function fillProfileForm(user) {
  if (!user) return;
  profileUsernameInput.value = user.username || '';
  profileEmailInput.value = user.email || '';
}

function parseDetailJson(raw) {
  if (!raw) return '-';
  try {
    const parsed = JSON.parse(raw);
    return JSON.stringify(parsed);
  } catch (_) {
    return raw;
  }
}

function getUploadScenario(selectEl) {
  const value = (selectEl && selectEl.value) || 'classroom';
  if (value === 'exam' || value === 'driving' || value === 'classroom') return value;
  return 'classroom';
}

function updateBatchDeleteButton() {
  const count = selectedMediaIds.size;
  if (batchRebuildMediaBtn) {
    batchRebuildMediaBtn.disabled = count === 0;
    batchRebuildMediaBtn.textContent = count > 0 ? `批量发起重建 (${count})` : '批量发起重建';
  }
  if (batchDeleteMediaBtn) {
    batchDeleteMediaBtn.disabled = count === 0;
    batchDeleteMediaBtn.textContent = count > 0 ? `批量删除 (${count})` : '批量删除';
  }
}

function updateBatchDeleteJobsButton() {
  const count = selectedJobIds.size;
  if (batchCancelJobsBtn) {
    batchCancelJobsBtn.disabled = count === 0;
    batchCancelJobsBtn.textContent = count > 0 ? `批量终止任务 (${count})` : '批量终止任务';
  }
  if (batchDeleteJobsBtn) {
    batchDeleteJobsBtn.disabled = count === 0;
    batchDeleteJobsBtn.textContent = count > 0 ? `批量删除任务 (${count})` : '批量删除任务';
  }
}

function syncMediaSelectAllState(totalCount) {
  if (!mediaSelectAll) return;
  if (totalCount <= 0) {
    mediaSelectAll.checked = false;
    mediaSelectAll.indeterminate = false;
    return;
  }
  const selectedCount = selectedMediaIds.size;
  mediaSelectAll.checked = selectedCount > 0 && selectedCount === totalCount;
  mediaSelectAll.indeterminate = selectedCount > 0 && selectedCount < totalCount;
}

function syncJobSelectAllState(totalCount) {
  if (!jobsSelectAll) return;
  if (totalCount <= 0) {
    jobsSelectAll.checked = false;
    jobsSelectAll.indeterminate = false;
    return;
  }
  const selectedCount = selectedJobIds.size;
  jobsSelectAll.checked = selectedCount > 0 && selectedCount === totalCount;
  jobsSelectAll.indeterminate = selectedCount > 0 && selectedCount < totalCount;
}

function renderProgressCell(cell, jobLike) {
  if (!cell) return;
  const percentRaw = Number(jobLike.progress_percent ?? 0);
  const percent = Math.max(0, Math.min(100, Number.isFinite(percentRaw) ? percentRaw : 0));
  const stage = displayStage(jobLike.progress_stage || jobLike.status || '-');
  const msg = displayMessage(jobLike.progress_message || '') || (jobLike.progress_message || '');
  const totalFrames = Number(jobLike.total_frames ?? 0);
  const processedFrames = Number(jobLike.processed_frames ?? 0);
  const frameText = totalFrames > 0 ? ` | 帧 ${Math.max(0, processedFrames)}/${totalFrames}` : '';
  cell.innerHTML = `
    <div class="progress-wrap">
      <div class="progress-bar"><span style="width:${percent}%"></span></div>
      <div class="progress-meta">${percent}% | ${stage}${frameText}${msg ? ` | ${msg}` : ''}</div>
    </div>
  `;
}

function stopFramePlayback() {
  if (framePlaybackTimer) {
    clearInterval(framePlaybackTimer);
    framePlaybackTimer = null;
  }
  playFramesBtn.textContent = '自动播放';
}

function closeJobEventStream(jobId) {
  const stream = jobEventStreams.get(jobId);
  if (stream) {
    stream.close();
    jobEventStreams.delete(jobId);
  }
}

function closeAllJobEventStreams() {
  Array.from(jobEventStreams.keys()).forEach((jobId) => closeJobEventStream(jobId));
}

function applyJobEventToRow(payload) {
  if (!payload || !payload.job_id) return;
  const row = jobsBody.querySelector(`tr[data-job-id="${payload.job_id}"]`);
  if (!row) return;

  const statusEl = row.querySelector('.status');
  if (statusEl && payload.status) {
    statusEl.className = `status ${payload.status}`;
    statusEl.textContent = displayStatus(payload.status);
  }

  const progressCell = row.querySelector('.job-progress');
  renderProgressCell(progressCell, payload);

  const scenarioEl = row.querySelector('.job-scenario');
  if (scenarioEl && payload.attention_scenario) {
    scenarioEl.textContent = displayScenario(payload.attention_scenario);
  }

  const errCell = row.querySelector('.job-error');
  if (errCell) {
    errCell.textContent = payload.error_message || '-';
  }

  if (payload.status === 'completed' || payload.status === 'failed' || payload.status === 'cancelled') {
    closeJobEventStream(payload.job_id);
    loadJobs().catch(() => {});
  }
}

async function cancelJob(jobId) {
  if (!confirm(`确定要终止任务 ${jobId} 吗？`)) return;
  try {
    const result = await api(`/reconstructions/${jobId}/cancel`, { method: 'POST' });
    const message = result.status === 'cancelled'
      ? `任务 ${jobId} 已终止`
      : `任务 ${jobId} 已发送终止请求`;
    reloadPagePreservingSession('view-jobs', message, 'ok');
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
}

async function deleteJob(jobId) {
  if (!confirm(`确定删除重建任务 ${jobId} 吗？`)) return;
  try {
    await api(`/reconstructions/${jobId}`, { method: 'DELETE' });
    reloadPagePreservingSession('view-jobs', `任务 ${jobId} 已删除`, 'ok');
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
}

function openJobEventStream(jobId) {
  if (!token || jobEventStreams.has(jobId)) return;
  const url = `${API_PREFIX}/reconstructions/${jobId}/events?access_token=${encodeURIComponent(token)}`;
  const stream = new EventSource(url);
  const onEvent = (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      applyJobEventToRow(payload);
    } catch (_) {}
  };
  stream.addEventListener('snapshot', onEvent);
  stream.addEventListener('queued', onEvent);
  stream.addEventListener('progress', onEvent);
  stream.addEventListener('completed', onEvent);
  stream.addEventListener('failed', onEvent);
  stream.addEventListener('cancelled', onEvent);
  stream.onerror = () => {
    closeJobEventStream(jobId);
  };
  jobEventStreams.set(jobId, stream);
}

function syncJobEventStreams(jobs) {
  const activeIds = new Set(
    jobs
      .filter((job) => job.status === 'queued' || job.status === 'running')
      .map((job) => String(job.id)),
  );
  jobs.forEach((job) => {
    if (activeIds.has(String(job.id))) {
      openJobEventStream(String(job.id));
    }
  });
  Array.from(jobEventStreams.keys()).forEach((id) => {
    if (!activeIds.has(String(id))) {
      closeJobEventStream(id);
    }
  });
}

function closeFramePreviewModal() {
  stopFramePlayback();
  frameModal.classList.add('hidden');
  framePreviewState.jobId = null;
  framePreviewState.entries = [];
  framePreviewState.currentIndex = 0;
  framePreviewImage.src = '';
  framePreviewStatus.textContent = '';
  framePreviewDetail.textContent = '';
}

function closeImagePreviewModal() {
  imagePreviewModal.classList.add('hidden');
  imagePreviewTitle.textContent = '关键帧在线预览';
  imagePreviewMeta.textContent = '';
  imagePreviewImage.src = '';
}

function getFrameListPath() {
  return `/reconstructions/${framePreviewState.jobId}/frames`;
}

function getFrameImagePath(frameIndex) {
  return `${API_PREFIX}/reconstructions/${framePreviewState.jobId}/frames/${frameIndex}?access_token=${encodeURIComponent(token)}`;
}

async function loadFramePreviewPage(page) {
  if (!framePreviewState.jobId) return;

  const path = `${getFrameListPath()}?page=${page}&page_size=${framePreviewState.pageSize}`;
  const data = await api(path);
  framePreviewState.page = data.page;
  framePreviewState.total = data.total;
  framePreviewState.entries = data.entries || [];
  framePreviewState.currentIndex = 0;

  const totalPages = Math.max(1, Math.ceil(data.total / framePreviewState.pageSize));
  pageInfo.textContent = `页码 ${data.page}/${totalPages}`;

  if (!framePreviewState.entries.length) {
    stopFramePlayback();
    framePreviewImage.src = '';
    framePreviewStatus.textContent = '该页没有可预览帧';
    framePreviewDetail.textContent = '';
    return;
  }

  showCurrentFrame();
}

function showCurrentFrame() {
  if (!framePreviewState.entries.length) {
    framePreviewStatus.textContent = '没有可预览帧';
    framePreviewImage.src = '';
    framePreviewDetail.textContent = '';
    return;
  }

  const entry = framePreviewState.entries[framePreviewState.currentIndex];
  const frameUrl = getFrameImagePath(entry.frame_index);
  framePreviewImage.src = frameUrl;
  framePreviewStatus.textContent = `当前帧: ${entry.frame_index} (第 ${framePreviewState.currentIndex + 1}/${framePreviewState.entries.length} 帧)`;
  framePreviewDetail.textContent = '';
}

async function openFramePreview(jobId) {
  stopFramePlayback();
  framePreviewState.jobId = jobId;
  framePreviewState.page = 1;
  framePreviewState.entries = [];
  framePreviewState.currentIndex = 0;

  framePreviewMeta.textContent = `任务 ${jobId} | 模式: 3D网格逐帧`;
  frameModal.classList.remove('hidden');
  framePreviewStatus.textContent = '加载中...';

  try {
    await loadFramePreviewPage(1);
  } catch (err) {
    framePreviewStatus.textContent = err.message;
  }
}

function getProtectedFrameImageUrl(jobId, frameIndex) {
  return `${API_PREFIX}/reconstructions/${jobId}/frames/${frameIndex}?access_token=${encodeURIComponent(token)}`;
}

function openImagePreview({
  title = '关键帧在线预览',
  imageUrl,
  meta = '',
}) {
  imagePreviewTitle.textContent = title;
  imagePreviewMeta.textContent = meta;
  imagePreviewImage.src = imageUrl;
  imagePreviewModal.classList.remove('hidden');
}

async function stepToNextFrame() {
  if (!framePreviewState.entries.length) return false;

  if (framePreviewState.currentIndex < framePreviewState.entries.length - 1) {
    framePreviewState.currentIndex += 1;
    showCurrentFrame();
    return true;
  }

  const totalPages = Math.max(1, Math.ceil(framePreviewState.total / framePreviewState.pageSize));
  if (framePreviewState.page >= totalPages) {
    stopFramePlayback();
    return false;
  }
  await loadFramePreviewPage(framePreviewState.page + 1);
  showCurrentFrame();
  return true;
}

function toggleFramePlayback() {
  if (!framePreviewState.jobId) return;
  if (framePlaybackTimer) {
    stopFramePlayback();
    return;
  }

  const fps = Math.max(1, Math.min(24, Number(framePlaybackFps.value) || 8));
  const interval = Math.max(40, Math.round(1000 / fps));
  playFramesBtn.textContent = `停止 (${fps} fps)`;
  framePlaybackTimer = setInterval(() => {
    stepToNextFrame().catch((err) => {
      framePreviewStatus.textContent = err.message;
      stopFramePlayback();
    });
  }, interval);
}

async function loadMedia() {
  const list = await api('/media');
  selectedMediaIds.clear();
  mediaBody.innerHTML = '';
  updateBatchDeleteButton();
  syncMediaSelectAllState(list.length);

  if (!list.length) {
    mediaBody.innerHTML = '<tr><td colspan="8">暂无媒体</td></tr>';
    return;
  }

  list.forEach((item) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><input type="checkbox" class="media-select" data-media-id="${item.id}" /></td>
      <td>${item.id}</td>
      <td>${displayMediaType(item.media_type)}</td>
      <td>${item.original_filename}</td>
      <td>${formatSize(item.file_size)}</td>
      <td>${displayScenario(item.default_attention_scenario || 'classroom')}</td>
      <td>${formatTime(item.created_at)}</td>
      <td class="actions"></td>
    `;

    const actionsCell = tr.querySelector('.actions');
    const selectBox = tr.querySelector('.media-select');
    selectBox.addEventListener('change', (e) => {
      const checked = Boolean(e.target.checked);
      if (checked) selectedMediaIds.add(item.id);
      else selectedMediaIds.delete(item.id);
      updateBatchDeleteButton();
      syncMediaSelectAllState(list.length);
    });

    const btnRebuild = document.createElement('button');
    btnRebuild.className = 'small';
    btnRebuild.textContent = '发起重建';
    btnRebuild.onclick = async () => {
      try {
        const result = await api('/reconstructions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ media_id: item.id }),
        });
        alert(`发起重建成功\n任务ID: ${result.id}\n媒体ID: ${item.id}\n场景: ${displayScenario(result.attention_scenario)}`);
        setMsg(appMsg, `媒体 ${item.id} 已创建重建任务 (${displayScenario(result.attention_scenario)})`, 'ok');
        await loadJobs();
      } catch (err) {
        alert(`发起重建失败\n${err.message}`);
        setMsg(appMsg, err.message, 'error');
      }
    };

    const btnDownload = document.createElement('button');
    btnDownload.className = 'small secondary';
    btnDownload.textContent = '下载原文件';
    btnDownload.onclick = () => openWithToken(`/media/${item.id}/download`);

    const btnDelete = document.createElement('button');
    btnDelete.className = 'small warn';
    btnDelete.textContent = '删除';
    btnDelete.onclick = async () => {
      if (!confirm(`确定删除媒体 ${item.id} 吗？`)) return;
      try {
        await api(`/media/${item.id}`, { method: 'DELETE' });
        reloadPagePreservingSession('view-media', `媒体 ${item.id} 已删除`, 'ok');
      } catch (err) {
        setMsg(appMsg, err.message, 'error');
      }
    };

    actionsCell.append(btnRebuild, btnDownload, btnDelete);
    mediaBody.appendChild(tr);
  });
  updateBatchDeleteButton();
  syncMediaSelectAllState(list.length);
}

async function batchDeleteSelectedMedia() {
  const ids = Array.from(selectedMediaIds.values());
  if (!ids.length) return;
  if (!confirm(`确定批量删除这 ${ids.length} 个媒体吗？`)) return;

  try {
    const result = await api('/media/batch/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ media_ids: ids }),
    });

    if ((result.deleted_ids || []).length > 0) {
      const parts = [`已删除 ${result.deleted_ids.length} 个媒体`];
      if ((result.blocked_ids || []).length > 0) parts.push(`阻止 ${result.blocked_ids.length} 个`);
      if ((result.missing_ids || []).length > 0) parts.push(`缺失 ${result.missing_ids.length} 个`);
      reloadPagePreservingSession('view-media', parts.join('，'), 'ok');
      return;
    }

    const blockedText = (result.blocked_ids || []).length
      ? `存在运行中或排队中的重建任务，媒体ID: ${result.blocked_ids.join(', ')}`
      : '没有可删除的媒体';
    alert(blockedText);
    setMsg(appMsg, blockedText, 'error');
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
}

async function batchReconstructSelectedMedia() {
  const ids = Array.from(selectedMediaIds.values());
  if (!ids.length) return;
  if (!confirm(`确定批量发起这 ${ids.length} 个媒体的重建任务吗？`)) return;

  try {
    const result = await api('/reconstructions/batch/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ media_ids: ids }),
    });

    const parts = [];
    if ((result.created || []).length > 0) parts.push(`已创建 ${(result.created || []).length} 个任务`);
    if ((result.blocked_media_ids || []).length > 0) parts.push(`阻止 ${(result.blocked_media_ids || []).length} 个媒体`);
    if ((result.missing_media_ids || []).length > 0) parts.push(`缺失 ${(result.missing_media_ids || []).length} 个媒体`);
    setMsg(appMsg, parts.join('，') || '没有可创建的任务', (result.created || []).length > 0 ? 'ok' : 'error');
    await Promise.all([loadMedia(), loadJobs()]);
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
}

async function batchDeleteSelectedJobs() {
  const ids = Array.from(selectedJobIds.values());
  if (!ids.length) return;
  if (!confirm(`确定批量删除这 ${ids.length} 个重建任务吗？`)) return;

  try {
    const result = await api('/reconstructions/batch/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_ids: ids }),
    });

    if ((result.deleted_ids || []).length > 0) {
      const parts = [`已删除 ${result.deleted_ids.length} 个任务`];
      if ((result.blocked_ids || []).length > 0) parts.push(`阻止 ${result.blocked_ids.length} 个`);
      if ((result.missing_ids || []).length > 0) parts.push(`缺失 ${result.missing_ids.length} 个`);
      reloadPagePreservingSession('view-jobs', parts.join('，'), 'ok');
      return;
    }

    const blockedText = (result.blocked_ids || []).length
      ? `存在运行中或排队中的任务，任务ID: ${result.blocked_ids.join(', ')}`
      : '没有可删除的任务';
    alert(blockedText);
    setMsg(appMsg, blockedText, 'error');
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
}

async function batchCancelSelectedJobs() {
  const ids = Array.from(selectedJobIds.values());
  if (!ids.length) return;
  if (!confirm(`确定批量终止这 ${ids.length} 个重建任务吗？`)) return;

  try {
    const result = await api('/reconstructions/batch/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_ids: ids }),
    });

    const parts = [];
    if ((result.requested_ids || []).length > 0) parts.push(`已请求终止 ${(result.requested_ids || []).length} 个任务`);
    if ((result.cancelled_ids || []).length > 0) parts.push(`已直接终止 ${(result.cancelled_ids || []).length} 个任务`);
    if ((result.already_terminal_ids || []).length > 0) parts.push(`已结束 ${(result.already_terminal_ids || []).length} 个任务`);
    if ((result.missing_ids || []).length > 0) parts.push(`缺失 ${(result.missing_ids || []).length} 个任务`);
    setMsg(
      appMsg,
      parts.join('，') || '没有可终止的任务',
      (result.requested_ids || []).length > 0 || (result.cancelled_ids || []).length > 0 ? 'ok' : 'error',
    );
    await loadJobs();
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
}

function addJobActionButton(actionsCell, text, className, path) {
  const btn = document.createElement('button');
  btn.className = className;
  btn.textContent = text;
  btn.onclick = () => openWithToken(path);
  actionsCell.appendChild(btn);
}

async function loadAttentionForJob(jobId, switchToAttentionView = false) {
  try {
    const [summary, timeline, curve] = await Promise.all([
      api(`/reconstructions/${jobId}/attention`),
      api(`/reconstructions/${jobId}/attention-timeline?page=1&page_size=40`),
      api(`/reconstructions/${jobId}/attention-curve?max_points=260`),
    ]);

    currentAttentionJobId = String(jobId);
    if (attentionJobSelect) attentionJobSelect.value = String(jobId);
    downloadAttentionCsvBtn.disabled = false;
    const selectedJobName = attentionJobSelect && attentionJobSelect.selectedOptions.length
      ? String(attentionJobSelect.selectedOptions[0].textContent || '').split(' | ')[0]
      : String(jobId);

    const warnings = (summary.warnings || []).length ? summary.warnings.map(displayWarning).join('、') : '无';
    attentionSummaryText.textContent = [
      `任务: ${selectedJobName}`,
      `场景: ${displayScenario(summary.scenario)}`,
      `平均专注度: ${Number(summary.avg_attention || 0).toFixed(2)}`,
      `低专注比例: ${(Number(summary.low_attention_ratio || 0) * 100).toFixed(2)}%`,
      `抬头率: ${Number(summary.classroom_head_up_rate || 0).toFixed(2)}%`,
      `考试专注分: ${Number(summary.exam_focus_score || 0).toFixed(2)}`,
      `驾驶风险分: ${Number(summary.driving_risk_score || 0).toFixed(2)}`,
      `预警: ${warnings}`,
    ].join(' | ');

    attentionTimelineBody.innerHTML = '';
    const entries = timeline.entries || [];
    if (!entries.length) {
      attentionTimelineBody.innerHTML = '<tr><td colspan="6">暂无时序数据</td></tr>';
    } else {
      entries.forEach((entry) => {
        const tr = document.createElement('tr');
        const frameImageUrl = getProtectedFrameImageUrl(jobId, entry.frame_index);
        const tags = [];
        if (entry.head_down) tags.push('低头');
        if (entry.side_view) tags.push('偏头');
        if (entry.tilted) tags.push('倾斜');
        if (entry.rapid_turn) tags.push('转头');
        if (entry.distracted) tags.push('分心');
        tr.innerHTML = `
          <td>${entry.frame_index}</td>
          <td class="attention-frame-cell">
            <img class="attention-frame-thumb" src="${frameImageUrl}" alt="帧 ${entry.frame_index}" loading="lazy" />
          </td>
          <td>${Number(entry.yaw || 0).toFixed(2)}</td>
          <td>${Number(entry.pitch || 0).toFixed(2)}</td>
          <td>${Number(entry.roll || 0).toFixed(2)}</td>
          <td>${Number(entry.attention_score || 0).toFixed(2)}</td>
          <td>${tags.join(' / ') || '正常'}</td>
        `;
        const thumb = tr.querySelector('.attention-frame-thumb');
        if (thumb) {
          thumb.addEventListener('click', () => {
            openImagePreview({
              title: '注意力帧预览',
              imageUrl: frameImageUrl,
              meta: `任务 ${selectedJobName} | 帧 ${entry.frame_index}`,
            });
          });
          thumb.addEventListener('error', () => {
            thumb.alt = `帧 ${entry.frame_index} 预览失败`;
            thumb.classList.add('broken');
          });
        }
        attentionTimelineBody.appendChild(tr);
      });
    }

    if (switchToAttentionView) {
      switchAppView('view-attention');
    }
    lastAttentionCurveData = curve || { points: [] };
    drawAttentionCurve(lastAttentionCurveData);
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
}

async function loadPhotoAnalysisFromJob(job, switchToPhotoView = true) {
  try {
    const meta = await api(`/reconstructions/${job.id}/attention-metadata`);
    const summary = meta.summary || {};
    const faces = Array.isArray(meta.faces) && meta.faces.length
      ? meta.faces
      : (meta.entries || []).map((entry, idx) => ({
          face_index: idx,
          yaw: entry.yaw ?? 0,
          pitch: entry.pitch ?? 0,
          roll: entry.roll ?? 0,
          attention_score: entry.attention_score ?? 0,
          head_down: Boolean(entry.head_down),
          side_view: Boolean(entry.side_view),
          tilted: Boolean(entry.tilted),
          distracted: Boolean(entry.distracted),
        }));
    if (!faces.length) {
      throw new Error('该照片任务没有可用的注意力分析结果');
    }

    const payload = {
      mode: meta.mode || (faces.length > 1 ? 'multi' : 'single'),
      scenario: summary.scenario || meta.scenario || job.attention_scenario || 'classroom',
      face_count: faces.length,
      avg_attention: summary.avg_attention ?? (faces.reduce((sum, face) => sum + Number(face.attention_score || 0), 0) / Math.max(1, faces.length)),
      classroom_head_up_rate: summary.classroom_head_up_rate ?? (
        faces.filter((face) => !face.head_down && !face.side_view).length / Math.max(1, faces.length)
      ) * 100,
      faces,
    };

    renderRealtimeAttention(payload);
    realtimeAttentionSummary.textContent = `任务：${job.task_name} | 场景：${displayScenario(payload.scenario)} | 模式：${displayMode(payload.mode)} | 人脸数：${payload.face_count} | 平均专注度：${Number(payload.avg_attention || 0).toFixed(2)} | 抬头率：${Number(payload.classroom_head_up_rate || 0).toFixed(2)}%`;
    document.getElementById('attention-scenario').value = payload.scenario;
    document.getElementById('attention-mode').value = payload.mode === 'multi' ? 'multi' : 'single';
    document.getElementById('attention-session-id').value = '';
    if (switchToPhotoView) {
      switchAppView('view-photo-analysis');
    }
    setMsg(appMsg, `已加载照片任务 ${job.task_name} 的分析结果`, 'ok');
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
}

async function loadAttentionJobOptions(preferredJobId = currentAttentionJobId) {
  if (!attentionJobSelect) return;

  const jobs = await api('/reconstructions?status=completed');
  const attentionJobs = jobs.filter((job) => job.media_type === 'video' && Boolean(job.output_attention_metadata_path));
  attentionJobSelect.innerHTML = '<option value="">选择任务名称加载分析</option>';

  attentionJobs.forEach((job) => {
    const option = document.createElement('option');
    option.value = String(job.id);
    option.textContent = `${job.task_name} | ${displayScenario(job.attention_scenario)} | ${formatTime(job.created_at)}`;
    attentionJobSelect.appendChild(option);
  });

  if (preferredJobId && attentionJobs.some((job) => String(job.id) === String(preferredJobId))) {
    attentionJobSelect.value = String(preferredJobId);
  }
}

function renderRealtimeAttention(data) {
  const faceCount = Number(data.face_count || 0);
  const avgAttention = Number(data.avg_attention || 0).toFixed(2);
  const headUpRate = Number(data.classroom_head_up_rate || 0).toFixed(2);
  realtimeAttentionSummary.textContent = `模式：${displayMode(data.mode)} | 场景：${displayScenario(data.scenario)} | 人脸数：${faceCount} | 平均专注度：${avgAttention} | 抬头率：${headUpRate}%`;

  realtimeAttentionBody.innerHTML = '';
  const faces = data.faces || [];
  if (!faces.length) {
    realtimeAttentionBody.innerHTML = '<tr><td colspan="6">未检测到人脸</td></tr>';
    return;
  }

  faces.forEach((face) => {
    const tags = [];
    if (face.head_down) tags.push('低头');
    if (face.side_view) tags.push('侧视');
    if (face.tilted) tags.push('倾斜');
    if (face.distracted) tags.push('分心');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${Number(face.face_index ?? 0) + 1}</td>
      <td>${Number(face.yaw || 0).toFixed(2)}</td>
      <td>${Number(face.pitch || 0).toFixed(2)}</td>
      <td>${Number(face.roll || 0).toFixed(2)}</td>
      <td>${Number(face.attention_score || 0).toFixed(2)}</td>
      <td>${tags.join(' / ') || '正常'}</td>
    `;
    realtimeAttentionBody.appendChild(tr);
  });
}

function drawAttentionCurve(curveData) {
  if (!attentionCurveCanvas) return;

  const ctx = attentionCurveCanvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const cssWidth = Math.max(640, attentionCurveCanvas.clientWidth || 640);
  const cssHeight = Number(attentionCurveCanvas.getAttribute('height') || 220);
  attentionCurveCanvas.width = Math.round(cssWidth * dpr);
  attentionCurveCanvas.height = Math.round(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = '#f8fbff';
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const points = (curveData && curveData.points) || [];
  const pad = { left: 44, right: 14, top: 16, bottom: 28 };
  const w = cssWidth - pad.left - pad.right;
  const h = cssHeight - pad.top - pad.bottom;

  ctx.strokeStyle = '#d7e1ee';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const y = pad.top + (h * i) / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + w, y);
    ctx.stroke();
  }

  ctx.fillStyle = '#5d6a79';
  ctx.font = '12px sans-serif';
  ctx.fillText('100', 10, pad.top + 4);
  ctx.fillText('0', 22, pad.top + h + 4);

  if (!points.length) {
    ctx.fillStyle = '#8895a7';
    ctx.fillText('暂无曲线数据', pad.left + 8, pad.top + h / 2);
    return;
  }

  const maxIndex = Math.max(1, points.length - 1);
  const xFor = (i) => pad.left + (w * i) / maxIndex;
  const yFor = (score) => pad.top + (1 - Math.max(0, Math.min(100, Number(score) || 0)) / 100) * h;

  ctx.strokeStyle = '#1f6feb';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  points.forEach((pt, i) => {
    const x = xFor(i);
    const y = yFor(pt.attention_score);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.strokeStyle = '#1aa784';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((pt, i) => {
    const x = xFor(i);
    const y = yFor(pt.moving_avg_score);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = '#c0392b';
  points.forEach((pt, i) => {
    if (!pt.distracted) return;
    const x = xFor(i);
    const y = yFor(pt.attention_score);
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = '#5d6a79';
  const startFrame = points[0].frame_index || 0;
  const endFrame = points[points.length - 1].frame_index || 0;
  ctx.fillText(`帧 ${startFrame} - ${endFrame}`, pad.left, cssHeight - 8);
  ctx.fillText('蓝线：专注度  绿线：移动平均  红点：分心', pad.left + 110, cssHeight - 8);
}

async function loadJobs() {
  const query = [];
  const statusVal = jobsStatusFilter ? jobsStatusFilter.value : '';
  const scenarioVal = jobsScenarioFilter ? jobsScenarioFilter.value : '';
  const searchVal = jobsSearchFilter ? jobsSearchFilter.value.trim() : '';
  const createdFromVal = jobsCreatedFrom ? jobsCreatedFrom.value : '';
  const createdToVal = jobsCreatedTo ? jobsCreatedTo.value : '';
  if (statusVal) query.push(`status=${encodeURIComponent(statusVal)}`);
  if (scenarioVal) query.push(`attention_scenario=${encodeURIComponent(scenarioVal)}`);
  if (searchVal) query.push(`search=${encodeURIComponent(searchVal)}`);
  if (createdFromVal) query.push(`created_from=${encodeURIComponent(createdFromVal)}`);
  if (createdToVal) query.push(`created_to=${encodeURIComponent(createdToVal)}`);
  const path = query.length ? `/reconstructions?${query.join('&')}` : '/reconstructions';
  const list = await api(path);
  jobsBody.innerHTML = '';

  const currentIds = new Set(list.map((job) => String(job.id)));
  Array.from(selectedJobIds).forEach((id) => {
    if (!currentIds.has(String(id))) selectedJobIds.delete(String(id));
  });

  if (!list.length) {
    jobsBody.innerHTML = '<tr><td colspan="12">暂无任务</td></tr>';
    updateBatchDeleteJobsButton();
    syncJobSelectAllState(0);
    syncJobEventStreams([]);
    return;
  }

  list.forEach((job) => {
    const terminalStatuses = new Set(['completed', 'failed', 'cancelled']);
    const tr = document.createElement('tr');
    tr.dataset.jobId = String(job.id);
    const taskName = job.task_name || `media-${job.media_id}`;
    const shortId = shortJobId(job.id);
    tr.innerHTML = `
      <td><input type="checkbox" class="job-select" data-job-id="${escapeHtml(job.id)}" ${selectedJobIds.has(String(job.id)) ? 'checked' : ''} /></td>
      <td title="${escapeHtml(job.id)}" class="job-id-text">${escapeHtml(shortId)}</td>
      <td title="${escapeHtml(taskName)}">${escapeHtml(taskName)}</td>
      <td>${job.media_id}</td>
      <td><span class="status ${job.status}">${displayStatus(job.status)}</span></td>
      <td class="job-scenario">${displayScenario(job.attention_scenario || 'classroom')}</td>
      <td class="job-progress"></td>
      <td>${job.keyframe_index == null ? '-' : job.keyframe_index}</td>
      <td>${formatTime(job.created_at)}</td>
      <td class="job-error">${job.error_message || '-'}</td>
      <td class="actions job-results"></td>
      <td class="actions job-ops"></td>
    `;
    renderProgressCell(tr.querySelector('.job-progress'), job);

    const selectBox = tr.querySelector('.job-select');
    selectBox.addEventListener('change', (e) => {
      const checked = Boolean(e.target.checked);
      if (checked) selectedJobIds.add(String(job.id));
      else selectedJobIds.delete(String(job.id));
      updateBatchDeleteJobsButton();
      syncJobSelectAllState(list.length);
    });

    const resultsCell = tr.querySelector('.job-results');
    const taskOpsCell = tr.querySelector('.job-ops');
    const copyBtn = document.createElement('button');
    copyBtn.className = 'small secondary';
    copyBtn.textContent = '复制ID';
    copyBtn.onclick = async () => {
      try {
        await copyText(job.id);
        setMsg(appMsg, `任务ID已复制: ${job.id}`, 'ok');
      } catch (err) {
        setMsg(appMsg, `复制失败: ${err.message}`, 'error');
      }
    };
    taskOpsCell.appendChild(copyBtn);

    if (job.status === 'completed') {
      if (job.output_model_path) addJobActionButton(resultsCell, '关键帧OBJ', 'small', `/reconstructions/${job.id}/download`);
      if (job.output_preview_path) addJobActionButton(resultsCell, '关键帧预览', 'small secondary', `/reconstructions/${job.id}/preview`);
      if (job.media_type === 'photo' && job.output_preview_path) {
        const inlinePreviewBtn = document.createElement('button');
        inlinePreviewBtn.className = 'small secondary';
        inlinePreviewBtn.textContent = '在线关键帧预览';
        inlinePreviewBtn.onclick = () => {
          openImagePreview({
            title: '关键帧在线预览',
            imageUrl: `${API_PREFIX}/reconstructions/${job.id}/preview?access_token=${encodeURIComponent(token)}`,
            meta: `任务 ${job.id}${taskName ? ` | ${taskName}` : ''}`,
          });
        };
        resultsCell.appendChild(inlinePreviewBtn);
      }
      if (job.output_animation_path) addJobActionButton(resultsCell, '动画MP4', 'small secondary', `/reconstructions/${job.id}/animation`);
      if (job.output_metadata_path) {
        const previewBtn = document.createElement('button');
        previewBtn.className = 'small';
        previewBtn.textContent = '在线逐帧预览';
        previewBtn.onclick = () => openFramePreview(job.id);
        resultsCell.appendChild(previewBtn);
      }
      if (job.output_attention_metadata_path) {
        const attentionBtn = document.createElement('button');
        attentionBtn.className = 'small secondary';
        attentionBtn.textContent = '注意力分析';
        attentionBtn.onclick = () => {
          if (job.media_type === 'photo') {
            loadPhotoAnalysisFromJob(job, true).catch(() => {});
          } else {
            loadAttentionForJob(job.id, true).catch(() => {});
          }
        };
        resultsCell.appendChild(attentionBtn);
      }
      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'small warn';
      deleteBtn.textContent = '删除任务';
      deleteBtn.onclick = () => {
        deleteJob(job.id).catch(() => {});
      };
      taskOpsCell.appendChild(deleteBtn);
    } else if (!terminalStatuses.has(job.status)) {
      const cancelBtn = document.createElement('button');
      const cancelling = job.progress_stage === 'cancel_requested';
      cancelBtn.className = 'small warn';
      cancelBtn.textContent = cancelling ? '终止中' : '终止重建';
      cancelBtn.disabled = cancelling;
      cancelBtn.onclick = () => {
        cancelJob(job.id).catch(() => {});
      };
      taskOpsCell.appendChild(cancelBtn);
    } else {
      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'small warn';
      deleteBtn.textContent = '删除任务';
      deleteBtn.onclick = () => {
        deleteJob(job.id).catch(() => {});
      };
      taskOpsCell.appendChild(deleteBtn);
    }

    jobsBody.appendChild(tr);
  });
  updateBatchDeleteJobsButton();
  syncJobSelectAllState(list.length);
  syncJobEventStreams(list);
}

async function loadAuditLogs() {
  if (!currentUser || !currentUser.is_admin) return;

  const logs = await api('/admin/audit-logs?limit=120');
  auditBody.innerHTML = '';

  if (!logs.length) {
    auditBody.innerHTML = '<tr><td colspan="5">暂无审计日志</td></tr>';
    return;
  }

  logs.forEach((log) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${formatTime(log.created_at)}</td>
      <td>${log.admin_username || log.admin_user_id}</td>
      <td>${log.action}</td>
      <td>${log.target_username || (log.target_user_id ?? '-')}</td>
      <td>${parseDetailJson(log.detail_json)}</td>
    `;
    auditBody.appendChild(tr);
  });
}

async function loadUsersAdmin() {
  if (!currentUser || !currentUser.is_admin) {
    adminPanel.classList.add('hidden');
    adminNavBtn.classList.add('hidden');
    if (!adminView.classList.contains('hidden')) {
      switchAppView('view-media');
    }
    usersBody.innerHTML = '';
    auditBody.innerHTML = '';
    return;
  }

  adminNavBtn.classList.remove('hidden');
  adminPanel.classList.remove('hidden');

  const users = await api('/admin/users');
  usersBody.innerHTML = '';

  if (!users.length) {
    usersBody.innerHTML = '<tr><td colspan="7">暂无用户</td></tr>';
    return;
  }

  users.forEach((user) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${user.id}</td>
      <td>${user.username}</td>
      <td>${user.email}</td>
      <td>${user.is_admin ? '管理员' : '普通用户'}</td>
      <td>${user.is_banned ? '封禁' : '正常'}</td>
      <td>${user.ban_reason || '-'}</td>
      <td class="actions"></td>
    `;

    const actions = tr.querySelector('.actions');

    const banBtn = document.createElement('button');
    banBtn.className = `small ${user.is_banned ? 'secondary' : 'warn'}`;
    banBtn.textContent = user.is_banned ? '解封' : '封禁';
    banBtn.onclick = async () => {
      try {
        if (user.is_banned) {
          await api(`/admin/users/${user.id}/unban`, { method: 'POST' });
          setMsg(appMsg, `用户 ${user.username} 已解封`, 'ok');
        } else {
          const reason = prompt('封禁原因（可选）', '违反平台规范') || '';
          await api(`/admin/users/${user.id}/ban`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reason }),
          });
          setMsg(appMsg, `用户 ${user.username} 已封禁`, 'ok');
        }
        await loadUsersAdmin();
        await loadAuditLogs();
      } catch (err) {
        setMsg(appMsg, err.message, 'error');
      }
    };

    const resetBtn = document.createElement('button');
    resetBtn.className = 'small secondary';
    resetBtn.textContent = '重置密码';
    resetBtn.onclick = async () => {
      try {
        const customPwd = prompt('输入新密码（留空则自动生成）', '');
        const payload = customPwd ? { new_password: customPwd } : {};
        const result = await api(`/admin/users/${user.id}/reset-password`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        alert(`用户 ${user.username} 新密码：${result.temporary_password}`);
        setMsg(appMsg, `用户 ${user.username} 密码已重置`, 'ok');
        await loadAuditLogs();
      } catch (err) {
        setMsg(appMsg, err.message, 'error');
      }
    };

    const adminBtn = document.createElement('button');
    adminBtn.className = 'small secondary';
    adminBtn.textContent = user.is_admin ? '取消管理员' : '设为管理员';
    adminBtn.onclick = async () => {
      try {
        await api(`/admin/users/${user.id}/set-admin`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ is_admin: !user.is_admin }),
        });
        setMsg(appMsg, `用户 ${user.username} 角色已更新`, 'ok');
        await loadUsersAdmin();
        await loadAuditLogs();
      } catch (err) {
        setMsg(appMsg, err.message, 'error');
      }
    };

    actions.append(banBtn, resetBtn, adminBtn);
    usersBody.appendChild(tr);
  });
}

async function enterApp() {
  try {
    const reloadState = consumeReloadState();
    const me = await fetchMe();
    currentUser = me;
    renderUserInfo();
    fillProfileForm(me);
    switchAppView(reloadState.viewId || 'view-media');

    authPanel.classList.add('hidden');
    appPanel.classList.remove('hidden');
    if (reloadState.message) setMsg(appMsg, reloadState.message, reloadState.type || 'ok');
    else setMsg(appMsg, '登录成功', 'ok');

    await loadMedia();
    await loadJobs();
    await loadAttentionJobOptions();
    await loadUsersAdmin();
    await loadAuditLogs();

    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      loadJobs().catch(() => {});
    }, 12000);
  } catch (_) {
    currentUser = null;
    setToken('');
    closeAllJobEventStreams();
    closeImagePreviewModal();
    authPanel.classList.remove('hidden');
    appPanel.classList.add('hidden');
    adminPanel.classList.add('hidden');
    adminNavBtn.classList.add('hidden');
    switchAppView('view-media');
  }
}

loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(loginForm);
  const payload = {
    username_or_email: fd.get('username_or_email'),
    password: fd.get('password'),
  };

  try {
    const data = await api('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setToken(data.access_token);
    setMsg(authMsg, '登录成功', 'ok');
    await enterApp();
  } catch (err) {
    setMsg(authMsg, err.message, 'error');
  }
});

registerForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(registerForm);
  const payload = {
    username: fd.get('username'),
    email: fd.get('email'),
    password: fd.get('password'),
  };

  try {
    const data = await api('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setToken(data.access_token);
    setMsg(authMsg, '注册成功并已登录', 'ok');
    await enterApp();
  } catch (err) {
    setMsg(authMsg, err.message, 'error');
  }
});

document.getElementById('photo-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('photo-file').files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const scenarioValue = getUploadScenario(photoUploadScenarioSelect);
    const scenario = encodeURIComponent(scenarioValue);
    await api(`/media/photos?auto_reconstruct=true&attention_scenario=${scenario}`, {
      method: 'POST',
      body: fd,
    });
    setMsg(appMsg, `照片上传成功，已自动发起${displayScenario(scenarioValue)}场景重建`, 'ok');
    e.target.reset();
    await loadMedia();
    await loadJobs();
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
});

document.getElementById('video-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('video-file').files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const scenarioValue = getUploadScenario(videoUploadScenarioSelect);
    const scenario = encodeURIComponent(scenarioValue);
    await api(`/media/videos?auto_reconstruct=true&attention_scenario=${scenario}`, {
      method: 'POST',
      body: fd,
    });
    setMsg(appMsg, `视频上传成功，已自动发起${displayScenario(scenarioValue)}场景重建`, 'ok');
    e.target.reset();
    await loadMedia();
    await loadJobs();
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
});

profileForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const username = profileUsernameInput.value.trim();
  const email = profileEmailInput.value.trim();

  try {
    const updated = await api('/auth/profile', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email }),
    });
    currentUser = updated;
    renderUserInfo();
    fillProfileForm(updated);
    setMsg(appMsg, '个人资料已更新', 'ok');
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
});

document.getElementById('change-password-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const oldPassword = document.getElementById('old-password').value;
  const newPassword = document.getElementById('new-password').value;

  try {
    await api('/auth/change-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
    });
    setMsg(appMsg, '密码更新成功', 'ok');
    e.target.reset();
    if (currentUser) {
      currentUser.must_reset_password = false;
      renderUserInfo();
    }
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
});

deleteAccountForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const passwordInput = document.getElementById('delete-account-password');
  const password = passwordInput.value;
  if (!password) return;
  if (!confirm('确定注销当前账号吗？此操作会删除你的媒体、任务记录和输出文件，且无法恢复。')) return;

  try {
    await api('/auth/delete-account', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    });
    setToken('');
    currentUser = null;
    currentAttentionJobId = null;
    lastAttentionCurveData = { points: [] };
    drawAttentionCurve(lastAttentionCurveData);
    if (attentionJobSelect) attentionJobSelect.innerHTML = '<option value="">选择任务名称加载分析</option>';
    closeAllJobEventStreams();
    closeFramePreviewModal();
    closeImagePreviewModal();
    adminNavBtn.classList.add('hidden');
    adminPanel.classList.add('hidden');
    downloadAttentionCsvBtn.disabled = true;
    setMsg(appMsg, '');
    setMsg(authMsg, '账号已注销', 'ok');
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    authPanel.classList.remove('hidden');
    appPanel.classList.add('hidden');
    switchAuthTab('login');
    switchAppView('view-media');
    deleteAccountForm.reset();
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
});

document.getElementById('realtime-attention-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('attention-frame-file').files[0];
  if (!file) return;

  const scenario = document.getElementById('attention-scenario').value || 'classroom';
  const mode = document.getElementById('attention-mode').value || 'single';
  const sessionId = document.getElementById('attention-session-id').value.trim();

  const fd = new FormData();
  fd.append('file', file);

  let path = `/attention/frame?scenario=${encodeURIComponent(scenario)}&mode=${encodeURIComponent(mode)}`;
  if (sessionId) {
    path += `&session_id=${encodeURIComponent(sessionId)}`;
  }

  try {
    const result = await api(path, {
      method: 'POST',
      body: fd,
    });
    renderRealtimeAttention(result);
    setMsg(appMsg, '照片分析完成', 'ok');
  } catch (err) {
    setMsg(appMsg, err.message, 'error');
  }
});

document.getElementById('attention-job-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const jobId = attentionJobSelect ? attentionJobSelect.value.trim() : '';
  if (!jobId) return;
  await loadAttentionForJob(jobId, false);
});

downloadAttentionCsvBtn.addEventListener('click', () => {
  if (!currentAttentionJobId) return;
  openWithToken(`/reconstructions/${currentAttentionJobId}/attention-csv`);
});

document.getElementById('logout-btn').addEventListener('click', () => {
  setToken('');
  currentUser = null;
  currentAttentionJobId = null;
  lastAttentionCurveData = { points: [] };
  drawAttentionCurve(lastAttentionCurveData);
  closeAllJobEventStreams();
  closeFramePreviewModal();
  closeImagePreviewModal();
  adminNavBtn.classList.add('hidden');
  downloadAttentionCsvBtn.disabled = true;
  setMsg(appMsg, '');
  setMsg(authMsg, '已退出登录', 'ok');
  if (pollTimer) clearInterval(pollTimer);
  authPanel.classList.remove('hidden');
  appPanel.classList.add('hidden');
  adminPanel.classList.add('hidden');
  switchAppView('view-media');
});

document.getElementById('refresh-media').addEventListener('click', () => {
  loadMedia().catch((err) => setMsg(appMsg, err.message, 'error'));
});

if (mediaSelectAll) {
  mediaSelectAll.addEventListener('change', () => {
    const allBoxes = Array.from(document.querySelectorAll('.media-select'));
    const checked = Boolean(mediaSelectAll.checked);
    selectedMediaIds.clear();
    allBoxes.forEach((box) => {
      box.checked = checked;
      if (checked) selectedMediaIds.add(Number(box.dataset.mediaId));
    });
    updateBatchDeleteButton();
    syncMediaSelectAllState(allBoxes.length);
  });
}

if (batchDeleteMediaBtn) {
  batchDeleteMediaBtn.addEventListener('click', () => {
    batchDeleteSelectedMedia().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

if (batchRebuildMediaBtn) {
  batchRebuildMediaBtn.addEventListener('click', () => {
    batchReconstructSelectedMedia().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

if (jobsSelectAll) {
  jobsSelectAll.addEventListener('change', () => {
    const allBoxes = Array.from(document.querySelectorAll('.job-select'));
    const checked = Boolean(jobsSelectAll.checked);
    selectedJobIds.clear();
    allBoxes.forEach((box) => {
      box.checked = checked;
      if (checked) selectedJobIds.add(String(box.dataset.jobId));
    });
    updateBatchDeleteJobsButton();
    syncJobSelectAllState(allBoxes.length);
  });
}

if (batchDeleteJobsBtn) {
  batchDeleteJobsBtn.addEventListener('click', () => {
    batchDeleteSelectedJobs().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

if (batchCancelJobsBtn) {
  batchCancelJobsBtn.addEventListener('click', () => {
    batchCancelSelectedJobs().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

document.getElementById('refresh-jobs').addEventListener('click', () => {
  Promise.all([
    loadJobs(),
    loadAttentionJobOptions(),
  ]).catch((err) => setMsg(appMsg, err.message, 'error'));
});

if (jobsFilterForm) {
  jobsFilterForm.addEventListener('submit', (e) => {
    e.preventDefault();
    loadJobs().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

if (jobsStatusFilter) {
  jobsStatusFilter.addEventListener('change', () => {
    loadJobs().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

if (jobsScenarioFilter) {
  jobsScenarioFilter.addEventListener('change', () => {
    loadJobs().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

if (jobsCreatedFrom) {
  jobsCreatedFrom.addEventListener('change', () => {
    loadJobs().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

if (jobsCreatedTo) {
  jobsCreatedTo.addEventListener('change', () => {
    loadJobs().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

if (jobsSearchClear) {
  jobsSearchClear.addEventListener('click', () => {
    if (jobsSearchFilter) jobsSearchFilter.value = '';
    if (jobsStatusFilter) jobsStatusFilter.value = '';
    if (jobsScenarioFilter) jobsScenarioFilter.value = '';
    if (jobsCreatedFrom) jobsCreatedFrom.value = '';
    if (jobsCreatedTo) jobsCreatedTo.value = '';
    loadJobs().catch((err) => setMsg(appMsg, err.message, 'error'));
  });
}

document.getElementById('refresh-users').addEventListener('click', () => {
  loadUsersAdmin().catch((err) => setMsg(appMsg, err.message, 'error'));
});

document.getElementById('refresh-audit-logs').addEventListener('click', () => {
  loadAuditLogs().catch((err) => setMsg(appMsg, err.message, 'error'));
});

document.getElementById('close-frame-modal').addEventListener('click', () => {
  closeFramePreviewModal();
});

document.getElementById('close-image-modal').addEventListener('click', () => {
  closeImagePreviewModal();
});

document.getElementById('play-frames').addEventListener('click', () => {
  toggleFramePlayback();
});

document.getElementById('prev-frame').addEventListener('click', () => {
  if (!framePreviewState.entries.length) return;
  if (framePreviewState.currentIndex > 0) {
    framePreviewState.currentIndex -= 1;
    showCurrentFrame();
  }
});

document.getElementById('next-frame').addEventListener('click', () => {
  if (!framePreviewState.entries.length) return;
  if (framePreviewState.currentIndex < framePreviewState.entries.length - 1) {
    framePreviewState.currentIndex += 1;
    showCurrentFrame();
  }
});

document.getElementById('prev-page').addEventListener('click', async () => {
  if (!framePreviewState.jobId || framePreviewState.page <= 1) return;
  try {
    await loadFramePreviewPage(framePreviewState.page - 1);
  } catch (err) {
    framePreviewStatus.textContent = err.message;
  }
});

document.getElementById('next-page').addEventListener('click', async () => {
  if (!framePreviewState.jobId) return;
  const totalPages = Math.max(1, Math.ceil(framePreviewState.total / framePreviewState.pageSize));
  if (framePreviewState.page >= totalPages) return;
  try {
    await loadFramePreviewPage(framePreviewState.page + 1);
  } catch (err) {
    framePreviewStatus.textContent = err.message;
  }
});

appNavButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const viewId = btn.dataset.view;
    if (!viewId) return;
    if (viewId === 'view-admin' && (!currentUser || !currentUser.is_admin)) return;
    switchAppView(viewId);
    if (viewId === 'view-attention') {
      loadAttentionJobOptions().catch(() => {});
    }
  });
});

tabLogin.addEventListener('click', () => switchAuthTab('login'));
tabRegister.addEventListener('click', () => switchAuthTab('register'));

switchAuthTab('login');
drawAttentionCurve({ points: [] });
window.addEventListener('resize', () => {
  drawAttentionCurve(lastAttentionCurveData);
});
enterApp();
