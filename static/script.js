/**
 * WHT Watermark Lab — Frontend Script
 * Handles: file uploads, API calls, image display, metrics visualization
 */

'use strict';

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
const state = {
  coverFile: null,
  wmFile: null,
  embedded: false,
  attacked: false,
};

// ─────────────────────────────────────────────
// DOM refs
// ─────────────────────────────────────────────
const $ = id => document.getElementById(id);

const inpCover    = $('inp-cover');
const inpWm       = $('inp-wm');
const btnEmbed    = $('btn-embed');
const btnAttack   = $('btn-attack');
const btnExtract  = $('btn-extract');
const selAttack   = $('sel-attack');
const chkAttacked = $('chk-use-attacked');
const loader      = $('loader');
const loaderMsg   = $('loader-msg');
const toast       = $('toast');

// ─────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────

function showLoader(msg = 'Processing…') {
  loaderMsg.textContent = msg;
  loader.classList.remove('hidden');
}

function hideLoader() {
  loader.classList.add('hidden');
}

let toastTimer = null;
function showToast(msg, type = 'info') {
  toast.textContent = msg;
  toast.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
}

/** Set a result image by base64 string, hide placeholder */
function setImage(imgId, phId, b64) {
  const img = $(imgId);
  const ph  = $(phId);
  img.src = `data:image/png;base64,${b64}`;
  img.classList.remove('hidden');
  ph.classList.add('hidden');
}

/** Activate a step pill in the header nav */
function activateStep(num) {
  document.querySelectorAll('.step-pill').forEach(p => {
    p.classList.toggle('active', Number(p.dataset.step) === num);
  });
}

// ─────────────────────────────────────────────
// File upload handlers
// ─────────────────────────────────────────────

function handleFileUpload(file, previewImgId, previewStripId, previewNameId) {
  if (!file) return;
  const strip = $(previewStripId);
  const img   = $(previewImgId);
  const name  = $(previewNameId);

  const reader = new FileReader();
  reader.onload = e => {
    img.src = e.target.result;
    name.textContent = file.name;
    strip.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

inpCover.addEventListener('change', () => {
  state.coverFile = inpCover.files[0] || null;
  handleFileUpload(state.coverFile, 'prev-cover', 'prev-cover-strip', 'prev-cover-name');
  // Show in display panel immediately
  if (state.coverFile) {
    const reader = new FileReader();
    reader.onload = e => {
      $('disp-cover').src = e.target.result;
      $('disp-cover').classList.remove('hidden');
      $('ph-cover').classList.add('hidden');
    };
    reader.readAsDataURL(state.coverFile);
  }
  activateStep(1);
});

inpWm.addEventListener('change', () => {
  state.wmFile = inpWm.files[0] || null;
  handleFileUpload(state.wmFile, 'prev-wm', 'prev-wm-strip', 'prev-wm-name');
  if (state.wmFile) {
    const reader = new FileReader();
    reader.onload = e => {
      $('disp-wm-orig').src = e.target.result;
      $('disp-wm-orig').classList.remove('hidden');
      $('ph-wm-orig').classList.add('hidden');
    };
    reader.readAsDataURL(state.wmFile);
  }
});

// Drag & drop for zones
['zone-cover', 'zone-wm'].forEach(zoneId => {
  const zone = $(zoneId);
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--accent)'; });
  zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith('image/')) return;
    if (zoneId === 'zone-cover') {
      state.coverFile = file;
      // Manually trigger same flow
      const dt = new DataTransfer();
      dt.items.add(file);
      inpCover.files = dt.files;
      inpCover.dispatchEvent(new Event('change'));
    } else {
      state.wmFile = file;
      const dt = new DataTransfer();
      dt.items.add(file);
      inpWm.files = dt.files;
      inpWm.dispatchEvent(new Event('change'));
    }
  });
});

// ─────────────────────────────────────────────
// Attack param UI switching
// ─────────────────────────────────────────────

const paramMap = {
  gaussian_noise:  'param-noise',
  jpeg_compression:'param-jpeg',
  cropping:        'param-crop',
  rotation:        'param-rot',
  scaling:         'param-scale',
};

selAttack.addEventListener('change', () => {
  Object.values(paramMap).forEach(id => $(id).classList.add('hidden'));
  const show = paramMap[selAttack.value];
  if (show) $(show).classList.remove('hidden');
});

// ─────────────────────────────────────────────
// Toggle: extract source label
// ─────────────────────────────────────────────

chkAttacked.addEventListener('change', () => {
  $('extract-source-label').textContent = chkAttacked.checked ? 'Attacked' : 'Watermarked';
});

// ─────────────────────────────────────────────
// Metrics display
// ─────────────────────────────────────────────

function updateMetric(id, value, unit, refDir = 'high') {
  const el = $(`mv-${id}`);
  const bar = $(`mb-${id}`);
  const card = $(`mc-${id}`);

  if (value === null || value === undefined) {
    el.textContent = '—';
    bar.style.width = '0%';
    card.className = 'metric-card';
    return;
  }

  el.textContent = typeof value === 'number' ? value.toFixed(4) : value;

  // Compute bar fill and color status
  let pct = 0;
  let status = 'neutral';

  if (id === 'psnr') {
    pct = Math.min(100, (value / 60) * 100);
    status = value >= 35 ? 'good' : value >= 30 ? 'warn' : 'bad';
    bar.style.background = value >= 35 ? 'var(--accent4)' : value >= 30 ? 'var(--accent5)' : 'var(--accent3)';
  } else if (id === 'ssim') {
    pct = value * 100;
    status = value >= 0.93 ? 'good' : value >= 0.85 ? 'warn' : 'bad';
    bar.style.background = value >= 0.93 ? 'var(--accent4)' : value >= 0.85 ? 'var(--accent5)' : 'var(--accent3)';
  } else if (id === 'nc') {
    pct = value * 100;
    status = value >= 0.90 ? 'good' : value >= 0.70 ? 'warn' : 'bad';
    bar.style.background = value >= 0.90 ? 'var(--accent4)' : value >= 0.70 ? 'var(--accent5)' : 'var(--accent3)';
  } else if (id === 'ber') {
    // BER: lower is better; bar shows inverse
    pct = Math.max(0, (1 - value) * 100);
    status = value <= 0.01 ? 'good' : value <= 0.1 ? 'warn' : 'bad';
    bar.style.background = value <= 0.01 ? 'var(--accent4)' : value <= 0.1 ? 'var(--accent5)' : 'var(--accent3)';
  }

  bar.style.width = `${pct}%`;
  card.className = `metric-card ${status}`;
  el.style.color = bar.style.background;
}

function resetMetrics() {
  ['psnr','ssim','nc','ber'].forEach(id => updateMetric(id, null));
}

// ─────────────────────────────────────────────
// API calls
// ─────────────────────────────────────────────

/** POST /embed */
btnEmbed.addEventListener('click', async () => {
  if (!state.coverFile) { showToast('Please upload a cover image first', 'error'); return; }
  if (!state.wmFile)    { showToast('Please upload a watermark image first', 'error'); return; }

  showLoader('Embedding watermark (WHT + chaotic encryption)…');
  activateStep(2);
  resetMetrics();

  const formData = new FormData();
  formData.append('cover_image', state.coverFile);
  formData.append('watermark_image', state.wmFile);

  try {
    const resp = await fetch('/embed', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) { showToast(`Error: ${data.error}`, 'error'); return; }

    setImage('disp-cover',      'ph-cover',      data.cover_b64);
    setImage('disp-wm-orig',    'ph-wm-orig',    data.watermark_b64);
    setImage('disp-watermarked','ph-watermarked', data.watermarked_b64);

    updateMetric('psnr', data.psnr);
    updateMetric('ssim', data.ssim);
    updateMetric('nc',   null);
    updateMetric('ber',  null);

    state.embedded = true;
    showToast(`✓ Embedded! PSNR=${data.psnr} dB · SSIM=${data.ssim}`, 'success');
  } catch (err) {
    showToast(`Network error: ${err.message}`, 'error');
  } finally {
    hideLoader();
  }
});

/** POST /attack */
btnAttack.addEventListener('click', async () => {
  if (!state.embedded) { showToast('Embed a watermark first', 'error'); return; }

  const attackType = selAttack.value;
  const params = getAttackParams(attackType);

  showLoader(`Applying ${attackType.replace('_',' ')} attack…`);
  activateStep(3);

  try {
    const resp = await fetch('/attack', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attack_type: attackType, params }),
    });
    const data = await resp.json();

    if (data.error) { showToast(`Error: ${data.error}`, 'error'); return; }

    setImage('disp-attacked',  'ph-attacked',  data.attacked_b64);
    setImage('disp-extracted', 'ph-extracted', data.extracted_wm_b64);

    updateMetric('nc',  data.nc);
    updateMetric('ber', data.ber);

    state.attacked = true;
    showToast(`⚡ Attack applied · NC=${data.nc} · BER=${data.ber}`, 'info');

    // Auto-enable "use attacked" toggle
    chkAttacked.checked = true;
    $('extract-source-label').textContent = 'Attacked';
  } catch (err) {
    showToast(`Network error: ${err.message}`, 'error');
  } finally {
    hideLoader();
  }
});

/** POST /extract */
btnExtract.addEventListener('click', async () => {
  if (!state.embedded) { showToast('Embed a watermark first', 'error'); return; }

  const useAttacked = chkAttacked.checked;
  showLoader('Extracting watermark…');
  activateStep(4);

  try {
    const resp = await fetch('/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_attacked: useAttacked }),
    });
    const data = await resp.json();

    if (data.error) { showToast(`Error: ${data.error}`, 'error'); return; }

    setImage('disp-extracted', 'ph-extracted', data.extracted_wm_b64);

    updateMetric('nc',  data.nc);
    updateMetric('ber', data.ber);

    showToast(`✓ Extracted! NC=${data.nc} · BER=${data.ber}`, 'success');
  } catch (err) {
    showToast(`Network error: ${err.message}`, 'error');
  } finally {
    hideLoader();
  }
});

// ─────────────────────────────────────────────
// Attack parameter helpers
// ─────────────────────────────────────────────

function getAttackParams(type) {
  switch (type) {
    case 'gaussian_noise':
      return { variance: parseFloat($('rng-noise').value) };
    case 'jpeg_compression':
      return { quality: parseInt($('rng-jpeg').value) };
    case 'cropping':
      return { percent: parseInt($('rng-crop').value) / 100 };
    case 'rotation':
      return { angle: parseInt($('rng-rot').value) };
    case 'scaling':
      return { scale: parseFloat($('rng-scale').value) };
    default:
      return {};
  }
}

// ─────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────

(function init() {
  // Show only first attack param panel
  Object.values(paramMap).forEach(id => $(id).classList.add('hidden'));
  $('param-noise').classList.remove('hidden');

  // Build lightbox DOM once
  initLightbox();
})();

// ─────────────────────────────────────────────
// Lightbox
// ─────────────────────────────────────────────

function initLightbox() {
  // Create overlay element
  const overlay = document.createElement('div');
  overlay.className = 'lightbox-overlay';
  overlay.id = 'lightbox';
  overlay.innerHTML = `
    <div class="lightbox-box">
      <button class="lightbox-close" id="lightbox-close" title="Close">✕</button>
      <img class="lightbox-img" id="lightbox-img" src="" alt="Preview" />
      <span class="lightbox-label" id="lightbox-label"></span>
    </div>
  `;
  document.body.appendChild(overlay);

  // Close on × button
  document.getElementById('lightbox-close').addEventListener('click', closeLightbox);

  // Close on clicking the dark backdrop (outside the box)
  overlay.addEventListener('click', e => {
    if (e.target === overlay) closeLightbox();
  });

  // Close on Escape key
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeLightbox();
  });

  // Attach click listeners to all current and future result-img elements
  // Use event delegation on document so dynamically-shown images are covered
  document.addEventListener('click', e => {
    if (e.target.classList.contains('result-img') && !e.target.classList.contains('hidden')) {
      // Find the label from the nearest img-card header
      const card = e.target.closest('.img-card');
      const label = card ? card.querySelector('.img-label')?.textContent : '';
      openLightbox(e.target.src, label);
    }
  });
}

function openLightbox(src, label = '') {
  const overlay = document.getElementById('lightbox');
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox-label').textContent = label.toUpperCase();
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  const overlay = document.getElementById('lightbox');
  overlay.classList.remove('open');
  document.body.style.overflow = '';
  // Small delay so fade-out finishes before clearing src
  setTimeout(() => {
    document.getElementById('lightbox-img').src = '';
  }, 260);
}