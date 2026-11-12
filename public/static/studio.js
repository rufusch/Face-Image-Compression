'use strict';
const byId = (id) => document.getElementById(id);
const sample = byId('sample');
const upload = byId('image');
const compare = byId('compare');
const sheet = byId('save-sheet');
const gallery = byId('gallery');
const viewer = byId('viewer');
const viewImage = byId('viewer-image');
const viewport = byId('viewport');
const archiveDialog = byId('archive');
let results = [];
let sourceName = 'image';
let scale = 1;
let viewerUrl;

for (let i = 1; i <= 400; i++) {
  sample.add(new Option('Olivetti face ' + String(i).padStart(3, '0'), i));
}

function status(element, message, error = false) {
  element.textContent = message;
  element.classList.toggle('error', error);
}

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || 'The request failed. Try again.');
  }
  return response;
}

const png = (data) => 'data:image/png;base64,' + data;

function download(href, filename) {
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  link.click();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  download(url, filename);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function button(label, action, className = 'button outline') {
  const element = document.createElement('button');
  element.type = 'button';
  element.className = className;
  element.textContent = label;
  element.addEventListener('click', action);
  return element;
}

function entryTitle(entry) {
  return entry.rank == null ? 'Original' : 'Rank ' + entry.rank;
}

function render(entries) {
  gallery.replaceChildren();
  for (const entry of entries) {
    const title = entryTitle(entry);
    const card = document.createElement('article');
    card.className = 'card';
    const heading = document.createElement('div');
    heading.className = 'card-head';
    const name = document.createElement('h3');
    name.textContent = title;
    const metric = document.createElement('span');
    const psnr = entry.metrics?.psnr;
    metric.textContent = entry.rank == null ? 'Reference' :
      psnr == null ? 'Exact pixels' : psnr.toFixed(1) + ' dB';
    heading.append(name, metric);
    const imageButton = button('', () => showViewer(png(entry.preview), title, entry.filename), 'thumb');
    imageButton.setAttribute('aria-label', 'Expand ' + title);
    imageButton.disabled = !entry.preview;
    if (entry.preview) {
      const img = document.createElement('img');
      img.src = png(entry.preview);
      img.alt = title + ' image';
      imageButton.append(img);
    }
    const actions = document.createElement('div');
    actions.className = 'card-actions';
    if (entry.preview) {
      actions.append(button('Save PNG', () => download(png(entry.preview), entry.filename)));
      if (entry.download) actions.append(button('Save archive', (event) => saveArchive(entry, event.currentTarget)));
    }
    card.append(heading, imageButton, actions);
    gallery.append(card);
  }
}

function showAnalysis(data) {
  const content = byId('analysis-content');
  content.replaceChildren();
  const lines = [
    data.width + ' × ' + data.height + ' pixels · ' + data.channels + ' channel(s)',
    'A = UΣVᵀ. Each reconstruction keeps the first r singular values and vectors.',
    'At rank r, each channel uses Uᵣ (' + data.height + ' × r), Σᵣ (r × r), and Vᵣᵀ (r × ' + data.width + ').',
    'For a channel with m rows and n columns, the retained factors contain r(m+n+1) values.',
    'Ranks used: ' + data.comparisons.map(entry => entry.rank === entry.requested_rank
      ? entry.rank : entry.rank + ' (requested ' + entry.requested_rank + ')').join(', ') + '.',
  ];
  for (const line of lines) {
    const paragraph = document.createElement('p');
    paragraph.textContent = line;
    content.append(paragraph);
  }
  if (data.analysis) {
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'Rank optimization, resizing and smoothing details';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(data.analysis, (key, value) => Array.isArray(value) && value.length > 25 ? 'See backend analysis data' : value, 2);
    pre.style.overflow = 'auto';
    details.append(summary, pre);
    content.append(details);
  }
  byId('analysis').hidden = false;
}

async function compareImages(event) {
  event?.preventDefault();
  if (!upload.files.length && !sample.value) return;
  compare.disabled = true;
  sheet.disabled = true;
  upload.disabled = true;
  sample.disabled = true;
  status(byId('status'), 'Computing the five reconstructions…');
  try {
    const file = upload.files[0];
    const image = file || await (await request('/api/samples/' + sample.value)).blob();
    const name = file ? file.name.replace(/\.[^.]+$/, '') : 'olivetti-' + String(sample.value).padStart(3, '0');
    const form = new FormData(byId('compare-form'));
    form.set('image', image, file?.name || name + '.png');
    form.set('gallery', '1');
    const data = await (await request('/api/compress', {method:'POST', body:form})).json();
    sourceName = name;
    results = [
      {preview:data.original, filename:name + '-original.png'},
      ...data.comparisons.map(entry => ({...entry, reference:data.original, sourceName:name,
        filename:name + '-r' + entry.rank + '.png'})),
    ];
    render(results);
    showAnalysis(data);
    status(byId('status'), 'Click an image to inspect its pixels.');
  } catch (error) {
    status(byId('status'), error.message, true);
  } finally {
    compare.disabled = !upload.files.length && !sample.value;
    upload.disabled = false;
    sample.disabled = false;
    sheet.disabled = results.length === 0;
  }
}

async function saveArchive(entry, control) {
  control.disabled = true;
  status(byId('status'), 'Saving the selected rank and pixel differences…');
  try {
    const form = new FormData();
    form.set('factors', await (await fetch('data:application/octet-stream;base64,' + entry.download)).blob(), 'factors.npz');
    form.set('reference', await (await fetch(png(entry.reference))).blob(), 'reference.png');
    const archive = await (await request('/api/archives', {method:'POST', body:form})).blob();
    downloadBlob(archive, entry.sourceName + '-r' + entry.rank + '.npz');
    status(byId('status'), 'Archive saved. Open it to recover the gallery original.');
  } catch (error) {
    status(byId('status'), error.message, true);
  } finally {
    control.disabled = false;
  }
}

function setZoom(nextScale) {
  if (!viewImage.naturalWidth) return;
  scale = Math.min(12, Math.max(.01, nextScale));
  viewImage.style.width = Math.round(viewImage.naturalWidth * scale) + 'px';
  viewImage.style.height = Math.round(viewImage.naturalHeight * scale) + 'px';
}

function fitViewer() {
  if (viewImage.naturalWidth) setZoom(Math.min(
    viewport.clientWidth / viewImage.naturalWidth,
    viewport.clientHeight / viewImage.naturalHeight
  ));
}

function showViewer(url, title, filename, objectUrl = false) {
  if (viewerUrl) URL.revokeObjectURL(viewerUrl);
  viewerUrl = objectUrl ? url : undefined;
  byId('viewer-title').textContent = title;
  viewImage.alt = title;
  const save = byId('viewer-download');
  save.href = url;
  save.download = filename;
  viewImage.onload = fitViewer;
  viewImage.src = url;
  if (!viewer.open) viewer.showModal();
  fitViewer();
}

byId('fit').onclick = fitViewer;
byId('actual').onclick = () => setZoom(1);
byId('zoom-out').onclick = () => setZoom(scale / 1.25);
byId('zoom-in').onclick = () => setZoom(scale * 1.25);
viewer.addEventListener('close', () => {
  if (viewerUrl) URL.revokeObjectURL(viewerUrl);
  viewerUrl = undefined;
  viewImage.removeAttribute('src');
});

for (const id of ['archive-trigger', 'archive-method']) {
  byId(id).addEventListener('click', () => archiveDialog.showModal());
}

byId('archive-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const submit = byId('archive-submit');
  const statusLine = byId('archive-status');
  const file = byId('archive-file').files[0];
  if (!file) return;
  submit.disabled = true;
  status(statusLine, 'Reconstructing image…');
  try {
    const blob = await (await request('/api/archives/open', {method:'POST', body:new FormData(event.currentTarget)})).blob();
    const name = file.name.replace(/\.npz$/i, '');
    status(statusLine, '');
    archiveDialog.close();
    showViewer(URL.createObjectURL(blob), 'Reconstructed archive', name + '-reconstructed.png', true);
  } catch (error) {
    status(statusLine, error.message, true);
  } finally {
    submit.disabled = false;
  }
});

sheet.addEventListener('click', async () => {
  const entries = results;
  const filename = sourceName + '-comparison.png';
  const canvas = document.createElement('canvas');
  canvas.width = 1500;
  canvas.height = 1040;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fbfaf6';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#171717';
  ctx.font = 'bold 26px sans-serif';
  for (const [i, entry] of entries.entries()) {
    const image = new Image();
    image.src = png(entry.preview);
    await image.decode();
    const x = (i % 3) * 500, y = Math.floor(i / 3) * 520;
    const factor = Math.min(460 / image.width, 435 / image.height);
    const w = image.width * factor, h = image.height * factor;
    ctx.fillText(entryTitle(entry), x + 20, y + 42);
    ctx.imageSmoothingEnabled = factor < 1;
    ctx.drawImage(image, x + (500 - w) / 2, y + 60 + (435 - h) / 2, w, h);
  }
  canvas.toBlob((blob) => { if (blob) downloadBlob(blob, filename); });
});

sample.addEventListener('change', () => {
  upload.value = '';
  compare.disabled = !sample.value;
});
upload.addEventListener('change', () => {
  if (upload.files.length) sample.value = '';
  compare.disabled = !upload.files.length && !sample.value;
});
byId('compare-form').addEventListener('submit', compareImages);
