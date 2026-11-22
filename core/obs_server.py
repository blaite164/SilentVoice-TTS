"""
core/obs_server.py — Вбудований локальний HTTP + WebSocket сервер (127.0.0.1) для:
1. OBS Browser Source субтитрів з плавними анімаціями, миттєвим оновленням стилю та (опційно) показом набору наживо.
2. Локального HTTP API для керування зі Stream Deck або зовнішніх скриптів (/api/speak, /api/toggle_input, /api/quick/1 тощо).
"""

from __future__ import annotations

import asyncio
import copy
import json
import socket
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from aiohttp import WSMsgType, web
from PySide6.QtCore import QObject, Signal

from core.config import OBS_TEMPLATES_DIR, config
from core.obs_animations import (
    ENTRY_ANIMATIONS,
    MOTION_ANIMATIONS,
    get_entry_spec,
    get_motion_spec,
    get_variant_defaults,
    get_variant_key,
    resolve_anim_params,
)

OBS_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="__SV_LANG__">
<head>
  <meta charset="UTF-8">
  <title>SilentVoice OBS Subtitles</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700;800&family=Inter:wght@500;600;700&family=Rubik:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 100%;
      height: 100%;
      background: transparent !important;
      overflow: hidden;
      font-family: 'Montserrat', 'Segoe UI', sans-serif;
    }
    body.preview-mode {
      background:
        radial-gradient(circle at 50% 30%, rgba(124, 92, 255, 0.16), transparent 60%),
        linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
    }
    #preview-bar {
      display: none;
      position: fixed;
      top: 16px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 50;
      align-items: center;
      gap: 12px;
      padding: 10px 16px;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.25);
      color: #E2E8F0;
      font-size: 13px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      pointer-events: auto;
    }
    body.preview-mode #preview-bar {
      display: flex;
    }
    #preview-test-btn {
      background: #7C5CFF;
      color: #FFFFFF;
      border: none;
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
    }
    #preview-test-btn:hover {
      background: #6D4AFF;
    }
    #preview-save-btn {
      background: #16805D;
      color: #FFFFFF;
      border: none;
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
    }
    #preview-save-btn:hover { background: #10916A; }
    #preview-save-btn:disabled { opacity: 0.55; cursor: default; }
    #preview-save-status { color: #A7F3D0; font-size: 11px; }
    #preview-controls {
      display: none;
    }
    body.preview-mode #preview-controls {
      position: fixed;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      grid-template-rows: auto minmax(0, 1fr);
      gap: 14px;
      top: 76px;
      left: 18px;
      right: 18px;
      bottom: 34%;
      z-index: 20;
      min-height: 0;
      pointer-events: auto;
    }
    .preview-shared-controls {
      grid-column: 1 / -1;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 18px;
      min-width: 0;
      padding: 9px 12px;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.25);
      color: #E2E8F0;
      font-size: 11px;
      pointer-events: auto;
    }
    .preview-mode-switch {
      display: inline-flex;
      padding: 3px;
      border-radius: 9px;
      background: #0B1220;
      border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .preview-mode-btn {
      border: 0;
      border-radius: 6px;
      padding: 6px 10px;
      color: #CBD5E1;
      background: transparent;
      font: inherit;
      cursor: pointer;
    }
    .preview-mode-btn.active { background: #6D5DF5; color: #fff; }
    .preview-check {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      white-space: nowrap;
    }
    .preview-check input { accent-color: #8B7BFF; }
    .preview-panel {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      padding: 16px;
      border-radius: 14px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.25);
      color: #E2E8F0;
      box-shadow: 0 12px 30px rgba(0,0,0,0.24);
      scrollbar-width: thin;
    }
    .preview-panel h2 {
      margin: 0 0 4px;
      font-size: 15px;
      font-weight: 700;
    }
    .preview-panel p {
      margin: 0 0 12px;
      color: #94A3B8;
      font-size: 11px;
    }
    .preview-field {
      display: grid;
      gap: 6px;
      margin: 10px 0;
      font-size: 11px;
      color: #CBD5E1;
    }
    .preview-field-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .preview-value {
      color: #A5B4FC;
      font-variant-numeric: tabular-nums;
    }
    .preview-select {
      width: 100%;
      min-height: 34px;
      padding: 7px 9px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 8px;
      background: #111827;
      color: #F1F5F9;
      font: inherit;
    }
    .preview-range {
      width: 100%;
      accent-color: #8B7BFF;
    }
    .preview-empty {
      padding: 10px 0;
      color: #94A3B8;
      font-size: 11px;
    }
    .preview-reset-btn {
      margin-top: 8px;
      padding: 7px 10px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 8px;
      background: rgba(30, 41, 59, 0.72);
      color: #CBD5E1;
      font: inherit;
      cursor: pointer;
    }
    .preview-reset-btn:hover { background: rgba(51, 65, 85, 0.9); color: #fff; }
    .preview-shared-controls .preview-reset-btn { margin: 0 0 0 auto; white-space: nowrap; }
    #stage {
      position: absolute;
      inset: 0;
      display: flex;
      justify-content: center;
      align-items: flex-end;
      padding: 28px 48px;
      pointer-events: none;
    }
    body.preview-mode #stage {
      inset: auto 0 0;
      height: 32%;
      padding: 12px 28px 24px;
      align-items: flex-end;
    }
    @media (max-width: 700px) {
      body.preview-mode #preview-bar { top: 8px; gap: 6px; padding: 7px 9px; font-size: 10px; }
      body.preview-mode #preview-test-btn { padding: 6px 8px; font-size: 10px; }
      body.preview-mode #preview-save-btn { padding: 6px 8px; font-size: 10px; }
      body.preview-mode #preview-controls { top: 58px; left: 8px; right: 8px; bottom: 36%; gap: 8px; }
      .preview-shared-controls { gap: 7px 10px; padding: 7px 8px; font-size: 10px; }
      .preview-mode-btn { padding: 5px 7px; }
      .preview-panel { padding: 10px; }
      .preview-panel h2 { font-size: 12px; }
      body.preview-mode #stage { height: 34%; padding: 8px 12px 16px; }
    }
    #subtitle-box {
      max-width: 92%;
      max-height: calc(100% - 20px);
      perspective: 1000px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      opacity: 0;
      transform: translateY(12px) scale(0.97);
      transition: opacity 0.45s cubic-bezier(0.22, 1, 0.36, 1), transform 0.45s cubic-bezier(0.22, 1, 0.36, 1);
      will-change: opacity, transform;
      overflow: visible;
    }
    #subtitle-box.waapi-entry.visible {
      opacity: 1;
      transform: none;
      transition: none;
    }
    #subtitle-entry {
      max-width: 100%;
      max-height: 100%;
      display: flex;
      align-items: stretch;
      justify-content: center;
      transform-style: preserve-3d;
    }
    #subtitle-panel {
      max-width: 100%;
      max-height: 100%;
      width: fit-content;
      min-height: 40px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      border-radius: 14px;
      padding: 4px 10px;
      overflow: visible;
      transform-style: preserve-3d;
    }
    #subtitle-box.visible {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
    #subtitle-box.live-typing #subtitle-panel {
      border: 1.5px dashed rgba(124, 92, 255, 0.65);
    }
    #text-wrap {
      flex: 0 1 auto;
      width: fit-content;
      max-width: 100%;
      min-height: 0;
      overflow-y: hidden;
      overflow-x: hidden;
      scroll-behavior: smooth;
      display: block;
      line-height: 1.38;
      word-break: break-word;
      white-space: pre-wrap;
      transition: -webkit-mask-image 0.2s ease, mask-image 0.2s ease;
    }
    #text-wrap.has-overflow {
      -webkit-mask-image: linear-gradient(to bottom, transparent 0%, rgba(0, 0, 0, 0.2) 14px, rgba(0, 0, 0, 0.75) 30px, black 50px, black 100%);
      mask-image: linear-gradient(to bottom, transparent 0%, rgba(0, 0, 0, 0.2) 14px, rgba(0, 0, 0, 0.75) 30px, black 50px, black 100%);
    }
    #subtitle-text {
      display: inline-block;
      color: #FFFFFF;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0.015em;
      perspective: 700px;
    }
    .anim-outer, .anim-inner {
      display: inline-block;
      transform-style: preserve-3d;
    }
    .anim-outer.mask-rise-window {
      overflow: hidden;
      vertical-align: bottom;
    }
    .anim-space {
      display: inline;
    }
    #cursor {
      display: inline-block;
      color: #9B82FF;
      font-size: 24px;
      margin-left: 3px;
      opacity: 0;
    }
    #cursor.active {
      opacity: 1;
      animation: blink 0.75s infinite ease-in-out;
    }
    @keyframes blink {
      0%, 100% { opacity: 0.95; }
      50% { opacity: 0.1; }
    }
    .word-span {
      display: inline-block;
      animation: wordPop 0.28s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }
    @keyframes wordPop {
      from { opacity: 0; transform: translateY(8px) scale(0.94); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .word-bounce {
      animation: wordBounce 0.48s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }
    @keyframes wordBounce {
      0% { opacity: 0; transform: translateY(14px) scale(0.75); }
      65% { opacity: 1; transform: translateY(-4px) scale(1.06); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .phrase-smooth-in {
      animation: phraseSmoothIn 0.48s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }
    @keyframes phraseSmoothIn {
      from { opacity: 0; transform: translateY(10px) scale(0.98); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .phrase-blur-in {
      animation: phraseBlurIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }
    @keyframes phraseBlurIn {
      from { opacity: 0; filter: blur(10px); transform: scale(1.04); }
      to { opacity: 1; filter: blur(0); transform: scale(1); }
    }
    .text-glitch {
      animation: glitchIn 0.38s steps(2, end) forwards;
    }
    @keyframes glitchIn {
      0% { opacity: 0; transform: translate(-4px, 1px) skewX(-6deg); filter: drop-shadow(2px 0 #ff2a85) drop-shadow(-2px 0 #00e5ff); }
      35% { opacity: 0.9; transform: translate(3px, -1px) skewX(4deg); filter: drop-shadow(-2px 0 #ff2a85) drop-shadow(2px 0 #00e5ff); }
      70% { opacity: 1; transform: translate(-1px, 0) skewX(-2deg); filter: drop-shadow(1px 0 #00e5ff); }
      100% { opacity: 1; transform: none; filter: none; }
    }
    /* Живі ефекти руху тексту (окремо від появи) */
    .motion-levitation {
      animation: motionLevitate 2.8s ease-in-out infinite;
    }
    @keyframes motionLevitate {
      0%, 100% { transform: translateY(0px); }
      50% { transform: translateY(-6px); }
    }
    .motion-pulse {
      animation: motionPulse 2.2s ease-in-out infinite;
    }
    @keyframes motionPulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.035); }
    }
    .motion-neon_glow {
      animation: motionNeon 2.2s ease-in-out infinite;
    }
    @keyframes motionNeon {
      0%, 100% { filter: drop-shadow(0 0 4px rgba(155, 130, 255, 0.35)); }
      50% { filter: drop-shadow(0 0 14px rgba(155, 130, 255, 0.95)); }
    }
    #text-wrap.motion-rainbow #subtitle-text {
      animation: motionRainbow 4.5s linear infinite;
    }
    @keyframes motionRainbow {
      0% { color: #FF7EB3; }
      20% { color: #FFD166; }
      40% { color: #6EE7B7; }
      60% { color: #60A5FA; }
      80% { color: #C084FC; }
      100% { color: #FF7EB3; }
    }
    .word-wave-loop {
      display: inline-block;
      animation: wordWaveLoop 1.6s ease-in-out infinite;
    }
    @keyframes wordWaveLoop {
      0%, 100% { transform: translateY(0px) rotate(0deg); }
      25% { transform: translateY(-4px) rotate(-1.2deg); }
      75% { transform: translateY(3px) rotate(1.2deg); }
    }
  </style>
</head>
<body>
  <div id="preview-bar">
    <span id="preview-status"></span>
    <button id="preview-test-btn" type="button"></button>
    <button id="preview-save-btn" type="button" data-i18n="obs.preview.save">Зберегти в програму</button>
    <span id="preview-save-status" role="status" aria-live="polite"></span>
  </div>
  <section id="preview-controls" aria-label="" data-i18n-aria-label="obs.preview.controls_aria">
    <div class="preview-shared-controls">
      <div class="preview-mode-switch" role="group" aria-label="" data-i18n-aria-label="obs.preview.mode_aria">
        <button class="preview-mode-btn" type="button" data-preview-style="text_only" data-i18n="obs.preview.text_only">Без фону</button>
        <button class="preview-mode-btn" type="button" data-preview-style="boxed" data-i18n="obs.preview.boxed">У блоці</button>
      </div>
      <label class="preview-check"><input id="preview-animate-entry-block" type="checkbox"><span data-i18n="obs.preview.animate_entry_block">Анімувати блок разом із появою</span></label>
      <label class="preview-check"><input id="preview-animate-motion-block" type="checkbox"><span data-i18n="obs.preview.animate_motion_block">Застосовувати рух до блоку</span></label>
      <button id="preview-animation-reset-btn" class="preview-reset-btn" type="button" data-i18n="obs.preview.reset_animations">↺ Скинути появу й рух</button>
    </div>
    <div class="preview-panel">
      <h2 data-i18n="obs.preview.motion_title">Жива анімація</h2>
      <p data-i18n="obs.preview.motion_description">Оберіть рух, який діятиме після появи або одночасно з нею.</p>
      <label class="preview-field"><span data-i18n="obs.preview.motion_label">Анімація руху</span>
        <select id="preview-motion-select" class="preview-select"></select>
      </label>
      <div id="preview-motion-controls"></div>
    </div>
    <div class="preview-panel">
      <h2 data-i18n="obs.preview.entry_title">Анімація появи</h2>
      <p data-i18n="obs.preview.entry_description">Оберіть ефект і налаштуйте його параметри.</p>
      <label class="preview-field"><span data-i18n="obs.preview.entry_label">Ефект появи</span>
        <select id="preview-entry-select" class="preview-select"></select>
      </label>
      <div id="preview-entry-controls"></div>
    </div>
  </section>
  <div id="stage">
    <div id="subtitle-box">
      <div id="subtitle-entry">
        <div id="subtitle-panel">
          <!-- Без пробілів/переносів усередині #text-wrap: white-space: pre-wrap зберігає їх як порожній рядок і відступ зліва -->
          <div id="text-wrap"><span id="subtitle-text"></span><span id="cursor">▌</span></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const I18N = __SV_I18N_JSON__;
    const ANIMATION_META = __SV_ANIM_META_JSON__;
    const boxEl = document.getElementById('subtitle-box');
    const entryEl = document.getElementById('subtitle-entry');
    const panelEl = document.getElementById('subtitle-panel');
    const textWrapEl = document.getElementById('text-wrap');
    const textEl = document.getElementById('subtitle-text');
    const cursorEl = document.getElementById('cursor');

    function updateScrollAndMask() {
      if (!textWrapEl) return;
      textWrapEl.scrollTop = textWrapEl.scrollHeight;
      if (currentStyle.top_fade !== false && textWrapEl.scrollHeight > textWrapEl.clientHeight + 4) {
        textWrapEl.classList.add('has-overflow');
      } else {
        textWrapEl.classList.remove('has-overflow');
      }
    }
    function reservePulsePadding(params) {
      const scale = Math.max(1, Number(params?.scaleMax || 1.08));
      const combinedScale = currentStyle.animate_motion_block ? scale * scale : scale;
      const textWidth = Math.max(textEl.scrollWidth, textEl.getBoundingClientRect().width);
      const basePad = Number(textWrapEl.dataset.basePaddingX || 0);
      const extraPad = Math.ceil(textWidth * (combinedScale - 1) / 2) + 3;
      textWrapEl.style.paddingLeft = `${basePad + extraPad}px`;
      textWrapEl.style.paddingRight = `${basePad + extraPad}px`;
    }
    window.addEventListener('resize', () => updateScrollAndMask());

    let currentStyle = {
      style: 'text_only',
      font_family: 'Montserrat',
      font_size: 18,
      text_color: '#FFFFFF',
      text_opacity: 100,
      outline_width: 3,
      shadow_blur: 10,
      shadow_enabled: true,
      bg_enabled: false,
      bg_color: '#13151F',
      border_color: '#7C5CFF',
      bg_opacity: 78,
      border_radius: 14,
      anim_style: 'typewriter',
      animate_block: true,
      animate_motion_block: false,
      motion_effect: 'none',
      fade_sec: 3.5,
      show_live_typing: false,
      show_cursor: false,
      cursor_char: '▌',
      cursor_color: '#9B82FF',
      top_fade: true
    };

    let timers = [];
    let fadeTimer = null;
    let previewSaveTimer = null;

    function clearTimers() {
      timers.forEach(t => clearTimeout(t));
      timers = [];
      if (fadeTimer) {
        clearTimeout(fadeTimer);
        fadeTimer = null;
      }
    }

    function hexToRgba(hex, alphaPct) {
      let c = (hex || '#12131A').replace('#', '');
      if (c.length === 3) c = c[0]+c[0]+c[1]+c[1]+c[2]+c[2];
      const num = parseInt(c, 16) || 0;
      const r = (num >> 16) & 255;
      const g = (num >> 8) & 255;
      const b = num & 255;
      return `rgba(${r}, ${g}, ${b}, ${(alphaPct / 100).toFixed(2)})`;
    }

    function applyStyle(s) {
      if (!s) return;
      currentStyle = Object.assign(currentStyle, s);
      syncPreviewControls();
      document.body.style.fontFamily = `'${currentStyle.font_family}', 'Montserrat', 'Segoe UI', sans-serif`;
      textEl.style.fontSize = currentStyle.font_size + 'px';
      cursorEl.style.fontSize = currentStyle.font_size + 'px';
      const txtOpacity = Number(currentStyle.text_opacity !== undefined ? currentStyle.text_opacity : 100);
      textEl.style.color = hexToRgba(currentStyle.text_color || '#FFFFFF', txtOpacity);

      const shadows = [];
      const w = Math.max(0, Math.min(16, Number(currentStyle.outline_width || 0)));
      const outAlpha = (0.94 * (txtOpacity / 100)).toFixed(2);
      const outCol = `rgba(0, 0, 0, ${outAlpha})`;
      if (w > 0) {
        // Кругла обводка: кілька кілець тіней із щільним кроком.
        // Старий варіант (8 тіней по квадрату) на великій товщині давав «рвані» кути й щілини.
        const radii = [];
        for (let r = w; r > 0.01; r -= 1.25) radii.push(r);
        const seen = new Set();
        radii.forEach(r => {
          const n = Math.max(8, Math.ceil((2 * Math.PI * r) / 1.1));
          for (let i = 0; i < n; i++) {
            const a = (2 * Math.PI * i) / n;
            const dx = (Math.cos(a) * r).toFixed(2);
            const dy = (Math.sin(a) * r).toFixed(2);
            const key = dx + ',' + dy;
            if (seen.has(key)) continue;
            seen.add(key);
            shadows.push(`${dx}px ${dy}px 0 ${outCol}`);
          }
        });
      }
      const shBlur = Number(currentStyle.shadow_blur !== undefined ? currentStyle.shadow_blur : (currentStyle.shadow_enabled ? 10 : 0));
      if (shBlur > 0) {
        const shAlpha = (0.85 * (txtOpacity / 100)).toFixed(2);
        shadows.push(`0 3px ${shBlur}px rgba(0,0,0,${shAlpha})`, `0 0 ${Math.max(4, Math.round(shBlur * 0.8))}px rgba(0,0,0,${shAlpha})`);
      }
      const isGradientMotion = !!(MOTION_DEFS[currentStyle.motion_effect] && MOTION_DEFS[currentStyle.motion_effect].isTextGradient);
      const shadowCss = shadows.length ? shadows.join(', ') : 'none';
      if (isGradientMotion) {
        const mDef = MOTION_DEFS[currentStyle.motion_effect];
        textEl.style.backgroundImage = mDef.gradient;
        textEl.style.backgroundSize = mDef.gradientSize || '300% 100%';
        textEl.style.webkitBackgroundClip = 'text';
        textEl.style.backgroundClip = 'text';
        textEl.style.color = 'transparent';
        textEl.style.textShadow = 'none';
        textEl.style.filter = (w > 0 || shBlur > 0) ? `drop-shadow(0 2px ${Math.max(2, Math.round(shBlur * 0.5))}px rgba(0,0,0,0.85))` : 'none';
      } else {
        textEl.style.backgroundImage = 'none';
        textEl.style.backgroundSize = '';
        textEl.style.webkitBackgroundClip = '';
        textEl.style.backgroundClip = '';
        textEl.style.filter = '';
        textEl.style.textShadow = shadowCss;
      }
      cursorEl.style.textShadow = shadowCss;
      // Товста обводка «наїжджала» на сусідні слова/рядки — додаємо трохи простору пропорційно товщині
      const fs = Number(currentStyle.font_size || 24);
      textEl.style.wordSpacing = `${Math.max(0, 2 * w + 2 - fs * 0.27).toFixed(1)}px`;
      textEl.style.lineHeight = `${(fs * 1.38 + Math.max(0, w - 2)).toFixed(1)}px`;

      const isBoxed = (currentStyle.style === 'boxed') || (currentStyle.style === undefined && currentStyle.bg_enabled);
      // Внутрішній запас у #text-wrap, щоб обводка, тінь та «пружина»/збільшення букв
      // не обрізались краями області прокрутки (раніше це виглядало як невидиме «полотно»).
      const fxPadY = Math.ceil(w + 5);
      const fxPadX = Math.ceil(w + 6);
      const waveHeight = Number(currentStyle.anim_params?.waveHeight || 28);
      const wavePadY = currentStyle.anim_style === 'wave-stagger' ? Math.ceil(waveHeight * 0.35) + 8 : 0;
      const safePadY = Math.max(fxPadY, wavePadY);
      const safePadX = isBoxed ? fxPadX : Math.max(fxPadX, w + Math.ceil(Math.min(24, shBlur * 0.8)));
      textWrapEl.dataset.basePaddingX = String(safePadX);
      if (isBoxed) {
        const bgOp = Number(currentStyle.bg_opacity !== undefined ? currentStyle.bg_opacity : 78);
        const br = Math.max(0, Number(currentStyle.border_radius !== undefined ? currentStyle.border_radius : 14));
        panelEl.style.background = hexToRgba(currentStyle.bg_color || '#13151F', bgOp);
        panelEl.style.boxShadow = '0 8px 28px rgba(0,0,0,0.55)';
        const borderOp = Math.max(35, Math.round(bgOp * 0.9));
        panelEl.style.border = `1.6px solid ${hexToRgba(currentStyle.border_color || '#7C5CFF', borderOp)}`;
        panelEl.style.borderRadius = `${br}px`;
        panelEl.style.padding = '3px 10px';
        panelEl.style.overflow = currentStyle.anim_style === 'wave-stagger' ? 'hidden' : 'visible';
        textWrapEl.style.padding = `${safePadY}px ${safePadX}px`;
      } else {
        panelEl.style.background = 'transparent';
        panelEl.style.boxShadow = 'none';
        panelEl.style.border = 'none';
        panelEl.style.padding = '0';
        panelEl.style.overflow = 'visible';
        textWrapEl.style.padding = `${safePadY}px ${safePadX}px`;
      }
      const showCur = !!currentStyle.show_cursor;
      cursorEl.style.display = showCur ? 'inline-block' : 'none';
      if (currentStyle.cursor_char !== undefined) {
        cursorEl.textContent = currentStyle.cursor_char;
      }
      cursorEl.style.color = currentStyle.cursor_color || '#9B82FF';
      applyMotionClass();
      updateScrollAndMask();
    }

    const ENTRY_DEFS = {
      'fade-in': {
        supportsSplit: true,
        defaults: { duration: 1000, easing: 'ease-out', split: 'words', stagger: 200 },
        getKeyframes: () => [{ opacity: 0 }, { opacity: 1 }]
      },
      'blur-focus': {
        supportsSplit: true,
        defaults: { duration: 850, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', split: 'words', stagger: 65, blur: 16, scaleFrom: 1.08 },
        getKeyframes: (p) => [
          { opacity: 0, filter: `blur(${p.blur}px)`, transform: `scale(${p.scaleFrom})` },
          { opacity: 1, filter: 'blur(0px)', transform: 'scale(1)' }
        ]
      },
      'zoom': {
        supportsSplit: true,
        defaults: { duration: 680, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', split: 'whole', stagger: 40, zoomType: 'in', scaleAmount: 0.45 },
        getKeyframes: (p) => {
          const startScale = p.zoomType === 'out'
            ? Number((1 + Number(p.scaleAmount || 0.45)).toFixed(2))
            : Number(Math.max(0.1, 1 - Number(p.scaleAmount || 0.45)).toFixed(2));
          return [
            { opacity: 0, transform: `scale(${startScale})` },
            { opacity: 1, transform: 'scale(1)' }
          ];
        }
      },
      'elastic-pop': {
        supportsSplit: true,
        defaults: { duration: 850, easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)', split: 'chars', stagger: 30, overshoot: 1.22 },
        getKeyframes: (p) => [
          { opacity: 0, transform: 'scale(0.3)', offset: 0 },
          { opacity: 1, transform: `scale(${p.overshoot})`, offset: 0.65 },
          { opacity: 1, transform: 'scale(0.95)', offset: 0.85 },
          { opacity: 1, transform: 'scale(1)', offset: 1 }
        ]
      },
      'flip-y': {
        supportsSplit: true,
        defaults: { duration: 800, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', split: 'chars', stagger: 35, angle: 90 },
        getKeyframes: (p) => [
          { opacity: 0, transform: `rotateY(${p.angle}deg)` },
          { opacity: 1, transform: 'rotateY(0deg)' }
        ]
      },
      'wipe-mask': {
        supportsSplit: false,
        defaults: { duration: 800, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', split: 'whole', stagger: 0, wipeStyle: 'center', scaleFrom: 0.94 },
        getKeyframes: (p) => p.wipeStyle === 'diagonal'
          ? [
              { clipPath: 'polygon(0 0, 0 0, 0 100%, 0 100%)', transform: `scale(${p.scaleFrom}) skewX(-10deg)`, opacity: 0.35 },
              { clipPath: 'polygon(0 0, 120% 0, 100% 100%, 0 100%)', transform: 'scale(1) skewX(0deg)', opacity: 1 }
            ]
          : [
              { clipPath: 'inset(0 50% 0 50%)', transform: `scale(${p.scaleFrom})`, opacity: 0.3 },
              { clipPath: 'inset(0 0% 0 0%)', transform: 'scale(1)', opacity: 1 }
            ]
      },
      'mask-rise': {
        supportsSplit: true,
        defaults: { duration: 750, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', split: 'words', stagger: 70, distance: 105 },
        getKeyframes: (p) => [
          { clipPath: 'inset(100% 0 0 0)', transform: `translateY(${p.distance}%)`, opacity: 0 },
          { clipPath: 'inset(0 0 0% 0)', transform: 'translateY(0%)', opacity: 1 }
        ]
      },
      'wave-stagger': {
        supportsSplit: true,
        defaults: { duration: 700, easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)', split: 'chars', stagger: 35, waveHeight: 28 },
        getKeyframes: (p) => [
          { opacity: 0, transform: `translateY(${p.waveHeight}px) scale(0.8)`, offset: 0 },
          { opacity: 1, transform: `translateY(-${Math.round(p.waveHeight * 0.35)}px) scale(1.06)`, offset: 0.65 },
          { opacity: 1, transform: 'translateY(0) scale(1)', offset: 1 }
        ]
      },
      'skew-slide': {
        supportsSplit: true,
        defaults: { duration: 700, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', split: 'whole', stagger: 45, distance: 70, skewDeg: 22 },
        getKeyframes: (p) => [
          { opacity: 0, transform: `translateX(-${p.distance}px) skewX(${p.skewDeg}deg)`, offset: 0 },
          { opacity: 1, transform: `translateX(4px) skewX(-${Math.round(p.skewDeg * 0.25)}deg)`, offset: 0.72 },
          { opacity: 1, transform: 'translateX(0) skewX(0deg)', offset: 1 }
        ]
      },
      'glitch-reveal': {
        supportsSplit: true,
        defaults: { duration: 700, easing: 'linear', split: 'whole', stagger: 35, glitchPalette: 'cyber', intensity: 5 },
        getKeyframes: (p) => {
          const palettes = { cyber: ['#00f5d4', '#ff006e'], rgb: ['#00e5ff', '#ff2a2a'], matrix: ['#22c55e', '#a3e635'] };
          const [c1, c2] = palettes[p.glitchPalette] || palettes.cyber;
          const d = Number(p.intensity || 5);
          const dHalf = Math.max(1, Math.round(d * 0.5));
          return [
            { opacity: 0, clipPath: 'inset(45% 0 40% 0)', transform: `translateX(-${d}px) skewX(3deg)`, textShadow: `${d}px 0 ${c1}, -${d}px 0 ${c2}`, filter: 'brightness(1.35)', offset: 0 },
            { opacity: 0.95, clipPath: 'inset(10% 0 60% 0)', transform: `translateX(${d}px) skewX(-2deg)`, textShadow: `-${d}px 0 ${c1}, ${d}px 0 ${c2}`, filter: 'brightness(1.65)', offset: 0.16 },
            { opacity: 0.4, clipPath: 'inset(58% 0 8% 0)', transform: `translateX(-${dHalf}px)`, textShadow: `${dHalf}px 0 ${c2}, -${dHalf}px 0 ${c1}`, filter: 'brightness(1.45)', offset: 0.32 },
            { opacity: 1, clipPath: 'inset(0% 0 0% 0)', transform: `translateX(${dHalf}px)`, textShadow: `-${dHalf}px 0 ${c1}, ${dHalf}px 0 ${c2}`, filter: 'brightness(1.25)', offset: 0.48 },
            { opacity: 0.75, clipPath: 'inset(22% 0 20% 0)', transform: 'translateX(-2px)', textShadow: `3px 0 ${c1}, -3px 0 ${c2}`, filter: 'brightness(1.15)', offset: 0.66 },
            { opacity: 1, clipPath: 'inset(0% 0 0% 0)', transform: 'translateX(1px)', textShadow: `-2px 0 ${c1}, 2px 0 ${c2}`, filter: 'brightness(1.08)', offset: 0.84 },
            { opacity: 1, clipPath: 'inset(0% 0 0% 0)', transform: 'translateX(0)', textShadow: '0 0 transparent, 0 0 transparent', filter: 'brightness(1)', offset: 1 }
          ];
        }
      },
      'neon-flicker': {
        supportsSplit: true,
        defaults: { duration: 1000, easing: 'ease-out', split: 'whole', stagger: 40, glowRadius: 22 },
        getKeyframes: (p) => [
          { opacity: 0, textShadow: 'none', offset: 0 },
          { opacity: 0.9, textShadow: `0 0 ${p.glowRadius}px rgba(99, 102, 241, 0.95)`, offset: 0.18 },
          { opacity: 0.15, textShadow: 'none', offset: 0.3 },
          { opacity: 1, textShadow: `0 0 ${Math.round(p.glowRadius * 1.2)}px rgba(129, 140, 248, 1)`, offset: 0.48 },
          { opacity: 0.45, textShadow: '0 0 6px rgba(99, 102, 241, 0.5)', offset: 0.62 },
          { opacity: 1, textShadow: `0 0 ${p.glowRadius}px rgba(99, 102, 241, 0.75)`, offset: 1 }
        ]
      }
    };

    const MOTION_DEFS = {
      'levitation': {
        supportsSplit: true,
        direction: 'alternate',
        defaults: { duration: 1800, split: 'whole', stagger: 120, easing: 'ease-in-out', amplitude: 14, trigger: 'after' },
        getKeyframes: (p) => {
          const half = Math.round(Number(p.amplitude || 14) / 2);
          return [{ transform: `translateY(${half}px)` }, { transform: `translateY(-${half}px)` }];
        }
      },
      'rainbow-flow': {
        supportsSplit: false,
        isTextGradient: true,
        gradient: 'linear-gradient(90deg, #ff0080, #ff8c00, #ffd700, #00e676, #00e5ff, #7c4dff, #ff0080)',
        gradientSize: '300% 100%',
        defaults: { duration: 3600, split: 'whole', stagger: 0, easing: 'linear', trigger: 'simultaneous' },
        getKeyframes: () => [{ backgroundPosition: '0% 50%' }, { backgroundPosition: '300% 50%' }]
      },
      'shimmer': {
        supportsSplit: false,
        isTextGradient: true,
        gradient: 'linear-gradient(110deg, #94a3b8 0%, #94a3b8 38%, #ffffff 50%, #94a3b8 62%, #94a3b8 100%)',
        gradientSize: '200% 100%',
        defaults: { duration: 2200, split: 'whole', stagger: 0, easing: 'linear', trigger: 'simultaneous' },
        getKeyframes: () => [{ backgroundPosition: '200% 0%' }, { backgroundPosition: '0% 0%' }]
      },
      'pulse-breathe': {
        supportsSplit: true,
        direction: 'alternate',
        defaults: { duration: 1100, split: 'whole', stagger: 70, easing: 'ease-in-out', scaleMax: 1.08, trigger: 'after' },
        getKeyframes: (p) => [{ transform: 'scale(1)' }, { transform: `scale(${p.scaleMax})` }]
      },
      'wave-loop': {
        supportsSplit: true,
        direction: 'alternate',
        defaults: { duration: 850, split: 'chars', stagger: 75, easing: 'ease-in-out', waveHeight: 10, trigger: 'after' },
        getKeyframes: (p) => {
          const half = Math.round(Number(p.waveHeight || 10) / 2);
          return [{ transform: `translateY(${half}px)` }, { transform: `translateY(-${half}px)` }];
        }
      },
      'neon-breathe': {
        supportsSplit: true,
        direction: 'alternate',
        defaults: { duration: 1400, split: 'whole', stagger: 70, easing: 'ease-in-out', glowMax: 28, trigger: 'after' },
        getKeyframes: (p) => [
          { textShadow: '0 0 6px rgba(99, 102, 241, 0.35)' },
          { textShadow: `0 0 ${p.glowMax}px rgba(129, 140, 248, 0.95), 0 0 ${Math.round(Number(p.glowMax || 28) * 1.5)}px rgba(236, 72, 153, 0.5)` }
        ]
      },
      'micro-jitter': {
        supportsSplit: true,
        defaults: { duration: 900, split: 'chars', stagger: 35, jitterPx: 2, trigger: 'after' },
        getKeyframes: (p) => {
          const j = Number(p.jitterPx || 2);
          return [
            { transform: 'translate(0, 0)', offset: 0 },
            { transform: `translate(-${j}px, ${j}px)`, offset: 0.2 },
            { transform: `translate(${j}px, -${j}px)`, offset: 0.4 },
            { transform: `translate(-${j}px, -${j}px)`, offset: 0.6 },
            { transform: `translate(${j}px, ${j}px)`, offset: 0.8 },
            { transform: 'translate(0, 0)', offset: 1 }
          ];
        }
      }
    };

    function applyMotionClass() {
      // Збережено для сумісності викликів; рух запускається в playSpeech разом з появою
    }

    function cancelAllTextAnimations() {
      if (textEl) {
        textEl.getAnimations({ subtree: true }).forEach(a => a.cancel());
      }
      if (boxEl) {
        boxEl.getAnimations().forEach(a => a.cancel());
      }
      if (panelEl) {
        panelEl.getAnimations().forEach(a => a.cancel());
      }
      if (entryEl) {
        entryEl.getAnimations().forEach(a => a.cancel());
      }
    }

    function hideSubtitles() {
      clearTimers();
      cancelAllTextAnimations();
      cursorEl.classList.remove('active');
      boxEl.classList.remove('visible', 'live-typing', 'waapi-entry');
      fadeTimer = setTimeout(() => {
        if (!boxEl.classList.contains('visible')) {
          textEl.innerHTML = '';
        }
      }, 500);
    }

    function showLiveTyping(text) {
      if (!currentStyle.show_live_typing) return;
      clearTimers();
      cancelAllTextAnimations();
      boxEl.classList.remove('waapi-entry');
      if (!text) {
        hideSubtitles();
        return;
      }
      textEl.textContent = text;
      boxEl.classList.add('visible', 'live-typing');
      if (currentStyle.show_cursor) cursorEl.classList.add('active');
      else cursorEl.classList.remove('active');
      updateScrollAndMask();
    }

    function buildSplitUnits(container, fullText, mode) {
      container.innerHTML = '';
      const units = [];
      const addUnit = (str) => {
        const outer = document.createElement('span');
        outer.className = 'anim-outer';
        const inner = document.createElement('span');
        inner.className = 'anim-inner';
        inner.textContent = str;
        outer.appendChild(inner);
        container.appendChild(outer);
        units.push({ outer, inner });
      };
      if (mode === 'whole') {
        addUnit(fullText);
        return units;
      }
      if (mode === 'words') {
        fullText.split(/(\s+)/).forEach(part => {
          if (/^\s+$/.test(part)) {
            container.appendChild(document.createTextNode(part));
          } else if (part.length > 0) {
            addUnit(part);
          }
        });
        return units;
      }
      Array.from(fullText).forEach(ch => {
        if (/\s/.test(ch)) {
          container.appendChild(document.createTextNode(ch));
        } else {
          addUnit(ch);
        }
      });
      return units;
    }

    function playSpeech(data) {
      clearTimers();
      cancelAllTextAnimations();
      boxEl.classList.remove('live-typing', 'waapi-entry');
      const fullText = data.text || '';
      const durationSec = Math.max(0.8, Number(data.duration || 2.5));
      if (!fullText) return;

      const animId = currentStyle.anim_style || 'typewriter';
      const eDef = ENTRY_DEFS[animId];
      if (eDef) boxEl.classList.add('waapi-entry');
      textEl.innerHTML = '';
      boxEl.classList.add('visible');

      const fadeMs = Math.max(500, Number(currentStyle.fade_sec || 3.5) * 1000);
      const scheduleHide = (afterMs) => {
        const t = setTimeout(() => {
          cursorEl.classList.remove('active');
          fadeTimer = setTimeout(() => {
            boxEl.classList.remove('waapi-entry');
            boxEl.classList.remove('visible');
          }, fadeMs);
        }, afterMs);
        timers.push(t);
      };

      const motionId = currentStyle.motion_effect || 'none';
      const mDef = motionId !== 'none' ? MOTION_DEFS[motionId] : null;
      const mp = mDef ? Object.assign({}, mDef.defaults, currentStyle.motion_params || {}) : null;
      const mTrigger = mp ? (mp.trigger || mDef.defaults.trigger || 'after') : 'after';

      if (mDef && mp && mTrigger === 'only') {
        textEl.textContent = fullText;
        const mKeyframes = mDef.getKeyframes(mp);
        const mDur = Number(mp.duration || mDef.defaults.duration || 1600);
        const mEase = mp.easing || mDef.defaults.easing || 'ease-in-out';
        const mDir = mDef.direction || 'normal';
        const motionSplit = mDef.supportsSplit && !mDef.isTextGradient
          ? (mp.split || mDef.defaults.split || 'whole')
          : 'whole';
        if (currentStyle.animate_motion_block && !mDef.isTextGradient) {
          panelEl.animate(mKeyframes, { duration: mDur, delay: 0, easing: mEase, direction: mDir, fill: 'backwards', iterations: Infinity });
        }
        if (mDef.isTextGradient) {
          textEl.animate(mKeyframes, { duration: mDur, delay: 0, easing: mEase, direction: mDir, fill: 'backwards', iterations: Infinity });
        } else {
          const units = buildSplitUnits(textEl, fullText, motionSplit);
          if (motionId === 'pulse-breathe') reservePulsePadding(mp);
          const stagger = motionSplit === 'whole' ? 0 : Number(mp.stagger || mDef.defaults.stagger || 70);
          units.forEach((unit, idx) => unit.inner.animate(mKeyframes, {
            duration: mDur, delay: idx * stagger, easing: mEase, direction: mDir, fill: 'backwards', iterations: Infinity
          }));
        }
        updateScrollAndMask();
        scheduleHide(Math.max(durationSec * 1000, mDur));
        return;
      }

      const useAudioDelays = Array.isArray(data.char_delays) && data.char_delays.length >= Array.from(fullText).length;

      // 1) Режим появи з каталогу Test (WAAPI), завжди поєднаний із рухом (якщо обрано)
      if (eDef) {
        cursorEl.classList.remove('active');
        const ap = Object.assign({}, eDef.defaults, currentStyle.anim_params || {});
        const entrySplit = eDef.supportsSplit ? (ap.split || eDef.defaults.split || 'whole') : 'whole';
        const motionSplit = (mDef && mDef.supportsSplit && !mDef.isTextGradient)
          ? (mp.split || mDef.defaults.split || 'whole')
          : 'whole';
        // Test applies gradient motions to one intact text node. Splitting it into
        // child spans prevents background-clip:text from painting the gradient.
        const gradientMotion = !!(mDef && mDef.isTextGradient);
        const effectiveEntrySplit = gradientMotion ? 'whole' : entrySplit;

        const domSplit = (effectiveEntrySplit === 'chars' || motionSplit === 'chars')
          ? 'chars'
          : ((effectiveEntrySplit === 'words' || motionSplit === 'words') ? 'words' : 'whole');

        const units = gradientMotion ? [] : buildSplitUnits(textEl, fullText, domSplit);
        if (motionId === 'pulse-breathe' && !gradientMotion) reservePulsePadding(mp);
        if (gradientMotion) textEl.textContent = fullText;
        const eKeyframes = eDef.getKeyframes(ap);
        const eDuration = Number(ap.duration || eDef.defaults.duration || 700);
        const eDelay = Number(ap.delay || 0);
        const eEasing = ap.easing || eDef.defaults.easing || 'ease-out';
        const eStagger = effectiveEntrySplit === 'whole' ? 0 : Number(ap.stagger || eDef.defaults.stagger || 45);
        const isBoxed = currentStyle.style === 'boxed' || (currentStyle.style === undefined && !!currentStyle.bg_enabled);
        const animateBlockWithEntry = !!currentStyle.animate_block && isBoxed;
        // These effects animate their word/letter units. Keep the block still
        // and reveal it with opacity so its transform cannot move the text twice.
        const blockKeyframes = (animId === 'mask-rise' || animId === 'wave-stagger')
          ? [{ opacity: 0 }, { opacity: 1 }]
          : eKeyframes;

        if (animateBlockWithEntry) {
          panelEl.animate(blockKeyframes, {
            duration: eDuration,
            delay: eDelay,
            easing: eEasing,
            fill: 'both'
          });
        }

        if (animId === 'mask-rise') {
          // Keep the clipping window stationary and move only its text child.
          const riseFrames = [
            { transform: `translateY(${Number(ap.distance || 105)}%)`, opacity: 0 },
            { transform: 'translateY(0)', opacity: 1 }
          ];
          if (effectiveEntrySplit === 'whole') {
            textEl.animate(riseFrames, { duration: eDuration, delay: eDelay, easing: eEasing, fill: 'both' });
          } else {
            units.forEach((unit, idx) => {
              const delay = eDelay + idx * eStagger;
              unit.outer.classList.add('mask-rise-window');
              unit.inner.animate(riseFrames, { duration: eDuration, delay, easing: eEasing, fill: 'both' });
            });
          }
        } else if (animId === 'wave-stagger' && effectiveEntrySplit !== 'whole') {
          // Keep each glyph's line box fixed and animate only its inner glyph.
          // The parent block clips any bounce outside its padded content area.
          const waveEasing = 'cubic-bezier(0.22, 1, 0.36, 1)';
          const waveFrames = eKeyframes.map((frame, idx, frames) => idx === frames.length - 1
            ? Object.assign({}, frame, { transform: 'none' })
            : frame);
          units.forEach((unit, idx) => {
            unit.inner.animate(waveFrames, { duration: eDuration, delay: eDelay + idx * eStagger, easing: waveEasing, fill: 'both' });
          });
        } else if (effectiveEntrySplit === 'whole') {
          textEl.animate(eKeyframes, { duration: eDuration, delay: eDelay, easing: eEasing, fill: 'both' });
        } else {
          units.forEach((u, idx) => {
            u.outer.animate(eKeyframes, { duration: eDuration, delay: eDelay + idx * eStagger, easing: eEasing, fill: 'both' });
          });
        }
        const entryTotalMs = eDelay + eDuration + (effectiveEntrySplit === 'whole' ? 0 : Math.max(0, units.length - 1) * eStagger);

        if (mDef && mp) {
          const mDelay = mTrigger === 'after' ? eDelay + eDuration : 0;
          const mKeyframes = mDef.getKeyframes(mp);
          const mDur = Number(mp.duration || mDef.defaults.duration || 1600);
          const mEase = mp.easing || mDef.defaults.easing || 'ease-in-out';
          const mDir = mDef.direction || 'normal';
          const mStagger = motionSplit === 'whole' ? 0 : Number(mp.stagger || mDef.defaults.stagger || 70);

          if (currentStyle.animate_motion_block && !mDef.isTextGradient) {
            panelEl.animate(mKeyframes, {
              duration: mDur,
              delay: mDelay,
              easing: mEase,
              direction: mDir,
              fill: 'backwards',
              iterations: Infinity
            });
          }

          if (mDef.isTextGradient) {
            textEl.animate(mKeyframes, { duration: mDur, delay: mDelay, easing: mEase, direction: mDir, fill: 'backwards', iterations: Infinity });
          } else if (motionSplit === 'whole') {
            if (entrySplit === 'whole' && units[0]) {
              units[0].inner.animate(mKeyframes, { duration: mDur, delay: mDelay, easing: mEase, direction: mDir, fill: 'backwards', iterations: Infinity });
            } else {
              textEl.animate(mKeyframes, { duration: mDur, delay: mDelay, easing: mEase, direction: mDir, fill: 'backwards', iterations: Infinity });
            }
          } else {
            units.forEach((u, idx) => {
              u.inner.animate(mKeyframes, { duration: mDur, delay: mDelay + idx * mStagger, easing: mEase, direction: mDir, fill: 'backwards', iterations: Infinity });
            });
          }
        }

        updateScrollAndMask();
        scheduleHide(Math.max(entryTotalMs + 200, durationSec * 1000));
        return;
      }

      // 2) Стандартний режим: Друкарська машинка (по буквах) + опційна анімація руху
      if (currentStyle.show_cursor) cursorEl.classList.add('active');
      else cursorEl.classList.remove('active');

      const chars = Array.from(fullText);
      const stepMs = (durationSec * 820) / Math.max(1, chars.length);
      let delay = 0;
      const charDelays = chars.map((_, idx) => useAudioDelays ? Number(data.char_delays[idx] || 0) : (delay += stepMs));
      const maxTypeDelay = charDelays.length ? Math.max(...charDelays) : 0;

      const motionSplit = (mDef && mDef.supportsSplit && !mDef.isTextGradient)
        ? (mp.split || mDef.defaults.split || 'whole')
        : 'whole';
      const splitMotionUnits = mDef && motionSplit !== 'whole';

      if (mDef && mp) {
        const mDelay = mTrigger === 'after' ? maxTypeDelay : 0;
        if (currentStyle.animate_motion_block && !mDef.isTextGradient) {
          panelEl.animate(mDef.getKeyframes(mp), {
            duration: Number(mp.duration || mDef.defaults.duration || 1600),
            delay: mDelay,
            easing: mp.easing || mDef.defaults.easing || 'ease-in-out',
            direction: mDef.direction || 'normal',
            fill: 'backwards',
            iterations: Infinity
          });
        }
        if (!splitMotionUnits) {
          textEl.animate(mDef.getKeyframes(mp), {
            duration: Number(mp.duration || mDef.defaults.duration || 1600),
            delay: mDelay,
            easing: mp.easing || mDef.defaults.easing || 'ease-in-out',
            direction: mDef.direction || 'normal',
            fill: 'backwards',
            iterations: Infinity
          });
        }
      }

      let curWordSpan = null;
      let motionUnitIdx = 0;
      const mKeyframes = mDef ? mDef.getKeyframes(mp) : null;
      const mDur = mDef ? Number(mp.duration || mDef.defaults.duration || 1600) : 1600;
      const mEase = mDef ? (mp.easing || mDef.defaults.easing || 'ease-in-out') : 'ease-in-out';
      const mDir = mDef ? (mDef.direction || 'normal') : 'normal';
      const mStagger = mDef ? Number(mp.stagger || mDef.defaults.stagger || 70) : 0;

      chars.forEach((ch, idx) => {
        const chDelay = charDelays[idx];
        const t = setTimeout(() => {
          if (splitMotionUnits) {
            if (/\s/.test(ch)) {
              curWordSpan = null;
              textEl.appendChild(document.createTextNode(ch));
            } else if (motionSplit === 'words') {
              if (!curWordSpan) {
                curWordSpan = document.createElement('span');
                curWordSpan.className = 'anim-inner';
                const uIdx = motionUnitIdx++;
                const baseDelay = mTrigger === 'after' ? Math.max(0, maxTypeDelay - chDelay) : 0;
                curWordSpan.animate(mKeyframes, {
                  duration: mDur,
                  delay: baseDelay + uIdx * mStagger,
                  easing: mEase,
                  direction: mDir,
                  fill: 'backwards',
                  iterations: Infinity
                });
                textEl.appendChild(curWordSpan);
              }
              curWordSpan.textContent += ch;
            } else {
              const sp = document.createElement('span');
              sp.className = 'anim-inner';
              sp.textContent = ch;
              const uIdx = motionUnitIdx++;
              const baseDelay = mTrigger === 'after' ? Math.max(0, maxTypeDelay - chDelay) : 0;
              sp.animate(mKeyframes, {
                duration: mDur,
                delay: baseDelay + uIdx * mStagger,
                easing: mEase,
                direction: mDir,
                fill: 'backwards',
                iterations: Infinity
              });
              textEl.appendChild(sp);
            }
          } else {
            textEl.textContent += ch;
          }
          updateScrollAndMask();
        }, chDelay);
        timers.push(t);
      });
      scheduleHide(Math.max(maxTypeDelay + 180, durationSec * 1000));
    }

    const isPreviewMode = new URLSearchParams(location.search).get('preview') === '1';
    const previewStatusEl = document.getElementById('preview-status');
    const previewTestBtn = document.getElementById('preview-test-btn');
    const previewSaveBtn = document.getElementById('preview-save-btn');
    const previewSaveStatus = document.getElementById('preview-save-status');
    let previewInitialShown = false;
    let wsConnected = false;
    let previewSocket = null;

    function t(key, fallback) {
      return (I18N && I18N[key]) ? I18N[key] : fallback;
    }
    function applyI18n(dict) {
      if (dict && typeof dict === 'object') Object.assign(I18N, dict);
      if (I18N.lang) document.documentElement.lang = I18N.lang;
      document.querySelectorAll('[data-i18n]').forEach((node) => {
        node.textContent = t(node.dataset.i18n, node.textContent);
      });
      document.querySelectorAll('[data-i18n-aria-label]').forEach((node) => {
        node.setAttribute('aria-label', t(node.dataset.i18nAriaLabel, node.getAttribute('aria-label') || ''));
      });
      if (previewTestBtn) previewTestBtn.textContent = t('test_btn', '✨ Test subtitles');
      if (previewSaveBtn) previewSaveBtn.textContent = t('obs.preview.save', 'Save to app');
      if (previewStatusEl) {
        previewStatusEl.textContent = wsConnected
          ? t('connected', '🟢 Connected to SilentVoice (OBS preview)')
          : t('connecting', '🟡 Connecting to SilentVoice...');
      }
    }
    function sampleSpeech() {
      const text = t('sample_text', 'Hello viewers! Testing OBS subtitles ✨');
      const words = text.split(/\s+/).filter(Boolean).length;
      return { text: text, duration: Math.min(12, Math.max(3.5, words * 0.32)) };
    }

    const previewParamCache = { entry: {}, motion: {} };
    const previewSplitCache = {};
    let previewControlsReady = false;
    function previewSpec(kind, id) {
      const list = kind === 'entry' ? ANIMATION_META.entry : ANIMATION_META.motion;
      return list.find((item) => item.id === id) || list[0];
    }
    function previewParams(kind, spec) {
      const styleKey = kind === 'entry' ? 'anim_params' : 'motion_params';
      const idKey = kind === 'entry' ? 'anim_style' : 'motion_effect';
      if (!previewParamCache[kind][spec.id]) {
        const activeId = currentStyle[idKey] || '';
        const activeParams = currentStyle[styleKey] || {};
        previewParamCache[kind][spec.id] = Object.assign({}, spec.defaults || {}, activeId === spec.id ? activeParams : {});
        if (kind === 'entry' && spec.supports_split) {
          const params = previewParamCache[kind][spec.id];
          const modes = {};
          ['whole', 'words', 'chars'].forEach((mode) => {
            const defaults = Object.assign({}, spec.defaults || {}, (spec.split_defaults || {})[mode] || {});
            modes[mode] = { duration: defaults.duration || 700, stagger: defaults.stagger || 45 };
          });
          modes[params.split || 'whole'] = { duration: params.duration, stagger: params.stagger };
          previewSplitCache[spec.id] = modes;
        }
      }
      return previewParamCache[kind][spec.id];
    }
    function addPreviewSelect(parent, label, key, value, options, onChange) {
      const wrap = document.createElement('label');
      wrap.className = 'preview-field';
      wrap.textContent = label;
      const select = document.createElement('select');
      select.className = 'preview-select';
      select.dataset.key = key;
      options.forEach((option) => {
        const node = document.createElement('option');
        node.value = option.value;
        node.textContent = option.label;
        select.appendChild(node);
      });
      select.value = value;
      select.addEventListener('change', () => onChange(select.value));
      wrap.appendChild(select);
      parent.appendChild(wrap);
    }
    function addPreviewRange(parent, label, key, value, min, max, step, unit, onChange) {
      const wrap = document.createElement('label');
      wrap.className = 'preview-field';
      const head = document.createElement('span');
      head.className = 'preview-field-head';
      const name = document.createElement('span');
      name.textContent = label;
      const output = document.createElement('span');
      output.className = 'preview-value';
      const range = document.createElement('input');
      range.type = 'range';
      range.className = 'preview-range';
      range.dataset.key = key;
      range.min = min;
      range.max = max;
      range.step = step;
      range.value = value;
      const showValue = () => { output.textContent = `${range.value}${unit || ''}`; };
      showValue();
      range.addEventListener('input', showValue);
      range.addEventListener('change', () => onChange(Number(range.value)));
      head.append(name, output);
      wrap.append(head, range);
      parent.appendChild(wrap);
    }
    function previewControlChanged(kind, key, value) {
      const spec = previewSpec(kind, kind === 'entry' ? currentStyle.anim_style : currentStyle.motion_effect);
      const params = previewParams(kind, spec);
      if (kind === 'entry' && key === 'split' && spec.supports_split) {
        const modes = previewSplitCache[spec.id] || (previewSplitCache[spec.id] = {});
        modes[params.split || 'whole'] = { duration: params.duration, stagger: params.stagger };
        params.split = value;
        const defaults = Object.assign({}, spec.defaults || {}, (spec.split_defaults || {})[value] || {});
        const next = modes[value] || { duration: defaults.duration || 700, stagger: defaults.stagger || 45 };
        params.duration = next.duration;
        params.stagger = next.stagger;
        modes[value] = next;
      } else {
        params[key] = value;
        if (kind === 'entry' && spec.supports_split) {
          const modes = previewSplitCache[spec.id] || (previewSplitCache[spec.id] = {});
          modes[params.split || 'whole'] = { duration: params.duration, stagger: params.stagger };
        }
      }
      if (kind === 'entry') {
        currentStyle.anim_params = params;
        applyStyle(currentStyle);
      } else {
        currentStyle.motion_params = params;
        applyStyle(currentStyle);
      }
      renderPreviewParams(kind);
      playSpeech(sampleSpeech());
    }
    function resetPreviewAnimationSettings() {
      const entrySpec = previewSpec('entry', currentStyle.anim_style || 'typewriter');
      const entryParams = Object.assign({}, entrySpec?.defaults || {});
      if (entrySpec?.supports_split) {
        const modes = {};
        ['whole', 'words', 'chars'].forEach((mode) => {
          const defaults = Object.assign({}, entrySpec.defaults || {}, (entrySpec.split_defaults || {})[mode] || {});
          modes[mode] = { duration: defaults.duration || 700, stagger: defaults.stagger || 45 };
        });
        const split = entryParams.split || 'whole';
        entryParams.duration = modes[split].duration;
        entryParams.stagger = modes[split].stagger;
        previewSplitCache[entrySpec.id] = modes;
      }
      previewParamCache.entry[entrySpec.id] = entryParams;
      currentStyle.anim_params = entryParams;

      const motionSpec = previewSpec('motion', currentStyle.motion_effect || 'none');
      const motionParams = Object.assign({}, motionSpec?.defaults || {});
      if (motionSpec) previewParamCache.motion[motionSpec.id] = motionParams;
      currentStyle.motion_params = motionParams;

      applyStyle(currentStyle);
      renderPreviewParams('entry');
      renderPreviewParams('motion');
      playSpeech(sampleSpeech());
    }
    function renderPreviewParams(kind) {
      const host = document.getElementById(kind === 'entry' ? 'preview-entry-controls' : 'preview-motion-controls');
      if (!host) return;
      host.replaceChildren();
      const spec = previewSpec(kind, kind === 'entry' ? currentStyle.anim_style : currentStyle.motion_effect);
      if (!spec || (kind === 'motion' && spec.id === 'none') || (kind === 'entry' && spec.id === 'typewriter')) {
        const empty = document.createElement('div');
        empty.className = 'preview-empty';
        empty.textContent = kind === 'motion'
          ? t('obs.preview.no_motion_params', 'No additional settings for this mode.')
          : t('obs.preview.typewriter_hint', 'Letters appear in time with the spoken voice—like lip-sync, but for text.');
        host.appendChild(empty);
        return;
      }
      const params = previewParams(kind, spec);
      const setOptions = (label, key, value, options) => addPreviewSelect(host, label, key, value, options, (next) => previewControlChanged(kind, key, next));
      const range = (label, key, min, max, step, unit) => addPreviewRange(host, label, key, params[key], min, max, step, unit, (next) => previewControlChanged(kind, key, next));
      range(
        t(kind === 'entry' ? 'obs.ctrl.duration' : 'obs.ctrl.motion_duration', kind === 'entry' ? 'Entry duration' : 'Cycle duration'),
        'duration', kind === 'entry' ? 150 : 600, kind === 'entry' ? 2500 : 6000,
        kind === 'entry' ? 50 : 100, ` ${t('obs.preview.ms', 'ms')}`
      );
      if (spec.supports_split) {
        setOptions(t('obs.ctrl.split', 'Text split'), 'split', params.split || 'whole', [
          { value: 'whole', label: t('obs.opt.split_whole', 'Whole text') },
          { value: 'words', label: t('obs.opt.split_words', 'By words') },
          { value: 'chars', label: t('obs.opt.split_chars', 'By letters') }
        ]);
        if (kind === 'entry' && (params.split || 'whole') !== 'whole') {
          range(t('obs.ctrl.stagger', 'Stagger'), 'stagger', 10, 200, 5, ` ${t('obs.preview.ms', 'ms')}`);
        }
      }
      if (kind === 'motion') {
        setOptions(t('obs.ctrl.motion_trigger', 'Start relative to entry'), 'trigger', params.trigger || 'after', [
          { value: 'after', label: t('obs.opt.trigger_after', 'After text appears') },
          { value: 'simultaneous', label: t('obs.opt.trigger_sim', 'Simultaneously with entry') },
          { value: 'only', label: t('obs.opt.trigger_only', 'Motion only (no entrance)') }
        ]);
      }
      (spec.controls || []).forEach((control) => {
        if (control.type === 'range') {
          range(t(control.i18n, control.label), control.key, control.min, control.max, control.step, control.unit);
        } else if (control.type === 'select') {
          const options = (control.options || []).map((option) => ({
            value: option.value,
            label: t(option.i18n, option.label)
          }));
          setOptions(t(control.i18n, control.label), control.key, params[control.key], options);
        }
      });
    }
    function syncPreviewControls() {
      if (!isPreviewMode) return;
      const entrySelect = document.getElementById('preview-entry-select');
      const motionSelect = document.getElementById('preview-motion-select');
      const entryBlockCheck = document.getElementById('preview-animate-entry-block');
      const motionBlockCheck = document.getElementById('preview-animate-motion-block');
      const animationResetBtn = document.getElementById('preview-animation-reset-btn');
      if (!entrySelect || !motionSelect) return;
      if (!previewControlsReady) {
        ANIMATION_META.entry.forEach((item) => {
          const option = document.createElement('option'); option.value = item.id; option.textContent = t(item.i18n_key, item.name); entrySelect.appendChild(option);
        });
        ANIMATION_META.motion.forEach((item) => {
          const option = document.createElement('option'); option.value = item.id; option.textContent = t(item.i18n_key, item.name); motionSelect.appendChild(option);
        });
        entrySelect.addEventListener('change', () => {
          const nextAnimId = entrySelect.value;
          const nextParams = previewParams('entry', previewSpec('entry', nextAnimId));
          currentStyle.anim_style = nextAnimId;
          currentStyle.anim_params = nextParams;
          applyStyle(currentStyle);
          renderPreviewParams('entry'); playSpeech(sampleSpeech());
        });
        motionSelect.addEventListener('change', () => {
          const nextMotionId = motionSelect.value;
          const nextParams = previewParams('motion', previewSpec('motion', nextMotionId));
          currentStyle.motion_effect = nextMotionId;
          currentStyle.motion_params = nextParams;
          applyStyle(currentStyle);
          renderPreviewParams('motion'); playSpeech(sampleSpeech());
        });
        document.querySelectorAll('[data-preview-style]').forEach((button) => {
          button.addEventListener('click', () => {
            currentStyle.style = button.dataset.previewStyle;
            currentStyle.bg_enabled = currentStyle.style === 'boxed';
            applyStyle(currentStyle);
            playSpeech(sampleSpeech());
          });
        });
        entryBlockCheck.addEventListener('change', () => {
          currentStyle.animate_block = entryBlockCheck.checked;
          playSpeech(sampleSpeech());
        });
        motionBlockCheck.addEventListener('change', () => {
          currentStyle.animate_motion_block = motionBlockCheck.checked;
          playSpeech(sampleSpeech());
        });
        animationResetBtn.addEventListener('click', resetPreviewAnimationSettings);
        previewControlsReady = true;
      }
      entrySelect.value = currentStyle.anim_style || 'typewriter';
      motionSelect.value = currentStyle.motion_effect || 'none';
      const activeStyle = currentStyle.style || (currentStyle.bg_enabled ? 'boxed' : 'text_only');
      document.querySelectorAll('[data-preview-style]').forEach((button) => {
        button.classList.toggle('active', button.dataset.previewStyle === activeStyle);
      });
      entryBlockCheck.checked = !!currentStyle.animate_block;
      motionBlockCheck.checked = !!currentStyle.animate_motion_block;
      renderPreviewParams('entry');
      renderPreviewParams('motion');
    }
    applyI18n();

    if (isPreviewMode) {
      document.body.classList.add('preview-mode');
      previewTestBtn.addEventListener('click', () => {
        playSpeech(sampleSpeech());
      });
      previewSaveBtn.addEventListener('click', () => {
        if (!previewSocket || previewSocket.readyState !== WebSocket.OPEN) {
          previewSaveStatus.textContent = t('obs.preview.no_connection', 'Not connected to the app');
          return;
        }
        if (previewSaveTimer) clearTimeout(previewSaveTimer);
        previewSaveBtn.disabled = true;
        previewSaveStatus.textContent = t('obs.preview.saving', 'Saving…');
        previewSocket.send(JSON.stringify({
          type: 'save_preview_settings',
          settings: {
            style: currentStyle.style || (currentStyle.bg_enabled ? 'boxed' : 'text_only'),
            anim_style: currentStyle.anim_style || 'typewriter',
            motion_effect: currentStyle.motion_effect || 'none',
            animate_block: !!currentStyle.animate_block,
            animate_motion_block: !!currentStyle.animate_motion_block,
            anim_params: currentStyle.anim_params || {},
            motion_params: currentStyle.motion_params || {}
          }
        }));
        previewSaveTimer = setTimeout(() => {
          previewSaveBtn.disabled = false;
          previewSaveStatus.textContent = t('obs.preview.save_timeout', 'No save confirmation received');
          previewSaveTimer = null;
        }, 10000);
      });
    }

    function connectWS() {
      const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
      const wsPath = isPreviewMode ? '/ws?preview=1' : '/ws';
      const ws = new WebSocket(proto + location.host + wsPath);
      if (isPreviewMode) previewSocket = ws;
      ws.onopen = () => {
        wsConnected = true;
        if (isPreviewMode && previewStatusEl) {
          previewStatusEl.textContent = t('connected', '🟢 Connected to SilentVoice (OBS preview)');
        }
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'reload') {
            window.location.reload();
            return;
          }
          if (msg.type === 'style') {
            if (msg.i18n) applyI18n(msg.i18n);
            applyStyle(msg.style);
            if (isPreviewMode && !previewInitialShown) {
              previewInitialShown = true;
              playSpeech(sampleSpeech());
            }
          } else if (msg.type === 'save_result') {
            if (previewSaveTimer) clearTimeout(previewSaveTimer);
            previewSaveTimer = null;
            previewSaveBtn.disabled = false;
            previewSaveStatus.textContent = msg.ok
              ? t('obs.preview.saved', 'Saved to app')
              : (msg.error || t('obs.preview.save_failed', 'Could not save settings'));
          } else if (msg.type === 'speech') {
            if (msg.style) applyStyle(msg.style);
            playSpeech(msg);
          } else if (msg.type === 'live_typing') {
            if (msg.style) applyStyle(msg.style);
            showLiveTyping(msg.text);
          } else if (msg.type === 'clear') {
            hideSubtitles();
          } else if (msg.type === 'clear_live_typing') {
            if (boxEl.classList.contains('live-typing')) {
              hideSubtitles();
            }
          }
        } catch (e) {}
      };
      ws.onclose = () => {
        wsConnected = false;
        if (isPreviewMode && previewSaveBtn.disabled) {
          if (previewSaveTimer) clearTimeout(previewSaveTimer);
          previewSaveTimer = null;
          previewSaveBtn.disabled = false;
          previewSaveStatus.textContent = t('waiting', '🟡 Waiting for connection to SilentVoice...');
        }
        if (isPreviewMode && previewStatusEl) {
          previewStatusEl.textContent = t('waiting', '🟡 Waiting for connection to SilentVoice...');
        }
        setTimeout(connectWS, 1500);
      };
      ws.onerror = () => ws.close();
    }
    connectWS();
  </script>
</body>
</html>
"""


DEFAULT_CUSTOM_OBS_CSS = """/* ============================================================================
 * SilentVoice Custom OBS Overlay CSS (style.css)
 * ============================================================================
 * Human guide:
 *   Edit colors, font, box style, and animations here, then click
 *   "🔄 Refresh OBS" in SilentVoice to see your changes in OBS right away.
 *
 * Guide for AI assistants (Claude / ChatGPT / etc.):
 *   - Keep `html, body { background: transparent; }` for OBS Browser Source.
 *   - `style.css` is injected at `{{CSS}}` inside `index.html`.
 *   - Visibility state classes toggled by `index.html`:
 *       #subtitle-box.visible      -> shown during speech / live typing
 *       #subtitle-box.live-typing  -> active while typing before Enter
 *       .sv-word / .sv-word.show   -> word-by-word audio-synced reveal
 *   - Keep timing / WebSocket sync logic inside `index.html` (`msg.char_delays`),
 *     never bypass it with fixed CSS delays that start before audio plays.
 * ============================================================================ */
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  width: 100%;
  height: 100%;
  background: transparent;
  overflow: hidden;
  font-family: 'Montserrat', 'Segoe UI', sans-serif;
}
.sv-wrap {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding: 16px 28px;
}
#subtitle-box {
  max-width: 92%;
  padding: 12px 22px;
  border-radius: 14px;
  background: rgba(19, 21, 31, 0.78);
  border: 2px solid #7C5CFF;
  color: #FFFFFF;
  font-size: 32px;
  font-weight: 700;
  line-height: 1.35;
  text-align: center;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.85);
  opacity: 0;
  transform: translateY(10px) scale(0.98);
  transition: opacity 0.22s ease, transform 0.22s ease;
}
#subtitle-box.visible {
  opacity: 1;
  transform: translateY(0) scale(1);
}
#subtitle-box.live-typing {
  border-color: #38BDF8;
}
.sv-word {
  display: inline-block;
  opacity: 0;
  transform: translateY(6px);
  transition: opacity 0.16s ease, transform 0.16s ease;
}
.sv-word.show {
  opacity: 1;
  transform: translateY(0);
}
"""

DEFAULT_CUSTOM_OBS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SilentVoice Custom OBS Template</title>
  <!--
    =========================================================================
    SILENTVOICE OBS CUSTOM TEMPLATE — HUMAN & AI GUIDE
    =========================================================================
    1. FILES:
       - `index.html` — layout, `/ws` WebSocket client, and animation logic.
       - `style.css`  — stylesheet (automatically injected at `{{CSS}}`).
       After saving your edits, click "🔄 Refresh OBS" in SilentVoice settings.

    2. WEBSOCKET CONTRACT (/ws) — MESSAGES SENT BY SILENTVOICE:
       - { type: "speech", text: string, duration: number (sec),
           char_delays?: number[] (ms per Unicode codepoint!), style: object }
       - { type: "live_typing", text: string, style: object }
       - { type: "clear" } / { type: "clear_live_typing" }
       - { type: "style", style: object }
       - { type: "reload" } -> MUST call window.location.reload()

    3. CRITICAL AUDIO-SYNC RULES (FOR AI ASSISTANTS):
       - SilentVoice sends `speech` BEFORE opening the Windows audio device
         and before any leading silence inside the synthesized TTS audio.
       - `msg.char_delays` already includes the playback startup offset (~85 ms)
         and exact audio-envelope timestamps (in ms) for each character in
         `Array.from(msg.text)` matching real speech onset.
       - NEVER reveal the final subtitle at `0 ms` and NEVER use a fixed cap
         like `Math.min(90, ...)` per word — that makes text appear earlier
         than the spoken voice and drift ahead!
       - ALWAYS synchronize the container reveal with `firstDelay = char_delays[0]`
         (or ~85 ms when `char_delays` is absent), and reveal each word/letter
         using `msg.char_delays[charIdx]`.
    =========================================================================
  -->
  <style>{{CSS}}</style>
</head>
<body>
  <div class="sv-wrap">
    <div id="subtitle-box" aria-live="polite"></div>
  </div>
  <script>
    const box = document.getElementById('subtitle-box');
    let startTimer = null;
    let hideTimer = null;
    let wordTimers = [];

    function clearAllTimers() {
      if (startTimer) clearTimeout(startTimer);
      if (hideTimer) clearTimeout(hideTimer);
      startTimer = null;
      hideTimer = null;
      for (const t of wordTimers) clearTimeout(t);
      wordTimers = [];
    }

    function hideBox() {
      clearAllTimers();
      box.classList.remove('visible', 'live-typing');
    }

    function showSpeech(msg) {
      clearAllTimers();
      box.classList.remove('visible', 'live-typing');

      const text = String(msg.text || '');
      const chars = Array.from(text);
      const rawDelays = Array.isArray(msg.char_delays) && msg.char_delays.length >= chars.length
        ? msg.char_delays
        : null;
      const totalMs = Math.max(400, Number(msg.duration || 2.0) * 1000);
      const defaultStartMs = 85;

      // Split text into words and map each word to its first character index in char_delays
      const frag = document.createDocumentFragment();
      const wordEntries = [];
      let charCursor = 0;
      for (const part of text.split(/(\\s+)/)) {
        if (!part) continue;
        const partLen = Array.from(part).length;
        if (/^\\s+$/.test(part)) {
          frag.appendChild(document.createTextNode(part));
        } else {
          const sp = document.createElement('span');
          sp.className = 'sv-word';
          sp.textContent = part;
          frag.appendChild(sp);
          const delayMs = rawDelays
            ? Math.max(0, Number(rawDelays[charCursor] || 0))
            : Math.round(defaultStartMs + (charCursor / Math.max(1, chars.length)) * (totalMs * 0.82));
          wordEntries.push({ el: sp, delayMs });
        }
        charCursor += partLen;
      }

      box.replaceChildren(frag);
      void box.offsetWidth;

      const firstDelay = wordEntries.length ? wordEntries[0].delayMs : defaultStartMs;
      startTimer = setTimeout(() => {
        box.classList.add('visible');
      }, firstDelay);

      for (const entry of wordEntries) {
        wordTimers.push(setTimeout(() => entry.el.classList.add('show'), entry.delayMs));
      }

      const maxWordDelay = wordEntries.length ? wordEntries[wordEntries.length - 1].delayMs : firstDelay;
      const endMs = Math.max(totalMs + defaultStartMs, maxWordDelay + 150);
      const fadeSec = Number((msg.style && msg.style.fade_sec) || 3.5);
      hideTimer = setTimeout(hideBox, endMs + fadeSec * 1000);
    }

    function showLiveTyping(text) {
      clearAllTimers();
      const clean = String(text || '');
      if (!clean) {
        hideBox();
        return;
      }
      box.textContent = clean;
      box.classList.add('visible', 'live-typing');
      hideTimer = setTimeout(hideBox, 12000);
    }

    function connectWS() {
      const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
      const ws = new WebSocket(proto + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'reload') {
            window.location.reload();
          } else if (msg.type === 'speech') {
            showSpeech(msg);
          } else if (msg.type === 'live_typing') {
            showLiveTyping(msg.text);
          } else if (msg.type === 'clear') {
            hideBox();
          } else if (msg.type === 'clear_live_typing' && box.classList.contains('live-typing')) {
            hideBox();
          }
        } catch (e) {}
      };
      ws.onclose = () => setTimeout(connectWS, 1500);
      ws.onerror = () => ws.close();
    }
    connectWS();
  </script>
</body>
</html>
"""

BLAITE_CUSTOM_TEST_NAME = "Blaite_castom_test"

BLAITE_CUSTOM_TEST_CSS = r"""/* ============================================================================
 * SilentVoice Custom OBS Overlay CSS (style.css) — "Futuristic Signal"
 * ============================================================================
 * Human guide:
 *   Main settings (colors, size, font, opacity) are in :root below.
 *   After editing, click "🔄 Refresh OBS" in SilentVoice.
 *
 * Guide for AI assistants (Claude / ChatGPT / etc.):
 *   - Keep `html, body { background: transparent; }` for OBS Browser Source.
 *   - `style.css` is injected at `{{CSS}}` inside `index.html`.
 *   - State classes are toggled on `#sv-panel` (NOT on #subtitle-box):
 *       #sv-panel.visible      -> shown during speech / live typing
 *       #sv-panel.speaking     -> audio is playing (equalizer animates)
 *       #sv-panel.live-typing  -> active while typing before Enter
 *   - `#subtitle-box.show` runs the whole-text materialize animation, timed
 *     by `index.html` to the first spoken character (`msg.char_delays`).
 *   - `#subtitle-box` is rewritten by JS: never put decoration inside it.
 *   - Never bypass the sync logic in `index.html` with fixed CSS delays.
 * ============================================================================ */
:root {
  --violet: #7C5CFF;
  --cyan: #38BDF8;
  --base: 6, 7, 13;          /* dark base RGB */
  --base-alpha: 0.90;        /* panel opacity: lower = more of the game shows through */
  --text: #F2F5FF;
  --font-text: 'Rajdhani', 'Montserrat', 'Segoe UI', sans-serif;
  --font-hud: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;
  --size: 30px;              /* subtitle text size */
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  width: 100%;
  height: 100%;
  background: transparent;
  overflow: hidden;
  font-family: var(--font-text);
}

/* Position: bottom center, max ~80% width */
.sv-wrap {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding: 0 28px 22px;
}

/* ---------- Panel ---------- */
#sv-panel {
  position: relative;
  max-width: 80%;
  min-width: 340px;
  opacity: 0;
  transform: translateY(12px);
  transition: opacity 0.3s ease, transform 0.3s ease;
  filter: drop-shadow(0 0 10px rgba(124, 92, 255, 0.38))
          drop-shadow(0 0 2px rgba(56, 189, 248, 0.45));
}
#sv-panel.visible {
  opacity: 1;
  transform: translateY(0);
}

/* Outer layer: thin gradient outline (chamfered, asymmetric corners) */
.sv-frame {
  padding: 1.5px;
  background: linear-gradient(115deg,
    var(--violet) 0%,
    var(--cyan) 42%,
    rgba(124, 92, 255, 0.3) 68%,
    var(--cyan) 100%);
  clip-path: polygon(
    26px 0, calc(100% - 8px) 0, 100% 8px,
    100% calc(100% - 26px), calc(100% - 26px) 100%,
    8px 100%, 0 calc(100% - 8px), 0 26px);
}

/* Inner layer: near-black base */
.sv-inner {
  position: relative;
  overflow: hidden;
  padding: 10px 30px 14px 32px;
  background:
    repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.022) 0 1px, transparent 1px 3px),
    linear-gradient(135deg, rgba(var(--base), var(--base-alpha)), rgba(2, 3, 8, calc(var(--base-alpha) + 0.04)));
  clip-path: polygon(
    25px 0, calc(100% - 7px) 0, 100% 7px,
    100% calc(100% - 25px), calc(100% - 25px) 100%,
    7px 100%, 0 calc(100% - 7px), 0 25px);
}
/* Left light strip */
.sv-inner::before {
  content: "";
  position: absolute;
  left: 0; top: 30px; bottom: 14px;
  width: 3px;
  background: linear-gradient(to bottom, var(--violet), var(--cyan));
  box-shadow: 0 0 8px var(--violet);
}
/* Bottom-right accent */
.sv-inner::after {
  content: "";
  position: absolute;
  right: 32px; bottom: 0;
  width: 64px; height: 2px;
  background: linear-gradient(90deg, var(--violet), var(--cyan));
  box-shadow: 0 0 8px var(--cyan);
}

/* One-shot scan sweep on appear */
.sv-scan {
  position: absolute;
  top: 0; left: -30%;
  width: 30%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.2), transparent);
  transform: skewX(-20deg);
  opacity: 0;
  pointer-events: none;
}
#sv-panel.visible .sv-scan { animation: sv-sweep 0.9s ease-out 1; }
@keyframes sv-sweep {
  0%   { transform: translateX(0) skewX(-20deg); opacity: 1; }
  100% { transform: translateX(480%) skewX(-20deg); opacity: 0; }
}

/* ---------- Header: equalizer + VOICE TRANSMISSION ---------- */
.sv-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 5px;
  font-family: var(--font-hud);
  font-size: 11px;
  letter-spacing: 0.3em;
  color: var(--cyan);
  text-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
  white-space: nowrap;
}
.sv-rule {
  flex: 1;
  min-width: 30px;
  height: 1px;
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.55), rgba(124, 92, 255, 0));
}
.sv-ticks {
  width: 4px; height: 4px;
  background: var(--violet);
  box-shadow: 8px 0 0 var(--cyan), 16px 0 0 rgba(124, 92, 255, 0.4);
  margin-right: 16px;
}

/* Equalizer: low bars when idle, animated while speaking */
.sv-eq {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 16px;
}
.sv-eq i {
  display: block;
  width: 3px;
  height: 100%;
  background: linear-gradient(to top, var(--violet), var(--cyan));
  box-shadow: 0 0 6px rgba(56, 189, 248, 0.6);
  transform-origin: bottom;
  transform: scaleY(0.2);
}
#sv-panel.speaking .sv-eq i,
#sv-panel.live-typing .sv-eq i {
  animation: sv-eq 0.8s ease-in-out infinite;
}
.sv-eq i:nth-child(1) { animation-duration: 0.70s !important; animation-delay: -0.10s !important; }
.sv-eq i:nth-child(2) { animation-duration: 0.95s !important; animation-delay: -0.45s !important; }
.sv-eq i:nth-child(3) { animation-duration: 0.60s !important; animation-delay: -0.25s !important; }
.sv-eq i:nth-child(4) { animation-duration: 1.05s !important; animation-delay: -0.70s !important; }
.sv-eq i:nth-child(5) { animation-duration: 0.80s !important; animation-delay: -0.35s !important; }
@keyframes sv-eq {
  0%, 100% { transform: scaleY(0.2); }
  30%      { transform: scaleY(1); }
  55%      { transform: scaleY(0.45); }
  75%      { transform: scaleY(0.85); }
}

/* ---------- Subtitle text ---------- */
#subtitle-box {
  font-size: var(--size);
  font-weight: 600;
  line-height: 1.3;
  letter-spacing: 0.01em;
  text-align: left;
  color: var(--text);
  text-shadow: 0 0 14px rgba(124, 92, 255, 0.5), 0 2px 6px rgba(0, 0, 0, 0.8);
}

/* Whole-text digital materialization (all text at once) */
#subtitle-box { opacity: 0; }
#subtitle-box.show {
  animation: sv-materialize 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}
@keyframes sv-materialize {
  0% {
    opacity: 0;
    filter: blur(10px);
    transform: translateY(5px) skewX(-8deg);
    color: var(--cyan);
    text-shadow: 0 0 20px var(--cyan);
  }
  30% { opacity: 1;    filter: blur(3px); color: #7DD3FC; }
  42% { opacity: 0.45; filter: blur(1px); }
  58% { opacity: 1;    filter: blur(0);   color: #C4B5FD; }
  100% {
    opacity: 1;
    filter: none;
    transform: none;
    color: var(--text);
    text-shadow: 0 0 14px rgba(124, 92, 255, 0.5), 0 2px 6px rgba(0, 0, 0, 0.8);
  }
}

/* ---------- live-typing mode ---------- */
#sv-panel.live-typing .sv-frame {
  background: linear-gradient(115deg, var(--cyan), #7DD3FC 50%, var(--violet));
}
#sv-panel.live-typing #subtitle-box { color: #D5F0FF; animation: none; opacity: 1; }
#sv-panel.live-typing #subtitle-box::after {
  content: "";
  display: inline-block;
  width: 0.5em; height: 3px;
  margin-left: 6px;
  background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan);
  vertical-align: baseline;
  animation: sv-caret 0.8s steps(1) infinite;
}
@keyframes sv-caret { 50% { opacity: 0; } }

@media (prefers-reduced-motion: reduce) {
  #subtitle-box.show { animation-duration: 0.01s; }
  #sv-panel.visible .sv-scan { animation: none; }
}"""

BLAITE_CUSTOM_TEST_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SilentVoice Custom OBS Template</title>
  <!--
    =========================================================================
    SILENTVOICE OBS CUSTOM TEMPLATE — HUMAN & AI GUIDE
    =========================================================================
    1. FILES:
       - `index.html` — layout, `/ws` WebSocket client, and animation logic.
       - `style.css`  — stylesheet (automatically injected at `{{CSS}}`).
       After saving your edits, click "🔄 Refresh OBS" in SilentVoice settings.

    2. WEBSOCKET CONTRACT (/ws) — MESSAGES SENT BY SILENTVOICE:
       - { type: "speech", text: string, duration: number (sec),
           char_delays?: number[] (ms per Unicode codepoint!), style: object }
       - { type: "live_typing", text: string, style: object }
       - { type: "clear" } / { type: "clear_live_typing" }
       - { type: "style", style: object }
       - { type: "reload" } -> MUST call window.location.reload()

    3. CRITICAL AUDIO-SYNC RULES (FOR AI ASSISTANTS):
       - SilentVoice sends `speech` BEFORE opening the Windows audio device
         and before any leading silence inside the synthesized TTS audio.
       - `msg.char_delays` already includes the playback startup offset (~85 ms)
         and exact audio-envelope timestamps (in ms) for each character in
         `Array.from(msg.text)` matching real speech onset.
       - NEVER reveal the final subtitle at `0 ms` and NEVER use a fixed cap
         like `Math.min(90, ...)` per word — that makes text appear earlier
         than the spoken voice and drift ahead!
       - ALWAYS synchronize the container reveal with `firstDelay = char_delays[0]`
         (or ~85 ms when `char_delays` is absent).
       - THIS TEMPLATE reveals the WHOLE text at once (no per-word spans):
         the full text materializes together at `firstDelay`. If you switch
         back to per-word reveal, use `msg.char_delays[charIdx]` per word.

    4. STRUCTURE OF THE "FUTURISTIC SIGNAL" TEMPLATE:
       - `#sv-panel` — outer panel; state classes `visible`, `live-typing`,
         `speaking` are toggled on it.
       - `#subtitle-box` — plain text container, rewritten by JS. Gets class
         `show` to run the materialize animation. Do not put decoration in it.
       - `speaking` runs the equalizer from first-word time until audio ends.
    =========================================================================
  -->
  <style>{{CSS}}</style>
</head>
<body>
  <div class="sv-wrap">
    <div id="sv-panel">
      <div class="sv-frame">
        <div class="sv-inner">
          <div class="sv-head">
            <div class="sv-eq"><i></i><i></i><i></i><i></i><i></i></div>
            <span id="sv-label">VOICE TRANSMISSION</span>
            <span class="sv-rule"></span>
            <span class="sv-ticks"></span>
          </div>
          <div id="subtitle-box" aria-live="polite"></div>
          <span class="sv-scan"></span>
        </div>
      </div>
    </div>
  </div>
  <script>
    const panel = document.getElementById('sv-panel');
    const box = document.getElementById('subtitle-box');
    const label = document.getElementById('sv-label');
    let startTimer = null;
    let hideTimer = null;
    let speakTimer = null;

    function clearAllTimers() {
      if (startTimer) clearTimeout(startTimer);
      if (hideTimer) clearTimeout(hideTimer);
      if (speakTimer) clearTimeout(speakTimer);
      startTimer = null;
      hideTimer = null;
      speakTimer = null;
    }

    function hideBox() {
      clearAllTimers();
      panel.classList.remove('visible', 'live-typing', 'speaking');
      box.classList.remove('show');
    }

    function showSpeech(msg) {
      clearAllTimers();
      panel.classList.remove('visible', 'live-typing', 'speaking');
      box.classList.remove('show');
      label.textContent = 'VOICE TRANSMISSION';

      const text = String(msg.text || '');
      const chars = Array.from(text);
      const rawDelays = Array.isArray(msg.char_delays) && msg.char_delays.length >= chars.length
        ? msg.char_delays
        : null;
      const totalMs = Math.max(400, Number(msg.duration || 2.0) * 1000);
      const defaultStartMs = 85;

      // Whole text is placed at once; it stays hidden until the voice starts.
      box.textContent = text;
      void panel.offsetWidth;

      // First spoken character (skip leading whitespace) defines the start time.
      let firstIdx = chars.findIndex((c) => !/\s/.test(c));
      if (firstIdx < 0) firstIdx = 0;
      const firstDelay = rawDelays
        ? Math.max(0, Number(rawDelays[firstIdx] || 0))
        : defaultStartMs;
      const lastIdx = Math.max(0, chars.length - 1);
      const lastDelay = rawDelays ? Math.max(0, Number(rawDelays[lastIdx] || 0)) : firstDelay;

      startTimer = setTimeout(() => {
        panel.classList.add('visible', 'speaking');
        box.classList.add('show');
      }, firstDelay);

      // Equalizer stops when the audio is over.
      const endMs = Math.max(totalMs + defaultStartMs, lastDelay + 150);
      speakTimer = setTimeout(() => panel.classList.remove('speaking'), endMs);

      const fadeSec = Number((msg.style && msg.style.fade_sec) || 3.5);
      hideTimer = setTimeout(hideBox, endMs + fadeSec * 1000);
    }

    function showLiveTyping(text) {
      clearAllTimers();
      const clean = String(text || '');
      if (!clean) {
        hideBox();
        return;
      }
      box.textContent = clean;
      label.textContent = 'INPUT STREAM';
      panel.classList.remove('speaking');
      panel.classList.add('visible', 'live-typing');
      box.classList.add('show');
      hideTimer = setTimeout(hideBox, 12000);
    }

    function connectWS() {
      const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
      const ws = new WebSocket(proto + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'reload') {
            window.location.reload();
          } else if (msg.type === 'speech') {
            showSpeech(msg);
          } else if (msg.type === 'live_typing') {
            showLiveTyping(msg.text);
          } else if (msg.type === 'clear') {
            hideBox();
          } else if (msg.type === 'clear_live_typing' && panel.classList.contains('live-typing')) {
            hideBox();
          }
        } catch (e) {}
      };
      ws.onclose = () => setTimeout(connectWS, 1500);
      ws.onerror = () => ws.close();
    }
    connectWS();
  </script>
</body>
</html>"""


def ensure_custom_obs_templates(
    target_dir: Optional[Path] = None,
    overwrite: bool = False,
    reset_name: Optional[str] = None,
) -> Path:
    """Create starter index.html/style.css and Blaite_castom_test preset without overwriting edits."""
    tpl_dir = Path(target_dir) if target_dir is not None else OBS_TEMPLATES_DIR
    tpl_dir.mkdir(parents=True, exist_ok=True)
    target_reset = (reset_name or "").strip()

    html_path = tpl_dir / "index.html"
    css_path = tpl_dir / "style.css"
    reset_index = overwrite and target_reset in ("", "index")
    if reset_index or not html_path.exists():
        html_path.write_text(DEFAULT_CUSTOM_OBS_HTML, encoding="utf-8")
    if reset_index or not css_path.exists():
        css_path.write_text(DEFAULT_CUSTOM_OBS_CSS, encoding="utf-8")

    b_html = tpl_dir / f"{BLAITE_CUSTOM_TEST_NAME}.html"
    b_css = tpl_dir / f"{BLAITE_CUSTOM_TEST_NAME}.css"
    reset_blaite = overwrite and target_reset == BLAITE_CUSTOM_TEST_NAME
    if reset_blaite or not b_html.exists():
        b_html.write_text(BLAITE_CUSTOM_TEST_HTML, encoding="utf-8")
    if reset_blaite or not b_css.exists():
        b_css.write_text(BLAITE_CUSTOM_TEST_CSS, encoding="utf-8")
    return tpl_dir


def format_custom_template_label(stem: str) -> str:
    """Convert filename stem like `Yri_1` or `Blaite_castom_test` to `Yri 1` / `Blaite castom test`."""
    clean = " ".join(str(stem or "").replace("_", " ").split())
    return clean or str(stem)


def list_custom_obs_templates(target_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """Return discovered `<Name>.html` templates in OBS_TEMPLATES_DIR (excluding `index.html`)."""
    tpl_dir = ensure_custom_obs_templates(target_dir=target_dir, overwrite=False)
    entries: List[Dict[str, str]] = []
    seen_low: set[str] = set()
    for html_file in sorted(tpl_dir.glob("*.html"), key=lambda p: (0 if p.stem == BLAITE_CUSTOM_TEST_NAME else 1, p.stem.lower())):
        stem = html_file.stem
        if stem.lower() == "index" or stem.lower() in seen_low:
            continue
        seen_low.add(stem.lower())
        css_file = tpl_dir / f"{stem}.css"
        if not css_file.is_file():
            css_file = next((c for c in tpl_dir.glob("*.css") if c.stem.lower() == stem.lower()), css_file)
        entries.append({
            "mode": f"custom:{stem}",
            "stem": stem,
            "label": format_custom_template_label(stem),
            "html_name": html_file.name,
            "css_name": css_file.name if css_file.is_file() else "",
        })
    return entries


def _resolve_custom_template_paths(tpl_dir: Path, mode: str) -> Tuple[Path, Optional[Path]]:
    raw = str(mode or "custom").strip()
    if raw.lower().startswith("custom:"):
        stem = raw.split(":", 1)[1].strip()
        if not stem or Path(stem).name != stem or "/" in stem or "\\" in stem or ".." in stem:
            raise ValueError(f"Invalid custom template name: {raw!r}")
        html_path = tpl_dir / f"{stem}.html"
        if not html_path.is_file():
            raise FileNotFoundError(f"Custom OBS template not found: {html_path}")
        css_candidate = tpl_dir / f"{stem}.css"
        if not css_candidate.is_file():
            css_candidate = next((c for c in tpl_dir.glob("*.css") if c.stem.lower() == stem.lower()), css_candidate)
        return html_path, (css_candidate if css_candidate.is_file() else None)
    return tpl_dir / "index.html", tpl_dir / "style.css"


def render_custom_page(target_dir: Optional[Path] = None, mode: str = "custom") -> str:
    """Render selected custom HTML + CSS template (substituting {{CSS}})."""
    tpl_dir = ensure_custom_obs_templates(target_dir=target_dir, overwrite=False)
    html_path, css_path = _resolve_custom_template_paths(tpl_dir, mode)
    html_text = html_path.read_text(encoding="utf-8")
    css_text = css_path.read_text(encoding="utf-8") if css_path is not None and css_path.is_file() else ""
    if "{{CSS}}" in html_text:
        return html_text.replace("{{CSS}}", css_text)
    if css_text and "</head>" in html_text:
        return html_text.replace("</head>", f"<style>\n{css_text}\n</style>\n</head>", 1)
    return html_text


def find_free_port(preferred_port: int = 8765) -> int:
    """Перевіряє, чи вільний бажаний порт на 127.0.0.1, інакше знаходить наступний вільний."""
    for port in range(preferred_port, preferred_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class OBSSubtitleServer(QObject):
    """
    Локальний сервер субтитрів для OBS Studio та зовнішніх HTTP-команд (Stream Deck).
    Працює у фоновому потоці asyncio.
    """

    server_status_changed = Signal(bool, int, str)  # (running, active_port, url)
    external_speak_requested = Signal(str)
    external_toggle_input_requested = Signal()
    external_stop_requested = Signal()
    external_repeat_requested = Signal()
    external_quick_phrase_requested = Signal(int)
    preview_settings_saved = Signal(object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Optional[web.AppRunner] = None
        self._clients: Set[web.WebSocketResponse] = set()
        self._active_port: int = int(config.get("obs_port", 8765))
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_port(self) -> int:
        return self._active_port

    @property
    def browser_source_url(self) -> str:
        return f"http://127.0.0.1:{self._active_port}/"

    def get_style_payload(self) -> Dict[str, Any]:
        obs_style = str(config.get("obs_style", "text_only"))
        anim_id = str(get_entry_spec(str(config.get("obs_anim_style", "typewriter")))["id"])
        motion_id = str(get_motion_spec(str(config.get("obs_motion_effect", "none")))["id"])
        return {
            "style": obs_style,
            "font_family": config.get("obs_font_family", "Montserrat"),
            "font_size": int(config.get("obs_font_size", 24)),
            "text_color": config.get("obs_text_color", "#FFFFFF"),
            "text_opacity": int(config.get("obs_text_opacity", 100)),
            "outline_width": int(config.get("obs_outline_width", 3)),
            "shadow_blur": int(config.get("obs_shadow_blur", 10)),
            "shadow_enabled": int(config.get("obs_shadow_blur", 10)) > 0,
            "bg_enabled": obs_style == "boxed",
            "bg_color": config.get("obs_bg_color", "#13151F"),
            "border_color": config.get("obs_border_color", "#7C5CFF"),
            "bg_opacity": int(config.get("obs_bg_opacity", 78)),
            "border_radius": int(config.get("obs_border_radius", 14)),
            "anim_style": anim_id,
            "animate_block": bool(config.get("obs_animate_block", True)),
            "animate_motion_block": bool(config.get("obs_animate_motion_block", False)),
            "motion_effect": motion_id,
            "anim_params": resolve_anim_params("entry", anim_id, config.get("obs_anim_params")),
            "motion_params": resolve_anim_params("motion", motion_id, config.get("obs_motion_params")),
            "show_live_typing": bool(config.get("obs_show_live_typing", False)),
            "live_typing_mode": str(config.get("obs_live_typing_mode", "typing_and_enter")),
            "fade_sec": float(config.get("obs_fade_sec", 3.5)),
            "show_cursor": bool(config.get("obs_show_cursor", False)),
            "cursor_char": str(config.get("obs_cursor_char", "▌")),
            "cursor_color": str(config.get("obs_cursor_color", "#9B82FF")),
            "top_fade": bool(config.get("obs_top_fade", True)),
        }

    @staticmethod
    def _clean_preview_params(kind: str, spec: Dict[str, Any], raw: Any) -> Dict[str, Any]:
        raw = raw if isinstance(raw, dict) else {}
        controls = spec.get("controls", [])
        seed: Dict[str, Any] = {}

        if spec.get("supports_split"):
            split = str(raw.get("split", spec.get("defaults", {}).get("split", "whole")))
            seed["split"] = split if split in ("whole", "words", "chars") else "whole"
        if kind == "motion":
            trigger = str(raw.get("trigger", spec.get("defaults", {}).get("trigger", "after")))
            seed["trigger"] = trigger if trigger in ("after", "simultaneous", "only") else "after"
        for control in controls:
            if control.get("type") != "select":
                continue
            key = str(control.get("key", ""))
            allowed = {str(option.get("value")) for option in control.get("options", [])}
            if key and str(raw.get(key, "")) in allowed:
                seed[key] = str(raw[key])

        params = get_variant_defaults(kind, spec, seed)
        ranges: Dict[str, tuple] = {}
        if "duration" in spec.get("defaults", {}):
            ranges["duration"] = (600, 6000, 100) if kind == "motion" else (150, 2500, 50)
        if kind == "entry" and spec.get("supports_split"):
            ranges["stagger"] = (10, 200, 5)
        for control in controls:
            if control.get("type") == "range" and control.get("key"):
                ranges[str(control["key"])] = (
                    float(control["min"]), float(control["max"]), float(control["step"])
                )

        for key, (minimum, maximum, step) in ranges.items():
            if key not in raw:
                continue
            try:
                value = float(raw[key])
            except (TypeError, ValueError):
                continue
            value = min(maximum, max(minimum, value))
            params[key] = int(round(value)) if float(step).is_integer() else round(value, 4)
        return params

    def _save_preview_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from core.i18n import tr

        style = str(payload.get("style", "text_only"))
        if style not in ("text_only", "boxed"):
            raise ValueError(tr("obs.preview.error_style", "Unknown subtitle style"))

        entry_id = str(payload.get("anim_style", "typewriter"))
        motion_id = str(payload.get("motion_effect", "none"))
        if entry_id not in {item["id"] for item in ENTRY_ANIMATIONS}:
            raise ValueError(tr("obs.preview.error_entry", "Unknown entry animation"))
        if motion_id not in {item["id"] for item in MOTION_ANIMATIONS}:
            raise ValueError(tr("obs.preview.error_motion", "Unknown motion animation"))

        entry_spec = get_entry_spec(entry_id)
        motion_spec = get_motion_spec(motion_id)
        entry_params = self._clean_preview_params("entry", entry_spec, payload.get("anim_params"))
        motion_params = self._clean_preview_params("motion", motion_spec, payload.get("motion_params"))
        entry_map = copy.deepcopy(config.get("obs_anim_params", {}) or {})
        motion_map = copy.deepcopy(config.get("obs_motion_params", {}) or {})

        def store_params(target: Dict[str, Any], kind: str, spec: Dict[str, Any], params: Dict[str, Any]) -> None:
            custom = target.get(spec["id"])
            custom = copy.deepcopy(custom) if isinstance(custom, dict) else {}
            by_variant = custom.get("by_variant")
            by_variant = copy.deepcopy(by_variant) if isinstance(by_variant, dict) else {}
            by_variant[get_variant_key(kind, spec, params)] = copy.deepcopy(params)
            custom["by_variant"] = by_variant
            for key in ("split", "trigger"):
                if key in params:
                    custom[key] = params[key]
            for control in spec.get("controls", []):
                key = control.get("key")
                if control.get("type") == "select" and key in params:
                    custom[key] = params[key]
            target[spec["id"]] = custom

        store_params(entry_map, "entry", entry_spec, entry_params)
        store_params(motion_map, "motion", motion_spec, motion_params)
        settings = {
            "obs_style": style,
            "obs_bg_enabled": style == "boxed",
            "obs_anim_style": entry_id,
            "obs_motion_effect": motion_id,
            "obs_animate_block": bool(payload.get("animate_block", False)),
            "obs_animate_motion_block": bool(payload.get("animate_motion_block", False)),
            "obs_anim_params": entry_map,
            "obs_motion_params": motion_map,
        }
        config.update(settings)
        saved = {
            "style": style,
            "anim_style": entry_id,
            "motion_effect": motion_id,
            "animate_block": settings["obs_animate_block"],
            "animate_motion_block": settings["obs_animate_motion_block"],
            "anim_params": entry_params,
            "motion_params": motion_params,
        }
        self.preview_settings_saved.emit(saved)
        self.broadcast_style()
        return saved

    def start(self, preferred_port: Optional[int] = None) -> None:
        if self._running:
            return
        if preferred_port is None:
            preferred_port = int(config.get("obs_port", 8765))
        self._active_port = find_free_port(preferred_port)
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="OBSSubtitleServerThread")
        self._thread.start()

    def stop(self) -> None:
        if not self._running or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)

    def restart(self, new_port: int) -> None:
        if self._running:
            self.stop()
            # Невелика пауза для звільнення сокета
            threading.Timer(0.25, lambda: self.start(new_port)).start()
        else:
            self.start(new_port)

    def broadcast_style(self) -> None:
        self._broadcast({"type": "style", "style": self.get_style_payload(), "i18n": self.get_page_i18n()})

    @staticmethod
    def get_page_i18n() -> Dict[str, str]:
        """Рядки веб-сторінки субтитрів мовою, обраною в програмі."""
        from core.i18n import get_ui_language, tr

        messages = {
            "lang": get_ui_language() or "en",
            "connecting": tr("obs.web.connecting", "🟡 Connecting to SilentVoice..."),
            "connected": tr("obs.web.connected", "🟢 Connected to SilentVoice (OBS preview)"),
            "waiting": tr("obs.web.waiting", "🟡 Waiting for connection to SilentVoice..."),
            "test_btn": tr("obs.web.test_btn", "✨ Test subtitles"),
            "sample_text": tr("obs.sample_text", "Hello viewers! Testing OBS subtitles ✨"),
            "obs.preview.controls_aria": tr("obs.preview.controls_aria", "Animation preview settings"),
            "obs.preview.mode_aria": tr("obs.preview.mode_aria", "Subtitle preview style"),
            "obs.preview.text_only": tr("obs.preview.text_only", "No background"),
            "obs.preview.boxed": tr("obs.preview.boxed", "In a block"),
            "obs.preview.animate_entry_block": tr("obs.preview.animate_entry_block", "Animate block with entry"),
            "obs.preview.animate_motion_block": tr("obs.preview.animate_motion_block", "Apply motion to block"),
            "obs.preview.reset_animations": tr("obs.preview.reset_animations", "Reset entry and motion"),
            "obs.preview.motion_title": tr("obs.preview.motion_title", "Live animation"),
            "obs.preview.motion_description": tr("obs.preview.motion_description", "Choose motion to run after or with the entry animation."),
            "obs.preview.motion_label": tr("obs.preview.motion_label", "Motion animation"),
            "obs.preview.entry_title": tr("obs.preview.entry_title", "Entry animation"),
            "obs.preview.entry_description": tr("obs.preview.entry_description", "Choose an effect and adjust its settings."),
            "obs.preview.entry_label": tr("obs.preview.entry_label", "Entry effect"),
            "obs.preview.save": tr("obs.preview.save", "Save to app"),
            "obs.preview.saving": tr("obs.preview.saving", "Saving…"),
            "obs.preview.saved": tr("obs.preview.saved", "Saved to app"),
            "obs.preview.no_connection": tr("obs.preview.no_connection", "Not connected to the app"),
            "obs.preview.save_failed": tr("obs.preview.save_failed", "Could not save settings"),
            "obs.preview.save_timeout": tr("obs.preview.save_timeout", "No save confirmation received"),
            "obs.preview.error_style": tr("obs.preview.error_style", "Unknown subtitle style"),
            "obs.preview.error_entry": tr("obs.preview.error_entry", "Unknown entry animation"),
            "obs.preview.error_motion": tr("obs.preview.error_motion", "Unknown motion animation"),
            "obs.preview.no_motion_params": tr("obs.preview.no_motion_params", "No additional settings for this mode."),
            "obs.preview.typewriter_hint": tr("obs.preview.typewriter_hint", "Letters appear in time with the spoken voice—like lip-sync, but for text."),
            "obs.preview.ms": tr("obs.preview.ms", "ms"),
            "obs.ctrl.duration": tr("obs.ctrl.duration", "Entry duration"),
            "obs.ctrl.motion_duration": tr("obs.ctrl.motion_duration", "Cycle duration"),
            "obs.ctrl.split": tr("obs.ctrl.split", "Text split"),
            "obs.ctrl.stagger": tr("obs.ctrl.stagger", "Stagger"),
            "obs.ctrl.motion_trigger": tr("obs.ctrl.motion_trigger", "Start relative to entry"),
            "obs.opt.split_whole": tr("obs.opt.split_whole", "Whole text"),
            "obs.opt.split_words": tr("obs.opt.split_words", "By words"),
            "obs.opt.split_chars": tr("obs.opt.split_chars", "By letters"),
            "obs.opt.trigger_after": tr("obs.opt.trigger_after", "After text appears"),
            "obs.opt.trigger_sim": tr("obs.opt.trigger_sim", "Simultaneously with entry"),
            "obs.opt.trigger_only": tr("obs.opt.trigger_only", "Motion only (no entrance)"),
        }
        for spec in ENTRY_ANIMATIONS + MOTION_ANIMATIONS:
            name_key = spec.get("i18n_key")
            if name_key:
                messages[name_key] = tr(name_key, spec.get("name", ""))
            for control in spec.get("controls", []):
                control_key = control.get("i18n")
                if control_key:
                    messages[control_key] = tr(control_key, control.get("label", ""))
                for option in control.get("options", []):
                    option_key = option.get("i18n")
                    if option_key:
                        messages[option_key] = tr(option_key, option.get("label", ""))
        return messages

    def render_page(self) -> str:
        tpl_mode = str(config.get("obs_template_mode", "builtin") or "builtin").strip()
        if tpl_mode.lower() == "custom" or tpl_mode.lower().startswith("custom:"):
            try:
                return render_custom_page(mode=tpl_mode)
            except Exception as exc:
                print(f"[OBSServer] Не вдалося завантажити власний шаблон ({tpl_mode}): {exc}")
        i18n = self.get_page_i18n()
        i18n_json = json.dumps(i18n, ensure_ascii=False).replace("</", "<\\/")
        animation_meta = {
            "entry": [
                {key: item[key] for key in ("id", "name", "i18n_key", "defaults", "supports_split", "controls", "split_defaults") if key in item}
                for item in ENTRY_ANIMATIONS
            ],
            "motion": [
                {key: item[key] for key in ("id", "name", "i18n_key", "defaults", "supports_split", "controls", "split_defaults") if key in item}
                for item in MOTION_ANIMATIONS
            ],
        }
        animation_meta_json = json.dumps(animation_meta, ensure_ascii=False).replace("</", "<\\/")
        lang = "".join(ch for ch in str(i18n.get("lang", "en")) if ch.isalnum() or ch == "-") or "en"
        return (
            OBS_HTML_PAGE.replace("__SV_LANG__", lang)
            .replace("__SV_I18N_JSON__", i18n_json)
            .replace("__SV_ANIM_META_JSON__", animation_meta_json)
        )

    def broadcast_speech(self, text: str, duration: float = 2.5, char_delays: Optional[Any] = None, force: bool = False) -> None:
        if not config.get("obs_enabled", True):
            return
        if (
            not force
            and bool(config.get("obs_show_live_typing", False))
            and str(config.get("obs_live_typing_mode", "typing_and_enter")) == "typing_only"
        ):
            return
        payload: Dict[str, Any] = {
            "type": "speech",
            "text": text,
            "duration": max(0.8, float(duration)),
            "style": self.get_style_payload(),
        }
        if char_delays:
            payload["char_delays"] = list(char_delays)
        self._broadcast(payload)

    def broadcast_live_typing(self, text: str) -> None:
        if not self._clients or not config.get("obs_enabled", True) or not config.get("obs_show_live_typing", False):
            return
        self._broadcast({
            "type": "live_typing",
            "text": text,
            "style": self.get_style_payload(),
        })

    def broadcast_clear(self) -> None:
        if not self._clients:
            return
        self._broadcast({"type": "clear"})

    def broadcast_clear_live_typing(self) -> None:
        if not self._clients or not config.get("obs_enabled", True):
            return
        self._broadcast({"type": "clear_live_typing"})

    def broadcast_reload(self) -> None:
        if not self._clients:
            return
        self._broadcast({"type": "reload"})

    def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self._running or not self._loop or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)

    async def _broadcast_async(self, payload: Dict[str, Any]) -> None:
        if not self._clients:
            return
        raw = json.dumps(payload, ensure_ascii=False)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_str(raw)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    @staticmethod
    def _is_trusted_local_request(request: web.Request) -> bool:
        """
        Blocks cross-origin browser attacks (CSRF / Cross-Origin WebSocket Hijacking)
        against the local OBS server while keeping OBS Browser Source, localhost
        preview, Stream Deck, curl, and local scripts working without friction.
        """
        from urllib.parse import urlsplit

        allowed_hosts = {"127.0.0.1", "localhost", "::1"}
        sec_fetch_site = str(request.headers.get("Sec-Fetch-Site", "") or "").strip().lower()
        if sec_fetch_site and sec_fetch_site not in ("same-origin", "none"):
            return False

        for header_name in ("Origin", "Referer"):
            raw_val = str(request.headers.get(header_name, "") or "").strip()
            if not raw_val:
                continue
            if raw_val.lower() == "null":
                return False
            parsed = urlsplit(raw_val)
            host = (parsed.hostname or "").strip().lower()
            if parsed.scheme.lower() not in ("http", "https") or host not in allowed_hosts:
                return False
        return True

    async def _handle_index(self, request: web.Request) -> web.Response:
        return web.Response(
            text=self.render_page(),
            content_type="text/html",
            charset="utf-8",
            headers={
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "SAMEORIGIN",
            },
        )

    async def _handle_ws(self, request: web.Request) -> web.StreamResponse:
        if not self._is_trusted_local_request(request):
            return web.Response(status=403, text="Forbidden cross-origin WebSocket request")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        # Відразу надсилаємо поточний стиль при підключенні OBS
        await ws.send_str(json.dumps({"type": "style", "style": self.get_style_payload(), "i18n": self.get_page_i18n()}, ensure_ascii=False))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT and request.query.get("preview") == "1":
                    try:
                        incoming = json.loads(msg.data)
                        if incoming.get("type") == "save_preview_settings":
                            self._save_preview_settings(incoming.get("settings", {}))
                            await ws.send_str(json.dumps({"type": "save_result", "ok": True}, ensure_ascii=False))
                        else:
                            continue
                    except Exception as exc:
                        from core.i18n import tr
                        await ws.send_str(json.dumps({
                            "type": "save_result",
                            "ok": False,
                            "error": str(exc) or tr("obs.preview.save_failed", "Could not save settings"),
                        }, ensure_ascii=False))
                if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _handle_api_speak(self, request: web.Request) -> web.Response:
        if not self._is_trusted_local_request(request):
            return web.json_response({"ok": False, "error": "Forbidden cross-origin request"}, status=403)
        text = request.query.get("text", "")
        if not text and request.can_read_body:
            try:
                body = await request.json()
                text = str(body.get("text", ""))
            except Exception:
                text = (await request.text()).strip()
        if text.strip():
            self.external_speak_requested.emit(text.strip())
            return web.json_response({"ok": True, "spoken": text.strip()})
        return web.json_response({"ok": False, "error": "Empty text"}, status=400)

    async def _handle_api_toggle_input(self, request: web.Request) -> web.Response:
        if not self._is_trusted_local_request(request):
            return web.json_response({"ok": False, "error": "Forbidden cross-origin request"}, status=403)
        self.external_toggle_input_requested.emit()
        return web.json_response({"ok": True})

    async def _handle_api_stop(self, request: web.Request) -> web.Response:
        if not self._is_trusted_local_request(request):
            return web.json_response({"ok": False, "error": "Forbidden cross-origin request"}, status=403)
        self.external_stop_requested.emit()
        return web.json_response({"ok": True})

    async def _handle_api_repeat(self, request: web.Request) -> web.Response:
        if not self._is_trusted_local_request(request):
            return web.json_response({"ok": False, "error": "Forbidden cross-origin request"}, status=403)
        self.external_repeat_requested.emit()
        return web.json_response({"ok": True})

    async def _handle_api_quick(self, request: web.Request) -> web.Response:
        if not self._is_trusted_local_request(request):
            return web.json_response({"ok": False, "error": "Forbidden cross-origin request"}, status=403)
        try:
            idx = int(request.match_info.get("idx", "1"))
            self.external_quick_phrase_requested.emit(idx)
            return web.json_response({"ok": True, "quick_index": idx})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        def _ignore_win_conn_reset(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
            exc = context.get("exception")
            if isinstance(exc, ConnectionResetError):
                return
            loop.default_exception_handler(context)

        self._loop.set_exception_handler(_ignore_win_conn_reset)
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_route("*", "/api/speak", self._handle_api_speak)
        app.router.add_route("*", "/api/toggle_input", self._handle_api_toggle_input)
        app.router.add_route("*", "/api/stop", self._handle_api_stop)
        app.router.add_route("*", "/api/repeat", self._handle_api_repeat)
        app.router.add_route("*", "/api/quick/{idx}", self._handle_api_quick)

        async def _start_server():
            self._runner = web.AppRunner(app, access_log=None)
            await self._runner.setup()
            site = web.TCPSite(self._runner, "127.0.0.1", self._active_port)
            await site.start()
            self._running = True
            self.server_status_changed.emit(True, self._active_port, self.browser_source_url)

        try:
            self._loop.run_until_complete(_start_server())
            self._loop.run_forever()
        except Exception as exc:
            print(f"[OBS Server] Помилка запуску: {exc}")
        finally:
            self._running = False
            try:
                self.server_status_changed.emit(False, self._active_port, "")
            except RuntimeError:
                pass
            self._loop.close()

    async def _shutdown_async(self) -> None:
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()
        runner = self._runner
        self._runner = None
        if runner:
            try:
                await runner.cleanup()
            except Exception:
                pass
        if self._loop and self._loop.is_running():
            self._loop.stop()


obs_server = OBSSubtitleServer()
