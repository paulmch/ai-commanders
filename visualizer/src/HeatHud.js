/**
 * Ship Registry heat bars: a thin bar under each hull bar showing the heat
 * sink fill on a 0..meltdown scale (tick at 100 %), amber from 90 % (guns
 * hold fire), red and pulsing above 100 %. Only drawn when the recording
 * carries sim_trace `heat`; old recordings keep the registry as it was.
 */
import { MELTDOWN_HEAT } from './render/Heat.js';

/** Bar width (% of the track) and level class for a heat fraction, into out. */
export function heatBar(heat, melt = MELTDOWN_HEAT, out = { width: 0, level: 'none' }) {
  if (!(heat >= 0)) { out.width = 0; out.level = 'none'; return out; }
  out.width = Math.round(Math.min(1, heat / melt) * 1000) / 10;
  out.level = heat >= 1 ? 'crit' : heat >= 0.9 ? 'hot' : heat >= 0.65 ? 'warm' : 'cool';
  return out;
}

const _bar = { width: 0, level: 'none' };

const _cache = new WeakMap();   // li -> {bar, fill, width, level, has}

function barFor(li) {
  let c = _cache.get(li);
  if (c) return c;
  const bar = document.createElement('span');
  bar.className = 'registry-heat';
  bar.hidden = true;
  bar.innerHTML = '<i></i>';
  (li.querySelector('.registry-identity') || li).appendChild(bar);
  c = { bar, fill: bar.firstChild, width: -1, level: '', pct: -1 };
  _cache.set(li, c);
  return c;
}

/** Called once per frame from main.js with SceneManager.heat (HeatFX). */
export function updateRegistryHeat(heatFX, time, root = typeof document !== 'undefined' ? document : null) {
  if (!heatFX || !root) return;
  const items = root.querySelectorAll('.ship-list li');
  for (const li of items) {
    const id = li.dataset.shipId;
    if (!heatFX.hasHeat(id)) {
      const c = _cache.get(li);
      if (c) c.bar.hidden = true;
      continue;
    }
    const c = barFor(li);
    const h = li.classList.contains('destroyed') ? NaN : heatFX.heatAt(id, time);
    const { width, level } = heatBar(h, MELTDOWN_HEAT, _bar);
    c.bar.hidden = level === 'none';
    if (width !== c.width) { c.width = width; c.fill.style.width = `${width}%`; }
    if (level !== c.level) { c.level = level; c.bar.dataset.level = level; li.dataset.heat = level; }
    const pct = h >= 0 ? Math.round(h * 100) : -1;
    if (pct !== c.pct) {
      c.pct = pct;
      c.bar.title = pct >= 0 ? `Heat ${pct}% (guns hold at 90%, radiators auto-extend at 100%, meltdown at ${Math.round(MELTDOWN_HEAT * 100)}%)` : '';
    }
  }
}
