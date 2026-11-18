import * as THREE from 'three';
import { getShipClassName } from './shipSilhouettes.js';
import { detonationTime, lastHitAt, reactorPulseRate, HIT_FLASH_S } from './DamageTimeline.js';
import { layoutTactical, groupWaves, wavesByTarget, waveLabel, shortName, DEFAULT_LABEL_MODE, normalizeLabelMode } from './TacticalLayout.js';

/** Inbound torpedoes closer than this (s) pulse the bracket red. */
export const THREAT_IMMINENT_S = 10;
/** A wave further out than this (s) is a forecast, drawn quiet. */
export const WAVE_DISTANT_S = 60;

/** Visual weight of a torpedo-wave chip: 'imminent' | 'inbound' | 'distant' | 'blind' (no live seeker). */
export function waveUrgency(wave) {
  if (wave.live === 0) return 'blind';
  return wave.eta < THREAT_IMMINENT_S ? 'imminent' : wave.eta < WAVE_DISTANT_S ? 'inbound' : 'distant';
}

/**
 * Fleet-range threat readout for one ship ('3 TORP · 6s'; '' when nothing
 * is inbound). `threat` is a WeaponTimeline.threatsAt entry: live rounds
 * plus seeker-blinded ones still closing (they can still strike).
 */
export function threatLabel(threat) {
  if (!threat || !(threat.count > 0)) return '';
  const eta = Number.isFinite(threat.eta) ? `${Math.max(0, Math.round(threat.eta))}s` : '--';
  return `${threat.count} TORP · ${eta}`;
}

/** Dying-hulk countdown text, `ttd` seconds to the reactor letting go. */
export function detonationLabel(ttd) {
  return Number.isFinite(ttd) ? `DETONATION ${Math.max(0, Math.ceil(ttd))}s` : 'REACTOR UNSTABLE';
}

/** Bracket flash 0..1 for a hit `age` seconds ago (fades over HIT_FLASH_S). */
export function hitFlash(age) {
  return age >= 0 && age <= HIT_FLASH_S ? (1 - age / HIT_FLASH_S) ** 1.5 : 0;
}

/** HUD panels contact brackets and labels keep clear of (looked up lazily: some mount after the overlay). */
export const CONTACT_PANEL_IDS = ['battleHeader', 'shipRegistry', 'shipTelemetry', 'decisionPanel', 'eventLog', 'liveHud',
  'viewControls', 'playbackControls', 'trackHud', 'commsFeed', 'killFeed', 'resultBanner',   // spectator HUD (pass 4)
  'directorChip', 'liveDeliberation', 'liveConnection'];                                      // live wire (pass 4)

const overlaps = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

/** Position a leader line element (TacticalLayout.leaderTo) relative to `ox, oy`. */
function setLeader(el, leader, ox, oy) {
  el.hidden = !leader;
  if (!leader) return;
  el.style.left = `${(leader.x - ox).toFixed(1)}px`;
  el.style.top = `${(leader.y - oy).toFixed(1)}px`;
  el.style.width = `${leader.len.toFixed(1)}px`;
  el.style.transform = `rotate(${leader.angle.toFixed(4)}rad)`;
}

/**
 * The fleet-range contact layer: a bracket on every ship too distant to
 * read by eye, plus chips laid out by TacticalLayout (L cycles the density:
 * auto / minimal / full). In auto a ship carries a compact chip (short
 * name, a hull bar once damaged), crowded ships of one side share a group
 * chip that lists them on hover, and only the selected or dying ship gets
 * the full card; hovering any chip (or its registry row) expands it on top
 * without moving anything else. Torpedo waves get one chip each by the
 * wave's leading round (count, target, time to impact); the target's
 * bracket turns red and pulses inside THREAT_IMMINENT_S. A hit flashes the
 * bracket; a dying hulk counts down to its reactor detonation. The C toggle
 * hides the whole layer and the engagement lines (render/TargetingLines.js).
 */
export class ContactOverlay {
  constructor(container, selectShip) {
    this.container = container;
    this.selectShip = selectShip;
    this.contacts = new Map();
    this.groupEls = [];
    this.waveEls = [];
    this.rounds = [];
    this.enabled = true;
    this.mode = DEFAULT_LABEL_MODE;
    this.hoverId = null;
    this.point = new THREE.Vector3();
    this.view = new THREE.Vector3();
    this.panels = [];
    this.last = null;   // this frame's layout (harness, tests)
  }

  setMode(mode) { this.mode = normalizeLabelMode(mode); }

  /** Registry row hover: expand that ship's chip as if hovered. */
  setHover(id) {
    if (this.hoverId === id) return;
    this.contacts.get(this.hoverId)?.button.classList.remove('linked');
    this.hoverId = id;
    this.contacts.get(id)?.button.classList.add('linked');
  }

  /** The HUD panel elements (re-resolved until every one has mounted). */
  panelElements() {
    if (this.panels.length < CONTACT_PANEL_IDS.length) {
      this.panels = CONTACT_PANEL_IDS.map(id => document.getElementById(id)).filter(Boolean);
    }
    return this.panels;
  }

  populate(ships) {
    this.container.replaceChildren();
    this.contacts.clear();
    this.groupEls = [];
    this.waveEls = [];
    this.hoverId = null;
    for (const [id, info] of Object.entries(ships)) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `ship-contact ${info.faction}`;
      button.dataset.shipId = id;
      button.setAttribute('aria-label', `Inspect ${info.name || id}`);
      button.innerHTML = '<span class="contact-bracket"></span><span class="contact-leader"></span><span class="contact-label"><span class="contact-name"></span><span class="contact-detail"></span><span class="contact-health"><i></i></span></span>';
      button.addEventListener('click', () => this.selectShip(id));
      // pointer on the chip: its engagement lines come up too (TargetingLines.focusId)
      button.addEventListener('mouseenter', () => this.setHover(id));
      button.addEventListener('mouseleave', () => { if (this.hoverId === id) this.setHover(null); });
      this.container.appendChild(button);
      this.contacts.set(id, {
        button, info, short: shortName(info.name || id), label: button.querySelector('.contact-label'),
        name: button.querySelector('.contact-name'),
        detail: button.querySelector('.contact-detail'), leader: button.querySelector('.contact-leader'),
        health: button.querySelector('.contact-health i'), bracket: button.querySelector('.contact-bracket'),
        kind: null, nameText: null
      });
    }
  }

  /** Pooled group chip i: lists its ships on hover (click a row to select one). */
  groupEl(i) {
    while (this.groupEls.length <= i) {
      const el = document.createElement('div');
      el.className = 'contact-group';
      el.innerHTML = '<span class="group-leader"></span><span class="group-chip"><span class="group-text"></span><span class="group-list"></span></span>';
      this.container.appendChild(el);
      this.groupEls.push({ el, chip: el.querySelector('.group-chip'), text: el.querySelector('.group-text'),
        list: el.querySelector('.group-list'), leader: el.querySelector('.group-leader'), key: null });
    }
    return this.groupEls[i];
  }

  /** Pooled torpedo-wave chip i (click: select the target). */
  waveEl(i) {
    while (this.waveEls.length <= i) {
      const el = document.createElement('div');
      el.className = 'wave-contact';
      el.innerHTML = '<span class="wave-leader"></span><button type="button" class="wave-chip"></button>';
      this.container.appendChild(el);
      const rec = { el, chip: el.querySelector('.wave-chip'), leader: el.querySelector('.wave-leader'), target: null };
      rec.chip.addEventListener('click', () => rec.target && this.selectShip(rec.target));
      this.waveEls.push(rec);
    }
    return this.waveEls[i];
  }

  update(scene, states, selectedId) {
    this.container.hidden = !this.enabled;
    if (scene.weapons?.engagements) scene.weapons.engagements.enabled = this.enabled;
    if (!this.enabled) return;
    const now = scene.currentTime || 0;
    const tl = scene.weapons?.timeline;
    const damage = scene.damageFX?.timeline || null;
    const width = window.innerWidth, height = window.innerHeight;
    const camera = scene.camera;
    camera.updateMatrixWorld();
    const panels = this.panelElements().filter(panel => !panel.hidden && getComputedStyle(panel).visibility !== 'hidden').map(panel => {
      const r = panel.getBoundingClientRect();
      return { x: r.left - 6, y: r.top - 6, w: r.width + 12, h: r.height + 12 };
    }).filter(r => r.w > 12 && r.h > 12);
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const project = (pos) => {
      this.point.copy(pos).project(camera);
      if (Math.abs(this.point.x) > 0.98 || Math.abs(this.point.y) > 0.96 || this.point.z > 1) return null;
      return { x: (this.point.x + 1) * width / 2, y: (1 - this.point.y) * height / 2 };
    };
    const ships = [];
    const screen = new Map();
    for (const [id, contact] of this.contacts) {
      contact.button.hidden = true;
      const ship = scene.ships.get(id), state = states?.[id];
      if (!ship?.visible || !state || state.destroyed) continue;
      this.view.copy(ship.position).applyMatrix4(camera.matrixWorldInverse);
      const depth = -this.view.z;
      const pixels = (ship.userData.drawn || ship.userData.size).length * height / (2 * depth * tanHalf);
      if (depth <= camera.near || pixels > 90) continue;
      const p = project(ship.position);
      if (!p) continue;
      if (panels.some(r => overlaps({ x: p.x - 8, y: p.y - 8, w: 16, h: 16 }, r))) continue;
      const s = { id, faction: contact.info.faction, x: p.x, y: p.y, r: pixels / 2, name: contact.info.name || id,
        short: contact.short, type: contact.info.type, dying: !!state.dying };
      ships.push(s);
      screen.set(id, s);
    }

    // Torpedo waves, each anchored at its leading round's drawn position
    // (Projectiles places tr.position this frame, before this overlay runs)
    const waves = [];
    const threatened = new Map();   // target id -> soonest eta
    if (tl?.roundsAt) {
      const rounds = tl.roundsAt(now, this.rounds);
      const byId = new Map(rounds.map(r => [r.id, r]));
      for (const w of wavesByTarget(groupWaves(rounds))) {
        if (states?.[w.target]?.dying) continue;
        threatened.set(w.target, Math.min(threatened.get(w.target) ?? Infinity, w.eta));
        const head = byId.get(w.head)?.track;
        const t = screen.get(w.target);
        const hp = head?.visible && head.position ? project(head.position) : null;
        const anchor = hp || (t ? { x: t.x, y: t.y } : null);
        if (!anchor || panels.some(r => overlaps({ x: anchor.x - 4, y: anchor.y - 4, w: 8, h: 8 }, r))) continue;
        waves.push({ ...w, x: anchor.x, y: anchor.y, tx: t?.x ?? anchor.x, ty: t?.y ?? anchor.y,
          text: waveLabel(w, this.contacts.get(w.target)?.short || '') });
      }
    }

    const layout = layoutTactical({ view: { width, height, top: 8, bottom: 8 }, panels, ships, waves,
      mode: this.mode, selectedId });
    this.last = layout;

    for (const s of ships) {
      const contact = this.contacts.get(s.id);
      const { button, info, label, name, detail, leader, health, bracket } = contact;
      const state = states[s.id];
      const place = layout.ships.get(s.id);
      const x = s.x, y = s.y;
      button.hidden = false;
      button.style.transform = `translate(${(x - 10).toFixed(1)}px, ${(y - 10).toFixed(1)}px)`;
      button.classList.toggle('selected', s.id === selectedId);
      button.classList.toggle('dying', !!state.dying);
      const hull = Math.round(THREE.MathUtils.clamp(state.hull ?? 100, 0, 100));
      const sd = damage?.ships.get(s.id);
      let ttd = NaN;
      if (state.dying) {
        const det = detonationTime(sd);
        ttd = det !== null ? det - now : NaN;
        // the bracket pulse quickens with the reactor (DamageTimeline.reactorPulseRate)
        button.style.setProperty('--pulse', `${(0.5 / reactorPulseRate(Number.isFinite(ttd) ? ttd : 30)).toFixed(3)}s`);
      }
      const eta = threatened.get(s.id);
      button.classList.toggle('threatened', eta !== undefined);
      button.classList.toggle('imminent', eta !== undefined && eta < THREAT_IMMINENT_S);
      button.classList.toggle('damaged', hull < 100);
      // A hit flashes the bracket white for HIT_FLASH_S
      const flash = hitFlash(now - lastHitAt(sd, now));
      bracket.style.boxShadow = flash > 0 ? `0 0 ${(3 + 9 * flash).toFixed(1)}px rgba(255,244,228,${(0.35 + 0.65 * flash).toFixed(2)})` : '';
      bracket.style.borderColor = flash > 0 ? `rgba(255,250,240,${(0.45 + 0.55 * flash).toFixed(2)})` : '';
      const cls = getShipClassName(info.type);
      button.title = `${info.name || s.id} · ${cls} · hull ${hull}%`;
      health.style.width = `${hull}%`;
      const kind = place?.box ? place.kind : 'none';
      if (contact.kind !== kind) {
        contact.kind = kind;
        label.classList.toggle('card', kind === 'card');
        label.classList.toggle('chip', kind === 'chip');
      }
      // a card carries the full name, a chip (and its hover expansion) the short one
      const nameText = kind === 'card' ? (info.name || s.id) : contact.short;
      if (contact.nameText !== nameText) { contact.nameText = nameText; name.textContent = nameText; }
      detail.textContent = state.dying ? detonationLabel(ttd) : `${cls} · ${hull}%`;
      label.hidden = !place?.box;
      if (!place?.box) { leader.hidden = true; continue; }
      const box = place.box;
      // a chip left of its ship is anchored by its right edge, so the hover
      // expansion grows away from the hull, never across it
      const leftOf = box.x + box.w / 2 < x;
      label.style.left = leftOf ? 'auto' : `${(box.x - x + 10).toFixed(1)}px`;
      label.style.right = leftOf ? `${(x + 10 - box.x - box.w).toFixed(1)}px` : 'auto';
      label.style.top = `${(box.y - y + 10).toFixed(1)}px`;
      label.style.width = `${box.w}px`;
      setLeader(leader, place.leader, x - 10, y - 10);
    }

    layout.groups.forEach((g, i) => {
      const rec = this.groupEl(i);
      rec.el.hidden = !g.box;
      if (!g.box) return;
      rec.el.className = `contact-group ${g.faction}${g.ids.includes(this.hoverId) ? ' linked' : ''}`;
      const leftOf = g.box.x + g.box.w / 2 < g.x;   // grows away from its ships on hover
      rec.chip.style.left = leftOf ? 'auto' : `${g.box.x.toFixed(1)}px`;
      rec.chip.style.right = leftOf ? `${(width - g.box.x - g.box.w).toFixed(1)}px` : 'auto';
      rec.chip.style.top = `${g.box.y.toFixed(1)}px`;
      rec.chip.style.minWidth = `${g.box.w}px`;
      if (rec.text.textContent !== g.text) rec.text.textContent = g.text;
      if (rec.key !== g.key) {
        rec.key = g.key;
        rec.list.replaceChildren(...g.ids.map(id => {
          const row = document.createElement('button');
          row.type = 'button';
          row.className = 'group-row';
          row.dataset.shipId = id;
          row.innerHTML = '<span class="group-row-name"></span><span class="group-row-hull"><i></i></span>';
          row.querySelector('.group-row-name').textContent = this.contacts.get(id)?.short || id;
          row.addEventListener('click', () => this.selectShip(id));
          return row;
        }));
      }
      for (const row of rec.list.children) {
        const st = states?.[row.dataset.shipId];
        row.classList.toggle('current', row.dataset.shipId === this.hoverId);
        row.querySelector('i').style.width = `${Math.round(THREE.MathUtils.clamp(st?.hull ?? 100, 0, 100))}%`;
      }
      setLeader(rec.leader, g.leader, 0, 0);
    });
    for (let i = layout.groups.length; i < this.groupEls.length; i++) this.groupEls[i].el.hidden = true;

    layout.waves.forEach((w, i) => {
      const rec = this.waveEl(i);
      rec.el.hidden = !w.box;
      if (!w.box) return;
      rec.target = w.target;
      rec.el.className = `wave-contact ${waveUrgency(w)}`;
      rec.chip.style.left = `${w.box.x.toFixed(1)}px`;
      rec.chip.style.top = `${w.box.y.toFixed(1)}px`;
      rec.chip.style.width = `${w.box.w}px`;
      if (rec.chip.textContent !== w.text) rec.chip.textContent = w.text;
      rec.chip.title = `${w.count} torpedoes (${w.live} guided) inbound on ${this.contacts.get(w.target)?.info.name || w.target}`;
      setLeader(rec.leader, w.leader, 0, 0);
    });
    for (let i = layout.waves.length; i < this.waveEls.length; i++) this.waveEls[i].el.hidden = true;
  }

  clear() {
    this.container.replaceChildren();
    this.contacts.clear();
    this.groupEls = [];
    this.waveEls = [];
  }
}
