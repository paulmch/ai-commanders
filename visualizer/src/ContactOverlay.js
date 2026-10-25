import * as THREE from 'three';
import { getShipClassName } from './shipSilhouettes.js';
import { detonationTime, lastHitAt, reactorPulseRate, HIT_FLASH_S } from './DamageTimeline.js';

/** Inbound torpedoes closer than this (s) pulse the bracket red. */
export const THREAT_IMMINENT_S = 10;

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

const overlaps = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

/**
 * Screen box of a contact's threat badge (.contact-threat: 9px mono at
 * left 21px / top -12px of the 20 px bracket centred on x, y). Reserved
 * before labels are placed so no label covers it, the ship's own included.
 */
export function threatBadgeBox(x, y, text) {
  return { x: x + 10, y: y - 23, w: 10 + 6 * String(text || '').length, h: 15 };
}

/** HUD panels contact brackets and labels keep clear of (looked up lazily: some mount after the overlay). */
export const CONTACT_PANEL_IDS = ['battleHeader', 'shipRegistry', 'shipTelemetry', 'decisionPanel', 'eventLog', 'liveHud',
  'viewControls', 'playbackControls', 'trackHud', 'commsFeed', 'killFeed', 'resultBanner',   // spectator HUD (pass 4)
  'directorChip', 'liveDeliberation', 'liveConnection'];                                      // live wire (pass 4)

/** Choose a clear label position while leaving the contact at its true location. */
export function placeContactLabel(x, y, side, occupied, width, height) {
  const w = 126, h = 34;
  for (const direction of [side, -side]) {
    for (const dy of [-17, -57, 23, -97, 63, -137, 103]) {
      const box = { x: direction < 0 ? x - w - 18 : x + 18, y: y + dy, w, h };
      if (box.x < 8 || box.y < 80 || box.x + w > width - 8 || box.y + h > height - 90) continue;
      if (!occupied.some(other => overlaps(box, other))) return box;
    }
  }
  return null;
}

/**
 * Screen-space contact brackets and labels for ships too distant to read by
 * eye. At fleet range each bracket also carries the ship's combat state:
 * an incoming-torpedo badge ('3 TORP · 6s', the bracket pulsing red inside
 * THREAT_IMMINENT_S), a flash on every hit, and on a dying hulk the
 * detonation countdown with a pulse that quickens toward zero. The C toggle
 * also hides the engagement lines (render/TargetingLines.js).
 */
export class ContactOverlay {
  constructor(container, selectShip) {
    this.container = container;
    this.selectShip = selectShip;
    this.contacts = new Map();
    this.threats = new Map();
    this.enabled = true;
    this.point = new THREE.Vector3();
    this.view = new THREE.Vector3();
    this.panels = [];
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
    for (const [id, info] of Object.entries(ships)) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `ship-contact ${info.faction}`;
      button.dataset.shipId = id;
      button.setAttribute('aria-label', `Inspect ${info.name || id}`);
      button.innerHTML = '<span class="contact-bracket"></span><span class="contact-threat" hidden></span><span class="contact-leader"></span><span class="contact-label"><span class="contact-name"></span><span class="contact-detail"></span><span class="contact-health"><i></i></span></span>';
      button.querySelector('.contact-name').textContent = info.name || id;
      button.addEventListener('click', () => this.selectShip(id));
      this.container.appendChild(button);
      this.contacts.set(id, {
        button, info, label: button.querySelector('.contact-label'),
        detail: button.querySelector('.contact-detail'), leader: button.querySelector('.contact-leader'),
        health: button.querySelector('.contact-health i'), bracket: button.querySelector('.contact-bracket'),
        threat: button.querySelector('.contact-threat')
      });
    }
  }

  update(scene, states, selectedId) {
    this.container.hidden = !this.enabled;
    if (scene.weapons?.engagements) scene.weapons.engagements.enabled = this.enabled;
    if (!this.enabled) return;
    const now = scene.currentTime || 0;
    const threats = scene.weapons?.timeline?.threatsAt ? scene.weapons.timeline.threatsAt(now, this.threats) : null;
    const damage = scene.damageFX?.timeline || null;
    const width = window.innerWidth, height = window.innerHeight;
    const camera = scene.camera;
    camera.updateMatrixWorld();
    const occupied = this.panelElements().filter(panel => !panel.hidden && getComputedStyle(panel).visibility !== 'hidden').map(panel => {
      const r = panel.getBoundingClientRect();
      return { x: r.left - 6, y: r.top - 6, w: r.width + 12, h: r.height + 12 };
    }).filter(r => r.w > 12 && r.h > 12);
    const candidates = [];
    for (const [id, contact] of this.contacts) {
      const { button } = contact;
      const ship = scene.ships.get(id), state = states?.[id];
      button.hidden = true;
      if (!ship?.visible || !state || state.destroyed) continue;
      this.view.copy(ship.position).applyMatrix4(camera.matrixWorldInverse);
      const depth = -this.view.z;
      const pixels = ship.userData.size.length * height / (2 * depth * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
      if (depth <= camera.near || pixels > 90) continue;
      this.point.copy(ship.position).project(camera);
      if (Math.abs(this.point.x) > 0.98 || Math.abs(this.point.y) > 0.96 || this.point.z > 1) continue;
      const x = (this.point.x + 1) * width / 2, y = (1 - this.point.y) * height / 2;
      if (occupied.some(r => overlaps({ x: x - 8, y: y - 8, w: 16, h: 16 }, r))) continue;
      // Inbound torpedoes: count and soonest ETA by the bracket
      const th = state.dying ? null : threats?.get(id);
      candidates.push({ id, contact, state, x, y, th, threatText: threatLabel(th) });
    }
    // Reserve every bracket and threat badge before placing labels; a label
    // must never hide another ship or a badge. Selected contacts get first
    // choice of the remaining space.
    for (const { x, y, threatText } of candidates) {
      occupied.push({ x: x - 10, y: y - 10, w: 20, h: 20 });
      if (threatText) occupied.push(threatBadgeBox(x, y, threatText));
    }
    candidates.sort((a, b) => Number(b.id === selectedId) - Number(a.id === selectedId));
    for (const { id, contact, state, x, y, th, threatText: text } of candidates) {
      const { button, info, label, detail, leader, health, bracket, threat } = contact;
      button.hidden = false;
      button.style.transform = `translate(${x - 10}px, ${y - 10}px)`;
      button.classList.toggle('selected', id === selectedId);
      button.classList.toggle('dying', !!state.dying);
      const hull = Math.round(THREE.MathUtils.clamp(state.hull ?? 100, 0, 100));
      const sd = damage?.ships.get(id);
      let ttd = NaN;
      if (state.dying) {
        const det = detonationTime(sd);
        ttd = det !== null ? det - now : NaN;
        // the bracket pulse quickens with the reactor (DamageTimeline.reactorPulseRate)
        button.style.setProperty('--pulse', `${(0.5 / reactorPulseRate(Number.isFinite(ttd) ? ttd : 30)).toFixed(3)}s`);
      }
      detail.textContent = state.dying ? detonationLabel(ttd) : `${getShipClassName(info.type)} · ${hull}%`;
      threat.hidden = !text;
      if (text) threat.textContent = text;
      button.classList.toggle('threatened', !!text);
      button.classList.toggle('imminent', !!text && th.eta < THREAT_IMMINENT_S);
      // A hit flashes the bracket white for HIT_FLASH_S
      const flash = hitFlash(now - lastHitAt(sd, now));
      bracket.style.boxShadow = flash > 0 ? `0 0 ${(3 + 9 * flash).toFixed(1)}px rgba(255,244,228,${(0.35 + 0.65 * flash).toFixed(2)})` : '';
      bracket.style.borderColor = flash > 0 ? `rgba(255,250,240,${(0.45 + 0.55 * flash).toFixed(2)})` : '';
      button.title = `${info.name || id} · ${getShipClassName(info.type)} · hull ${hull}%`;
      health.style.width = `${hull}%`;
      const box = placeContactLabel(x, y, info.faction === 'alpha' ? -1 : 1, occupied, width, height);
      label.hidden = leader.hidden = !box;
      if (!box) continue;
      occupied.push(box);
      label.style.left = `${box.x - x + 10}px`;
      label.style.top = `${box.y - y + 10}px`;
      const dx = (box.x < x ? box.x + box.w : box.x) - x;
      const dy = box.y + box.h / 2 - y;
      leader.style.width = `${Math.hypot(dx, dy)}px`;
      leader.style.transform = `rotate(${Math.atan2(dy, dx)}rad)`;
    }
  }

  clear() {
    this.container.replaceChildren();
    this.contacts.clear();
  }
}
