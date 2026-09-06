import * as THREE from 'three';
import { getShipClassName } from './shipSilhouettes.js';

const overlaps = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

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

/** Screen-space contact brackets and labels for ships too distant to read by eye. */
export class ContactOverlay {
  constructor(container, selectShip) {
    this.container = container;
    this.selectShip = selectShip;
    this.contacts = new Map();
    this.enabled = true;
    this.point = new THREE.Vector3();
    this.view = new THREE.Vector3();
    this.panels = ['battleHeader', 'shipRegistry', 'shipTelemetry', 'decisionPanel', 'eventLog', 'liveHud', 'viewControls', 'playbackControls']
      .map(id => document.getElementById(id)).filter(Boolean);
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
      button.innerHTML = '<span class="contact-bracket"></span><span class="contact-leader"></span><span class="contact-label"><span class="contact-name"></span><span class="contact-detail"></span><span class="contact-health"><i></i></span></span>';
      button.querySelector('.contact-name').textContent = info.name || id;
      button.addEventListener('click', () => this.selectShip(id));
      this.container.appendChild(button);
      this.contacts.set(id, {
        button, info, label: button.querySelector('.contact-label'),
        detail: button.querySelector('.contact-detail'), leader: button.querySelector('.contact-leader'),
        health: button.querySelector('.contact-health i')
      });
    }
  }

  update(scene, states, selectedId) {
    this.container.hidden = !this.enabled;
    if (!this.enabled) return;
    const width = window.innerWidth, height = window.innerHeight;
    const camera = scene.camera;
    camera.updateMatrixWorld();
    const occupied = this.panels.filter(panel => getComputedStyle(panel).visibility !== 'hidden').map(panel => {
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
      candidates.push({ id, contact, state, x, y });
    }
    // Reserve every bracket before placing labels; a label must never hide
    // another ship. Selected contacts get first choice of the remaining space.
    occupied.push(...candidates.map(({ x, y }) => ({ x: x - 10, y: y - 10, w: 20, h: 20 })));
    candidates.sort((a, b) => Number(b.id === selectedId) - Number(a.id === selectedId));
    for (const { id, contact, state, x, y } of candidates) {
      const { button, info, label, detail, leader, health } = contact;
      button.hidden = false;
      button.style.transform = `translate(${x - 10}px, ${y - 10}px)`;
      button.classList.toggle('selected', id === selectedId);
      button.classList.toggle('dying', !!state.dying);
      const hull = Math.round(THREE.MathUtils.clamp(state.hull ?? 100, 0, 100));
      detail.textContent = state.dying ? 'REACTOR UNSTABLE' : `${getShipClassName(info.type)} · ${hull}%`;
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
