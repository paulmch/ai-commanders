/**
 * FittingsIndex - what each ship can still do, as a pure function of the
 * recording at time t (pass-5 battle damage, docs/pass-5-recording-additions.md).
 *
 * A ship's external fittings are its coilgun mounts, torpedo tubes, PD
 * turrets and drive. Their layout comes from the recording's fleet specs
 * (`weapons` per ship: slot -> type); their state from the events, folded in
 * order:
 *
 *   mount_damaged       absolute health of one fitting (slot = fleet-spec key),
 *                       drive thrust fraction, torpedo rounds left
 *   module_destroyed    the older path: a penetrating hit wrecks a weapon or
 *                       engine MODULE and the engine silences the matching
 *                       slot without a mount_damaged (mirror of
 *                       CombatSimulation._disable_weapon_for_module)
 *   magazine_cookoff    rounds burned, rounds left
 *   torpedo_launched    one round less
 *   blast_damage / reactor_blast   who a reactor blast caught, and what it cost
 *
 * The fold keeps, per ship, a short time-sorted list of immutable state
 * summaries; fittingsAt() binary-searches it, so a HUD surface can cache on
 * the returned object's identity and redraw only when something changed.
 * It also derives the moments a spectator must not miss ("notices": a ship
 * disarmed, adrift, a drive hit, a cook-off, a ship caught in a blast) and a
 * per-death blast summary for the kill feed.
 *
 * Everything is gated on `active`: the recording has fleet weapon specs AND
 * at least one pass-5 damage event. Older recordings (which do carry
 * module_destroyed) keep the HUD exactly as it was.
 *
 * Plain data only; no DOM, no imports from HudIndex (it imports this).
 */
import { SHIP_MODULES } from './shipModules.js';

/**
 * Torpedo rounds per tube, by hull (data/fleet_ships.json magazines). Only
 * used for the "rounds left" readout and the EMPTY state; every recorded
 * `torpedoes_remaining` re-syncs the count, and an unknown hull simply has
 * no count (null) until an event states one.
 */
export const TUBE_MAGAZINE = { corvette: 8, cruiser_torpedo: 12 };

const PASS5_EVENTS = new Set(['mount_damaged', 'reactor_blast', 'blast_damage', 'magazine_cookoff']);

/** Explicit module -> slot map of the engine (destroyer layout), tried first. */
const MODULE_SLOT = {
  'Spinal Coiler Mount': 'weapon_0',
  'Dorsal Turret Mount': 'weapon_1',
  'PD Laser Dorsal': 'pd_0',
  'PD Laser Ventral': 'pd_1'
};

const isPdModule = name => {
  const l = String(name).toLowerCase();
  return l.includes('point defense') || l.startsWith('pd') || ` ${l} `.includes(' pd ');
};

const kindOfWeapon = (slot, type) => {
  const s = `${slot} ${type || ''}`.toLowerCase();
  if (slot.startsWith('pd') || s.includes('pd_laser') || s.includes('point_defense')) return 'pd';
  if (s.includes('torpedo') || s.includes('launcher')) return 'launcher';
  return 'gun';
};

const slotOrder = (a, b) => (parseInt(a.split('_').pop(), 10) || 0) - (parseInt(b.split('_').pop(), 10) || 0);

/**
 * Fitting layout per ship from the recording's fleet specs:
 * {shipId: {type, faction, guns: [slot], tubes: [slot], pd: [slot]}}. Fleet battles
 * list ships under alpha_fleet / beta_fleet; a 1v1 has alpha_specs / beta_specs.
 */
export function fittingLayouts(sourceData = {}, ships = {}) {
  const out = {};
  const put = (id, weapons, type) => {
    if (!id || !weapons || typeof weapons !== 'object') return;
    const lay = { type: type || ships[id]?.type || '', guns: [], tubes: [], pd: [],
      faction: ships[id]?.faction || (String(id).startsWith('alpha') ? 'alpha' : String(id).startsWith('beta') ? 'beta' : null) };
    for (const [slot, w] of Object.entries(weapons)) {
      const kind = kindOfWeapon(slot, w?.type);
      (kind === 'pd' ? lay.pd : kind === 'launcher' ? lay.tubes : lay.guns).push(slot);
    }
    for (const k of ['guns', 'tubes', 'pd']) lay[k].sort(slotOrder);
    if (lay.guns.length + lay.tubes.length + lay.pd.length) out[id] = lay;
  };
  for (const fleet of [sourceData?.alpha_fleet, sourceData?.beta_fleet]) {
    for (const s of fleet?.ships || []) put(s.ship_id, s.weapons, s.ship_type);
  }
  if (!Object.keys(out).length) {
    put('alpha', sourceData?.alpha_specs?.weapons, sourceData?.alpha_specs?.ship_type);
    put('beta', sourceData?.beta_specs?.weapons, sourceData?.beta_specs?.ship_type);
  }
  return out;
}

/**
 * Slots a destroyed module silences on a hull of `type` with layout `lay`:
 * {slots: [slot], engine: bool}. Same rule as the engine: the explicit map,
 * then the Nth gun module -> Nth non-PD slot and PD modules by ordinal (a
 * module named like a whole PD array takes every turret). Unknown hull
 * types and non-weapon modules map to nothing.
 */
export function slotsForModule(type, lay, moduleName) {
  const none = { slots: [], engine: false };
  if (!lay || !moduleName) return none;
  const modules = SHIP_MODULES[String(type || '').toLowerCase()];
  const mod = modules?.find(m => m.name === moduleName);
  if (mod?.type === 'engine' || (!modules && /engine assembly/i.test(moduleName))) return { slots: [], engine: true };
  const nonPd = [...lay.guns, ...lay.tubes].sort(slotOrder);
  const has = slot => nonPd.includes(slot) || lay.pd.includes(slot);
  const explicit = MODULE_SLOT[moduleName];
  if (explicit) return { slots: has(explicit) ? [explicit] : [], engine: false };
  if (!modules || mod?.type !== 'weapon') return none;
  const weaponModules = modules.filter(m => m.type === 'weapon').map(m => m.name);
  const pdNames = weaponModules.filter(isPdModule);
  const gunNames = weaponModules.filter(n => !isPdModule(n));
  if (gunNames.includes(moduleName)) {
    const slot = nonPd[gunNames.indexOf(moduleName)];
    return { slots: slot ? [slot] : [], engine: false };
  }
  if (pdNames.includes(moduleName)) {
    if (moduleName.toLowerCase().includes('array')) return { slots: [...lay.pd], engine: false };
    const slot = lay.pd[pdNames.indexOf(moduleName)];
    return { slots: slot ? [slot] : [], engine: false };
  }
  return none;
}

const upper = (rows, t) => {
  let lo = 0, hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].t <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
};

const count = (arr, s) => arr.reduce((n, x) => n + (x === s ? 1 : 0), 0);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : (word.endsWith('o') ? 'es' : 's')}`;
const KIND_WORD = { gun: 'gun', launcher: 'tube', pd: 'PD' };

/** "2 PD lost, 1 gun damaged" from blast mount hits [{kind, destroyed}]. */
function mountsText(mounts) {
  const parts = [];
  for (const destroyed of [true, false]) {
    for (const kind of ['gun', 'launcher', 'pd', 'engine']) {
      const n = mounts.filter(m => m.kind === kind && !!m.destroyed === destroyed).length;
      if (!n) continue;
      if (kind === 'engine') parts.push(destroyed ? 'drive destroyed' : 'drive damaged');
      else parts.push(`${n} ${KIND_WORD[kind]}${kind !== 'pd' && n > 1 ? 's' : ''} ${destroyed ? 'lost' : 'damaged'}`);
    }
  }
  return parts.join(', ');
}

/**
 * The fold. add() takes events[from..] in order; state survives between
 * calls, so one pass or any number of in-order appends end the same.
 */
export class FittingFold {
  constructor() {
    this.active = false;
    this.layouts = {};
    this.live = new Map();     // shipId -> mutable current {fit: Map slot -> state, thrust, rounds}
    this.states = {};          // shipId -> [{t, ...summary}] (immutable entries, time-sorted)
    this.notices = [];         // every derived notice, time-sorted (filtered against kills in view())
    this.blasts = {};          // dead shipId -> blast summary
    this.pending = new Map();  // detonating shipId -> Map victimId -> partial victim record
  }

  /** (Re)read the layouts; called until the recording's fleet specs are there. */
  setLayouts(sourceData, ships) {
    if (Object.keys(this.layouts).length) return;
    this.layouts = fittingLayouts(sourceData, ships);
    for (const [id, lay] of Object.entries(this.layouts)) {
      const mag = TUBE_MAGAZINE[String(lay.type).toLowerCase()];
      const cur = { fit: new Map(), thrust: 1, rounds: lay.tubes.length && mag ? mag * lay.tubes.length : null };
      this.live.set(id, cur);
      this.states[id] = [this._summary(id, -Infinity)];
    }
  }

  _summary(id, t) {
    const lay = this.layouts[id], cur = this.live.get(id);
    const st = slot => cur.fit.get(slot) || 'ok';
    const guns = lay.guns.map(st), tubes = lay.tubes.map(st), pd = lay.pd.map(st);
    const gunsUp = guns.length - count(guns, 'destroyed');
    const tubesUp = tubes.length - count(tubes, 'destroyed');
    const dry = tubes.length > 0 && tubesUp > 0 && cur.rounds === 0;
    const hadWeapons = guns.length + tubes.length > 0;
    return {
      t, guns, tubes, pd, thrust: cur.thrust, rounds: cur.rounds,
      gunsUp, tubesUp, pdUp: pd.length - count(pd, 'destroyed'),
      // Able to deal damage: a working gun, or a working tube that is not known to be empty
      armed: gunsUp > 0 || (tubesUp > 0 && !dry),
      disarmed: hadWeapons && gunsUp === 0 && tubesUp === 0,
      dry: hadWeapons && gunsUp === 0 && dry,
      adrift: !(cur.thrust > 0)
    };
  }

  /** Record the ship's state after a change at t (several changes in one tick collapse into one entry). */
  _commit(id, t) {
    const list = this.states[id];
    const next = this._summary(id, t);
    if (list[list.length - 1].t === t) list[list.length - 1] = next; else list.push(next);
    return next;
  }

  _notice(e, id, kind, ship, ships, status, detail, extra = '') {
    const info = ships[ship] || {};
    this.notices.push({
      id: `${id}:${kind}${extra}`, t: e.timestamp, kind, shipId: ship, name: info.name || ship,
      faction: info.faction || (String(ship).startsWith('alpha') ? 'alpha' : String(ship).startsWith('beta') ? 'beta' : null),
      status, detail
    });
  }

  add(events, ids, from, ships = {}) {
    const name = id => (id && (ships[id]?.name || id)) || '';
    for (let i = from; i < events.length; i++) {
      const e = events[i];
      const d = e.data || {};
      const type = e.event_type;
      if (PASS5_EVENTS.has(type)) this.active = true;
      const ship = e.ship_id;
      const cur = ship ? this.live.get(ship) : null;
      const lay = ship ? this.layouts[ship] : null;
      const before = cur ? this.states[ship][this.states[ship].length - 1] : null;
      let changed = false;
      let what = '';

      if (type === 'mount_damaged') {
        if (d.cause === 'reactor_blast' && d.source_ship_id) {
          this._victim(d.source_ship_id, ship).mounts.push({ kind: d.kind, destroyed: !!d.destroyed });
        }
        if (cur) {
          if (d.kind === 'engine') {
            cur.thrust = d.destroyed ? 0 : Math.max(0, Math.min(1, Number(d.thrust_fraction) || 0));
            changed = true;
          } else if (d.slot) {
            const hp = Number(d.health_pct);
            cur.fit.set(d.slot, d.destroyed ? 'destroyed' : hp < 99.5 ? 'damaged' : 'ok');
            if (d.kind === 'launcher' && Number.isFinite(d.torpedoes_remaining)) cur.rounds = d.torpedoes_remaining;
            changed = true;
            what = d.kind === 'launcher' ? 'torpedo tube' : d.kind === 'pd' ? 'PD turret' : 'gun';
          }
        }
      } else if (type === 'module_destroyed') {
        if (cur) {
          const hit = slotsForModule(lay.type, lay, d.module_name);
          if (hit.engine && cur.thrust > 0) { cur.thrust = 0; changed = true; }
          for (const slot of hit.slots) {
            if (cur.fit.get(slot) !== 'destroyed') { cur.fit.set(slot, 'destroyed'); changed = true; what = lay.pd.includes(slot) ? 'PD turret' : lay.tubes.includes(slot) ? 'torpedo tube' : 'gun'; }
          }
        }
      } else if (type === 'magazine_cookoff') {
        if (cur && lay.tubes.length && Number.isFinite(d.torpedoes_remaining)) { cur.rounds = d.torpedoes_remaining; changed = true; }
        if (ship) {
          const burned = d.torpedoes_lost > 0 ? plural(d.torpedoes_lost, 'torpedo') : `${d.gun_rounds_lost || 0} rounds`;
          const wrecked = d.modules_destroyed?.length ? ` · ${plural(d.modules_destroyed.length, 'compartment')} wrecked` : '';
          this._notice(e, ids[i], 'cookoff', ship, ships, 'MAGAZINE COOK-OFF', `${burned} burned${wrecked}`);
        }
      } else if (type === 'torpedo_launched') {
        if (cur && cur.rounds != null && cur.rounds > 0) { cur.rounds--; changed = true; }
      } else if (type === 'blast_damage') {
        if (d.source_ship_id && ship) {
          Object.assign(this._victim(d.source_ship_id, ship), {
            distanceKm: Number(d.distance_km) || 0,
            armorCm: Number(d.armor_ablation_cm) || 0,
            location: d.hit_location || null,
            heatGj: Number(d.heat_gj) || 0,
            radiators: (d.radiators || []).length,
            fuseCutS: Number.isFinite(d.fuse_cut_s) ? d.fuse_cut_s : null
          });
        }
      } else if (type === 'reactor_blast') {
        if (ship) this._blast(e, ids[i], ship, ships);
      }

      if (!changed || !cur) continue;
      const after = this._commit(ship, e.timestamp);
      const by = d.source_ship_id ? ` by ${name(d.source_ship_id)}` : '';
      const blast = d.cause === 'reactor_blast' ? ` in ${name(d.source_ship_id)}'s reactor blast` : by;
      if (before.thrust > 0 && after.adrift) {
        this._notice(e, ids[i], 'adrift', ship, ships, 'ADRIFT', `drive destroyed${blast}`);
      } else if (type === 'mount_damaged' && d.kind === 'engine' && after.thrust < before.thrust) {
        this._notice(e, ids[i], 'drive', ship, ships, `DRIVE HIT · ${Math.round(after.thrust * 100)}% THRUST`, `drive damaged${blast}`);
      }
      if (!before.disarmed && after.disarmed) {
        this._notice(e, ids[i], 'disarmed', ship, ships, 'DISARMED', `last ${what || 'weapon'} knocked out${blast}`);
      }
    }
  }

  _victim(source, victim) {
    if (!this.pending.has(source)) this.pending.set(source, new Map());
    const m = this.pending.get(source);
    if (!m.has(victim)) m.set(victim, { shipId: victim, distanceKm: null, armorCm: 0, location: null, heatGj: 0, radiators: 0, fuseCutS: null, mounts: [] });
    return m.get(victim);
  }

  _blast(e, id, ship, ships) {
    const d = e.data || {};
    const name = x => (x && (ships[x]?.name || x)) || '';
    const pend = this.pending.get(ship) || new Map();
    this.pending.delete(ship);
    // The summary's ships_hit is the roster; details come from the blast_damage / mount_damaged before it
    const hit = Array.isArray(d.ships_hit) ? d.ships_hit : [...pend.keys()];
    const victims = hit.map(v => ({ ...(pend.get(v) || { shipId: v, distanceKm: null, armorCm: 0, location: null, heatGj: 0, radiators: 0, fuseCutS: null, mounts: [] }), name: name(v) }));
    const blast = {
      t: e.timestamp, shipId: ship, radiusKm: Number(d.radius_km) || 0, victims,
      torpsKilled: (d.torpedoes_destroyed || []).length, torpsBlinded: (d.torpedoes_blinded || []).length
    };
    this.blasts[ship] = blast;
    for (const v of victims) {
      const cost = [];
      if (v.fuseCutS != null) cost.push(`fuse cut ${v.fuseCutS.toFixed(0)} s`);
      if (v.armorCm > 0) cost.push(`−${v.armorCm.toFixed(1)} cm ${v.location || ''} armour`.replace('  ', ' '));
      const mounts = mountsText(v.mounts);
      if (mounts) cost.push(mounts);
      if (v.radiators) cost.push(`${plural(v.radiators, 'radiator')} scorched`);
      const at = v.distanceKm != null ? `${v.distanceKm.toFixed(1)} km from ` : 'in ';
      this._notice(e, id, 'blast', v.shipId, ships, 'CAUGHT IN BLAST',
        `${at}${name(ship)}'s reactor${cost.length ? ': ' + cost.join(', ') : ''}`, `:${v.shipId}`);
    }
  }

  /**
   * The plain-data view for the hud: {fittings, notices, blasts}. Notices
   * about a ship that was already knocked out by then are dropped (the kill
   * feed tells that story); `kills` = the hud's kill list.
   */
  view(kills = []) {
    if (!this.active || !Object.keys(this.layouts).length) {
      return { fittings: { active: false, layouts: {}, states: {}, totals: emptyTotals() }, notices: [], blasts: {} };
    }
    const downAt = new Map(kills.map(k => [k.shipId, k.t]));
    const notices = this.notices.filter(n => !(downAt.has(n.shipId) && downAt.get(n.shipId) <= n.t + 1e-6));
    const totals = emptyTotals();
    for (const [id, lay] of Object.entries(this.layouts)) {
      const f = lay.faction;
      if (!totals[f]) continue;
      totals[f].guns += lay.guns.length; totals[f].tubes += lay.tubes.length; totals[f].pd += lay.pd.length;
    }
    return { fittings: { active: true, layouts: this.layouts, states: this.states, totals }, notices, blasts: this.blasts };
  }
}

const emptyTotals = () => ({ alpha: { guns: 0, tubes: 0, pd: 0 }, beta: { guns: 0, tubes: 0, pd: 0 } });

/** {fittings, notices, blasts} for a whole recording (tests, tools). */
export function buildFittings(events, ids, ships, sourceData, kills = []) {
  const fold = new FittingFold();
  fold.setLayouts(sourceData, ships);
  fold.add(events, ids, 0, ships);
  return fold.view(kills);
}

/**
 * A ship's fitting state at time t, or null when the recording carries no
 * pass-5 damage data (or no spec for this ship):
 * {t, guns: ['ok'|'damaged'|'destroyed'], tubes, pd, thrust 0..1, rounds|null,
 *  gunsUp, tubesUp, pdUp, armed, disarmed, dry, adrift}.
 * The same object is returned until the next change - cache on identity.
 */
export function fittingsAt(fittings, shipId, t) {
  if (!fittings?.active) return null;
  const list = fittings.states[shipId];
  if (!list) return null;
  const i = upper(list, t + 1e-6) - 1;
  return list[Math.max(0, i)];
}

/** One-line readout of a fitting state ("Guns 3/5 · PD 2/3 · Drive 53% · 38 torpedoes"). */
export function fittingsText(st) {
  if (!st) return '';
  const parts = [];
  const dmg = arr => { const n = count(arr, 'damaged'); return n ? ` (${n} damaged)` : ''; };
  if (st.guns.length) parts.push(`Guns ${st.gunsUp}/${st.guns.length}${dmg(st.guns)}`);
  if (st.tubes.length) parts.push(`Tubes ${st.tubesUp}/${st.tubes.length}${dmg(st.tubes)}${st.rounds != null ? `, ${plural(st.rounds, 'torpedo')}` : ''}`);
  if (st.pd.length) parts.push(`PD ${st.pdUp}/${st.pd.length}${dmg(st.pd)}`);
  parts.push(st.adrift ? 'Drive OUT' : `Drive ${Math.round(st.thrust * 100)}%`);
  return parts.join(' · ');
}

/** The flag a spectator should see first: DISARMED / EMPTY / ADRIFT (or '' when the ship is fine). */
export function fittingsFlag(st) {
  if (!st) return '';
  if (st.disarmed) return 'DISARMED';
  if (st.adrift) return 'ADRIFT';
  if (st.dry) return 'EMPTY';
  return '';
}

/** Blast summary line for a death ("BLAST 18 km · caught OCS Falchion-1 · 1 torpedo killed, 3 blinded"), '' when it touched nothing. */
export function blastLine(blast) {
  if (!blast) return '';
  const parts = [];
  if (blast.victims.length) {
    const names = blast.victims.map(v => v.name);
    parts.push(`caught ${names.length > 2 ? `${names.slice(0, 2).join(', ')} +${names.length - 2}` : names.join(', ')}`);
  }
  const torps = [];
  if (blast.torpsKilled) torps.push(`${plural(blast.torpsKilled, 'torpedo')} killed`);
  if (blast.torpsBlinded) torps.push(`${blast.torpsKilled ? blast.torpsBlinded : plural(blast.torpsBlinded, 'torpedo')} blinded`);
  if (torps.length) parts.push(torps.join(', '));
  if (!parts.length) return '';
  return `BLAST ${blast.radiusKm.toFixed(0)} km · ${parts.join(' · ')}`;
}
