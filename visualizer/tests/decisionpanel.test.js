import test from 'node:test';
import assert from 'node:assert/strict';
import { FakeDocument, build, allElements } from './fakedom.js';
import { BattleLoader } from '../src/BattleLoader.js';

// DecisionPanel reaches for the global document (as in the browser)
const doc = new FakeDocument();
globalThis.document = doc;
const { DecisionPanel } = await import('../src/DecisionPanel.js');

function panelFixture({ live = null, events = null } = {}) {
  doc.body.replaceChildren(
    build(doc, ['button', { id: 'decisionsBtn' }]),
    build(doc, ['aside', { id: 'decisionPanel', class: 'hidden' },
      ['button', { 'data-close': '' }],
      ['select', { 'data-ship': '' }],
      ['button', { 'data-prev': '' }], ['select', { 'data-decision': '' }], ['button', { 'data-next': '' }],
      ['button', { 'data-follow': '' }],
      ['div', { 'data-body': '' }]]));
  const loader = new BattleLoader();
  const image = 'data:image/png;base64,' + 'A'.repeat(4000);
  loader.parse({
    alpha_fleet: { ships: [{ ship_id: 'alpha_1', ship_name: 'TIS Dart-1', ship_type: 'corvette' }] },
    beta_fleet: { ships: [{ ship_id: 'beta_1', ship_name: 'OCS Falchion-1', ship_type: 'destroyer' }] },
    assets: { img: { role: 'user', content: [{ type: 'text', text: 'ADMIRAL CHECKPOINT 2' }, { type: 'image_url', image_url: { url: image } }] } },
    sim_trace: [{ t: 0, ships: {} }, { t: 120, ships: {} }],
    events: events || [
      { event_id: 'e1', sequence: 1, timestamp: 30, event_type: 'captain_decision', ship_id: 'alpha_1', data: { maneuver_type: 'HEADING', commands: [] } },
      { event_id: 'e2', sequence: 2, timestamp: 60, event_type: 'model_call', ship_id: null, data: { actor: 'admiral', faction: 'alpha', model: 'm', message_refs: ['img'] } },
      { event_id: 'e3', sequence: 3, timestamp: 60, event_type: 'captain_decision', ship_id: 'alpha_1', data: { maneuver_type: 'INTERCEPT', commands: [], parent_event_ids: ['e2'] } },
      { event_id: 'e4', sequence: 4, timestamp: 60, event_type: 'captain_report', ship_id: 'alpha_1', data: { status: 'DEVIATE', reason: 'Lancer is closer.', checkpoint: 2 } },
      { event_id: 'e5', sequence: 5, timestamp: 90, event_type: 'captain_decision', ship_id: 'alpha_1', data: { maneuver_type: 'EVADE', commands: [] } }
    ]
  });
  const tc = { currentTime: 0, pause() {}, seek(t) { this.currentTime = t; }, play() {} };
  const app = { loader, live, timeController: tc, selectedShipId: 'alpha_1', setLiveFollow() {}, updatePlayPauseButton() {}, selectShip() {} };
  const panel = new DecisionPanel(app);
  panel.reset();
  panel.setOpen(true);
  return { panel, app, tc, image };
}

const selected = panel => panel.decisionSelect.value;

test('the decision panel follows the latest decision as the playhead advances', () => {
  const { panel } = panelFixture();
  panel.update(35);
  assert.equal(selected(panel), 'e1');
  panel.update(65);
  assert.equal(selected(panel), 'e3', 'a followed panel must not freeze on its first decision (LV-01)');
  panel.update(95);
  assert.equal(selected(panel), 'e5');
  assert.match(panel.followBtn.textContent, /LATEST/);
});

test('picking a decision pins it until LATEST is re-armed', () => {
  const { panel, tc } = panelFixture();
  panel.update(95);
  panel.choose('e1');
  assert.equal(tc.currentTime, 30);
  panel.update(100);
  assert.equal(selected(panel), 'e1');
  assert.match(panel.followBtn.textContent, /PINNED 00:30/);
  panel.setFollow(true);
  panel.update(101);
  assert.equal(selected(panel), 'e5');
});

test('live follow forces the latest decision; before any orders the panel says it is waiting', () => {
  const live = { started: true, ended: false, follow: true, checkpoint: 1, status: 'paused' };
  const { panel } = panelFixture({ live });
  panel.followLatest = false;
  panel.update(20);
  assert.match(panel.body.textContent, /Awaiting first orders: commanders deciding \(CP1\)/);
  panel.update(62);
  assert.equal(selected(panel), 'e3');
});

test('admiral vision renders as a thumbnail and the raw JSON carries no base64', () => {
  const { panel, image } = panelFixture();
  panel.update(65);
  const imgs = allElements(panel.body).filter(el => el.tagName === 'IMG');
  assert.equal(imgs.length, 1);
  assert.equal(imgs[0].src, image);
  const pres = allElements(panel.body).filter(el => el.tagName === 'PRE').map(el => el.textContent).join('');
  assert.ok(pres.includes('shown above'));
  assert.ok(!pres.includes('AAAA'), 'no data URL inside the raw JSON');
  assert.match(panel.body.textContent, /report to admiral · DEVIATE/);
  assert.match(panel.body.textContent, /Lancer is closer\./);
});
