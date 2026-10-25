/**
 * Tiny DOM helpers shared by the HUD modules. Every recorded string (LLM
 * orders, captain notes, enemy transmissions, ship names) goes through
 * textContent - never innerHTML - so model-written text cannot inject markup.
 */
export const defaultDoc = () => (typeof document !== 'undefined' ? document : null);

/** <tag class=cls>text</tag> built with textContent. */
export function node(doc, tag, text, cls) {
  const el = doc.createElement(tag);
  if (cls) el.className = cls;
  if (text != null && text !== '') el.textContent = String(text);
  return el;
}

/** Set textContent only when it changed (cheap per-frame updates). */
export function setText(el, text) {
  const s = String(text ?? '');
  if (el && el.textContent !== s) el.textContent = s;
}

/**
 * Keep a list element showing `desired` row ids (oldest -> newest, newest at
 * the top of the list). When the change is "the window slid forward" (new
 * rows at the head, old rows dropped from the tail) only those nodes are
 * touched; anything else (seek back, filter change, re-sorted live data)
 * rebuilds the list once. Returns 'same' | 'append' | 'rebuild'.
 *
 * @param {Element} list
 * @param {string[]} shown - ids currently in the list, oldest first (mutated)
 * @param {Object[]} desired - rows with .id, oldest first
 * @param {Function} make - row -> Element
 */
export function syncNewestFirst(list, shown, desired, make) {
  const same = shown.length === desired.length && shown.every((id, i) => id === desired[i].id);
  if (same) return 'same';
  const last = shown.length ? shown[shown.length - 1] : null;
  const q = last == null ? -1 : desired.findIndex(r => r.id === last);
  if (q >= 0) {
    const drop = shown.length - (q + 1);
    let prefix = drop >= 0;
    for (let i = 0; prefix && i <= q; i++) prefix = shown[drop + i] === desired[i].id;
    if (prefix) {
      for (let i = 0; i < drop; i++) list.removeChild(list.lastChild);
      shown.splice(0, drop);
      for (let i = q + 1; i < desired.length; i++) {
        list.insertBefore(make(desired[i], true), list.firstChild);
        shown.push(desired[i].id);
      }
      return 'append';
    }
  }
  const nodes = [];
  for (let i = desired.length - 1; i >= 0; i--) nodes.push(make(desired[i], false));
  list.replaceChildren(...nodes);
  shown.length = 0;
  for (const r of desired) shown.push(r.id);
  return 'rebuild';
}

let lightbox = null;

/** Full-size image overlay (admiral tactical plots); click or Escape closes it. */
export function showLightbox(doc, src, caption = '') {
  if (!doc?.body || !src) return null;
  if (!lightbox) {
    const root = node(doc, 'div', null, 'hud-lightbox hidden');
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    const img = doc.createElement('img');
    const cap = node(doc, 'p', '', 'hud-lightbox-caption');
    root.append(img, cap);
    const close = () => {
      root.classList.add('hidden');
      img.removeAttribute('src');
      doc.removeEventListener?.('keydown', onKey, true);
    };
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation?.(); e.preventDefault?.(); close(); } };
    root.addEventListener('click', close);
    doc.body.append(root);
    lightbox = { root, img, cap, close, onKey };
  }
  lightbox.img.src = src;
  lightbox.img.alt = caption;
  lightbox.cap.textContent = caption;
  lightbox.root.classList.remove('hidden');
  doc.addEventListener?.('keydown', lightbox.onKey, true);
  return lightbox.root;
}
