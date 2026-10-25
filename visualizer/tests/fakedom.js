/**
 * Minimal DOM stand-in for node tests of the HUD modules: enough element
 * tree, classList, dataset, listeners and simple selectors to drive them,
 * and an innerHTML that records every write so tests can prove recorded
 * text never reaches the HTML parser.
 */
class FakeText {
  constructor(text) { this.nodeType = 3; this.data = String(text); this.parentNode = null; }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
}

class FakeClassList {
  constructor(el) { this.el = el; }
  _set() { return new Set(this.el._className.split(/\s+/).filter(Boolean)); }
  _write(s) { this.el._className = [...s].join(' '); }
  add(...c) { const s = this._set(); c.forEach(x => s.add(x)); this._write(s); }
  remove(...c) { const s = this._set(); c.forEach(x => s.delete(x)); this._write(s); }
  contains(c) { return this._set().has(c); }
  toggle(c, force) {
    const s = this._set();
    const on = force === undefined ? !s.has(c) : !!force;
    if (on) s.add(c); else s.delete(c);
    this._write(s);
    return on;
  }
}

export class FakeElement {
  constructor(tag, doc) {
    this.nodeType = 1;
    this.tagName = String(tag).toUpperCase();
    this.ownerDocument = doc;
    this.childNodes = [];
    this.parentNode = null;
    this._className = '';
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.listeners = {};
    this.hidden = false;
    this.title = '';
    this.scrollTop = 0;
    this.value = '';
    this.open = false;
    this.disabled = false;
    this.innerHTMLWrites = [];
  }
  get className() { return this._className; }
  set className(v) { this._className = String(v); }
  get classList() { return new FakeClassList(this); }
  get children() { return this.childNodes.filter(n => n.nodeType === 1); }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get childElementCount() { return this.children.length; }
  get textContent() { return this.childNodes.map(n => n.textContent).join(''); }
  set textContent(v) {
    this.childNodes.forEach(n => { n.parentNode = null; });
    this.childNodes = v === '' || v == null ? [] : [new FakeText(v)];
  }
  get innerHTML() { return this.innerHTMLWrites.at(-1) ?? ''; }
  set innerHTML(v) {
    this.innerHTMLWrites.push(String(v));
    this.ownerDocument?.innerHTMLWrites.push(String(v));
    this.childNodes = [];
  }
  _adopt(n) {
    if (typeof n === 'string') n = new FakeText(n);
    n.parentNode?.removeChild(n);
    n.parentNode = this;
    return n;
  }
  appendChild(n) { n = this._adopt(n); this.childNodes.push(n); return n; }
  append(...nodes) { nodes.forEach(n => this.appendChild(n)); }
  prepend(...nodes) { nodes.reverse().forEach(n => this.insertBefore(n, this.firstChild)); }
  insertBefore(n, ref) {
    n = this._adopt(n);
    const i = ref ? this.childNodes.indexOf(ref) : -1;
    if (i < 0) this.childNodes.push(n); else this.childNodes.splice(i, 0, n);
    return n;
  }
  removeChild(n) {
    const i = this.childNodes.indexOf(n);
    if (i >= 0) this.childNodes.splice(i, 1);
    n.parentNode = null;
    return n;
  }
  remove() { this.parentNode?.removeChild(this); }
  replaceChildren(...nodes) {
    this.childNodes.forEach(n => { n.parentNode = null; });
    this.childNodes = [];
    this.append(...nodes);
  }
  setAttribute(k, v) { this.attributes[k] = String(v); if (k === 'class') this._className = String(v); }
  getAttribute(k) { return this.attributes[k] ?? null; }
  removeAttribute(k) { delete this.attributes[k]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter(f => f !== fn); }
  dispatch(type, ev = {}) {
    const e = { type, target: this, stopPropagation() {}, preventDefault() {}, ...ev };
    (this.listeners[type] || []).forEach(fn => fn(e));
  }
  click() { this.dispatch('click'); }
  focus() {}
  scrollIntoView() {}
  getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
  get id() { return this.attributes.id ?? ''; }
  set id(v) { this.attributes.id = String(v); }
  get options() { return this.children.filter(c => c.tagName === 'OPTION'); }
  _matches(sel) {
    const m = /^([a-z0-9]*)((?:\.[\w-]+)*)(?:\[([\w-]+)(?:="([^"]*)")?\])?$/i.exec(sel.trim());
    if (!m) throw new Error(`fakedom: unsupported selector ${sel}`);
    const [, tag, classes, attr, val] = m;
    if (tag && this.tagName !== tag.toUpperCase()) return false;
    for (const c of classes.split('.').filter(Boolean)) if (!this.classList.contains(c)) return false;
    if (attr) {
      const key = attr.startsWith('data-') ? attr.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase()) : null;
      const have = key ? this.dataset[key] : this.attributes[attr];
      if (have === undefined) return false;
      if (val !== undefined && String(have) !== val) return false;
    }
    return true;
  }
  querySelectorAll(sel) {
    const out = [];
    const walk = el => el.children.forEach(c => { if (c._matches(sel)) out.push(c); walk(c); });
    walk(this);
    return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

export class FakeDocument {
  constructor() { this.innerHTMLWrites = []; this.body = new FakeElement('body', this); }
  createElement(tag) { return new FakeElement(tag, this); }
  createElementNS(_ns, tag) { return new FakeElement(tag, this); }
  createTextNode(text) { return new FakeText(text); }
  getElementById(id) { return [this.body, ...allElements(this.body)].find(el => el.id === id) || null; }
  addEventListener() {}
  removeEventListener() {}
}

/** Build a fake element tree from a compact spec: ['tag', {attrs, data}, ...children]. */
export function build(doc, spec) {
  const [tag, attrs = {}, ...children] = spec;
  const el = doc.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else if (k.startsWith('data-')) el.dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = v;
    else el.setAttribute(k, v);
  }
  for (const c of children) el.append(typeof c === 'string' ? c : build(doc, c));
  return el;
}

/** Every element in a subtree (depth first). */
export function allElements(root) {
  const out = [];
  const walk = el => el.children.forEach(c => { out.push(c); walk(c); });
  walk(root);
  return out;
}
