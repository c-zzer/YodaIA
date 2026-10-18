// Exécute dans le navigateur le mini-GPT entraîné avec mini_gpt.py (mêmes calculs, avec cache clés/valeurs).
class MiniGPTJS {
  constructor(meta, buffer) {
    this.m = meta; this.d = meta.dim; this.H = meta.tetes; this.L = meta.couches; this.bloc = meta.bloc; this.chars = meta.chars;
    this.vocab = meta.chars.length; this.w = {};
    for (const t of meta.tensors) this.w[t.name] = new Float32Array(buffer, t.offset, t.shape.reduce((a, b) => a * b, 1));
    this.K = Array.from({ length: this.L }, () => new Float32Array(this.bloc * this.d));
    this.V = Array.from({ length: this.L }, () => new Float32Array(this.bloc * this.d));
    this.n = 0; this.a2i = new Map(meta.chars.map((c, i) => [c, i]));
  }
  encode(t) { const o = []; for (const c of t) if (this.a2i.has(c)) o.push(this.a2i.get(c)); return o; }
  reset() { this.n = 0; }
  static erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); }
  ln(x, w, b) { const n = x.length; let m = 0, v = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
    for (let i = 0; i < n; i++) v += (x[i] - m) ** 2; v /= n; const r = 1 / Math.sqrt(v + 1e-5), o = new Float32Array(n);
    for (let i = 0; i < n; i++) o[i] = (x[i] - m) * r * w[i] + b[i]; return o; }
  lin(x, W, b, out) { const inn = x.length, o = new Float32Array(out);
    for (let j = 0; j < out; j++) { let s = b ? b[j] : 0; const r = j * inn; for (let i = 0; i < inn; i++) s += x[i] * W[r + i]; o[j] = s; } return o; }
  step(tok) {
    const d = this.d, hs = d / this.H, pos = this.n, w = this.w;
    if (pos >= this.bloc) throw new Error('contexte plein');
    let x = new Float32Array(d);
    for (let i = 0; i < d; i++) x[i] = w['tok.weight'][tok * d + i] + w['pos.weight'][pos * d + i];
    for (let l = 0; l < this.L; l++) {
      const p = `blocs.${l}.`, h = this.ln(x, w[p + 'n1.weight'], w[p + 'n1.bias']);
      const qkv = this.lin(h, w[p + 'att.qkv.weight'], null, 3 * d);
      this.K[l].set(qkv.subarray(d, 2 * d), pos * d); this.V[l].set(qkv.subarray(2 * d, 3 * d), pos * d);
      const y = new Float32Array(d), sc = new Float32Array(pos + 1);
      for (let hh = 0; hh < this.H; hh++) {
        const o = hh * hs; let mx = -Infinity;
        for (let t = 0; t <= pos; t++) { let s = 0; for (let j = 0; j < hs; j++) s += qkv[o + j] * this.K[l][t * d + o + j]; s /= Math.sqrt(hs); sc[t] = s; if (s > mx) mx = s; }
        let z = 0; for (let t = 0; t <= pos; t++) { sc[t] = Math.exp(sc[t] - mx); z += sc[t]; }
        for (let t = 0; t <= pos; t++) { const a = sc[t] / z; for (let j = 0; j < hs; j++) y[o + j] += a * this.V[l][t * d + o + j]; }
      }
      const pr = this.lin(y, w[p + 'att.out.weight'], w[p + 'att.out.bias'], d); for (let i = 0; i < d; i++) x[i] += pr[i];
      const h2 = this.ln(x, w[p + 'n2.weight'], w[p + 'n2.bias']);
      const m = this.lin(h2, w[p + 'mlp.0.weight'], w[p + 'mlp.0.bias'], 4 * d);
      for (let i = 0; i < m.length; i++) m[i] = 0.5 * m[i] * (1 + MiniGPTJS.erf(m[i] / Math.SQRT2));
      const m2 = this.lin(m, w[p + 'mlp.2.weight'], w[p + 'mlp.2.bias'], d); for (let i = 0; i < d; i++) x[i] += m2[i];
    }
    this.n++;
    return this.lin(this.ln(x, w['norm.weight'], w['norm.bias']), w['tete.weight'], w['tete.bias'], this.vocab);
  }
  feed(ids) { this.reset(); let lg; for (const t of ids) lg = this.step(t); return lg; }
  sample(lg, temp) { let mx = -Infinity; for (const v of lg) if (v > mx) mx = v;
    const p = new Float32Array(lg.length); let z = 0; for (let i = 0; i < lg.length; i++) { p[i] = Math.exp((lg[i] - mx) / temp); z += p[i]; }
    let r = Math.random() * z; for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; } return p.length - 1; }
  async generate(ids, { maxNew = 300, temperature = 0.8, stop = null } = {}) {
    const keep = this.bloc - Math.max(1, Math.floor(this.bloc / 4)); const all = ids.length ? ids.slice() : [0];
    let lg = this.feed(all.slice(-keep)), out = '';
    for (let i = 0; i < maxNew; i++) {
      const id = this.sample(lg, temperature); all.push(id); out += this.chars[id];
      if (stop && out.endsWith(stop)) break;
      lg = this.n >= this.bloc ? this.feed(all.slice(-keep)) : this.step(id);
      if (i % 8 === 7) await new Promise((r) => setTimeout(r));
    }
    return out;
  }
}
if (typeof module !== 'undefined') module.exports = { MiniGPTJS };
