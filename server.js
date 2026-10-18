import 'dotenv/config';
import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PROVIDER = process.env.PROVIDER || 'anthropic';
const PORT = parseInt(process.env.PORT) || 3000;
const MAX_CONTEXT = 20;      // nombre de derniers messages transmis au modèle (le « contexte »)
const MAX_CHARS = 8000;      // taille maximale d'un message
const MAX_TOKENS = 1024;     // longueur maximale d'une réponse
const SYSTEM = 'Tu es un assistant conversationnel clair et honnête. Tu réponds en français par défaut. Si tu ne sais pas, tu le dis.';

class UserError extends Error { constructor(msg, status = 500) { super(msg); this.status = status; } }

// --- « Cerveau » : un modèle existant appelé via API ---
async function askAnthropic(messages) {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) throw new UserError('Aucune clé API configurée. Ajoutez ANTHROPIC_API_KEY dans le fichier .env puis relancez le serveur (ou utilisez PROVIDER=ollama, gratuit).', 503);
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model: process.env.ANTHROPIC_MODEL || 'claude-sonnet-5-5', max_tokens: MAX_TOKENS, system: SYSTEM, messages }),
  });
  if (res.status === 401) throw new UserError('Clé API refusée par Anthropic. Vérifiez ANTHROPIC_API_KEY.', 502);
  if (res.status === 429) throw new UserError('Limite de requêtes atteinte. Réessayez dans un instant.', 429);
  if (!res.ok) throw new UserError(`Le modèle a renvoyé une erreur (${res.status}). Vérifiez votre crédit et le nom du modèle.`, 502);
  const data = await res.json();
  return { text: data.content.filter((b) => b.type === 'text').map((b) => b.text).join(''), truncated: data.stop_reason === 'max_tokens' };
}

async function askOllama(messages) {
  let res;
  try {
    res = await fetch(`${process.env.OLLAMA_URL || 'http://localhost:11434'}/api/chat`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: process.env.OLLAMA_MODEL || 'llama3.2', stream: false, options: { num_predict: MAX_TOKENS },
        messages: [{ role: 'system', content: SYSTEM }, ...messages] }),
    });
  } catch { throw new UserError('Ollama est injoignable. Est-il installé et lancé ?', 503); }
  if (!res.ok) throw new UserError('Ollama a répondu avec une erreur. Le modèle est-il téléchargé (ollama pull) ?', 502);
  const data = await res.json();
  return { text: data.message?.content || '', truncated: data.done_reason === 'length' };
}

// Mode « bête » gratuit : aucun modèle, recherche Wikipédia et recopie d'extraits
async function askWebSearch(messages) {
  const q = messages.at(-1).content.slice(0, 200);
  const u = new URL('https://fr.wikipedia.org/w/api.php');
  Object.entries({ action: 'query', format: 'json', generator: 'search', gsrsearch: q, gsrlimit: '3', prop: 'extracts|info',
    exintro: '1', explaintext: '1', exsentences: '3', inprop: 'url' }).forEach(([k, v]) => u.searchParams.set(k, v));
  let res;
  try { res = await fetch(u, { headers: { 'User-Agent': 'IASimple/1.0 (projet personnel)' } }); }
  catch { throw new UserError('Impossible de joindre le web depuis le serveur.', 502); }
  if (!res.ok) throw new UserError(`La recherche web a échoué (code ${res.status}).`, 502);
  const pages = Object.values((await res.json()).query?.pages || {}).sort((a, b) => a.index - b.index).filter((p) => p.extract?.trim());
  if (!pages.length) return { text: `J’ai cherché « ${q} » sur le web, sans rien trouver d’utile. Essayez d’autres mots.`, truncated: false };
  return { text: `J’ai cherché « ${q} » sur le web. Voici ce que j’ai trouvé :\n\n` +
    pages.slice(0, 2).map((p) => `${p.title}\n${p.extract.trim()}\nSource : ${p.fullurl}`).join('\n\n') +
    '\n\n(Je n’ai pas de vrai modèle de langage : je recopie des extraits sans les comprendre.)', truncated: false };
}

const providers = { anthropic: askAnthropic, ollama: askOllama, websearch: askWebSearch };

// --- Serveur ---
const app = express();
app.set('trust proxy', 1);   // derrière le proxy de l'hébergeur : vraie IP du visiteur
// CORS : nécessaire seulement si le HTML est hébergé sur un AUTRE domaine que ce serveur
const ORIGIN = process.env.ALLOWED_ORIGIN;   // ex. https://monsite.com
if (ORIGIN) app.use((req, res, next) => {
  res.set({ 'Access-Control-Allow-Origin': ORIGIN, 'Access-Control-Allow-Headers': 'Content-Type, x-access-code', 'Access-Control-Allow-Methods': 'POST, OPTIONS' });
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});
app.use(express.json({ limit: '200kb' }));
app.get('/healthz', (_req, res) => res.send('ok'));

// Protections : code d'accès optionnel + limite de requêtes par IP (en mémoire)
const hits = new Map();
const RATE = parseInt(process.env.RATE_LIMIT_PER_MINUTE) || 15;
function guard(req, res, next) {
  const code = process.env.ACCESS_CODE;
  if (code && req.get('x-access-code') !== code) return res.status(401).json({ error: 'Code d’accès requis ou invalide.' });
  const now = Date.now(), list = (hits.get(req.ip) || []).filter((t) => now - t < 60000);
  if (list.length >= RATE) return res.status(429).json({ error: 'Trop de messages. Réessayez dans une minute.' });
  list.push(now); hits.set(req.ip, list);
  if (hits.size > 5000) for (const [k, v] of hits) if (!v.some((t) => now - t < 60000)) hits.delete(k);
  next();
}
app.use(express.static(path.join(path.dirname(fileURLToPath(import.meta.url)), 'public')));

app.post('/api/chat', guard, async (req, res) => {
  try {
    const ask = providers[PROVIDER];
    if (!ask) throw new UserError(`PROVIDER inconnu : « ${PROVIDER} » (anthropic, ollama ou websearch).`, 503);
    const input = req.body?.messages;
    if (!Array.isArray(input) || !input.length) throw new UserError('Aucun message reçu.', 400);
    let messages = input.slice(-MAX_CONTEXT).map((m) => ({ role: m?.role, content: m?.content }));
    for (const m of messages) {
      if (!['user', 'assistant'].includes(m.role) || typeof m.content !== 'string' || !m.content.trim()) throw new UserError('Message invalide.', 400);
      if (m.content.length > MAX_CHARS) throw new UserError(`Message trop long (maximum ${MAX_CHARS} caractères).`, 400);
    }
    while (messages.length && messages[0].role !== 'user') messages.shift();
    if (!messages.length || messages.at(-1).role !== 'user') throw new UserError('Le dernier message doit venir de l’utilisateur.', 400);
    const { text, truncated } = await ask(messages);
    if (!text.trim()) throw new UserError('Le modèle n’a rien répondu. Réessayez.', 502);
    res.json({ reply: text, truncated });
  } catch (e) {
    if (!(e instanceof UserError)) console.error(e);
    res.status(e.status || 500).json({ error: e instanceof UserError ? e.message : 'Erreur interne du serveur.' });
  }
});

app.listen(PORT, () => {
  console.log(`Site : http://localhost:${PORT}  (modèle : ${PROVIDER})`);
  if (PROVIDER === 'anthropic' && !process.env.ANTHROPIC_API_KEY) console.warn('⚠ ANTHROPIC_API_KEY manquante : le chat affichera une erreur de configuration.');
});
