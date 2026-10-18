// Fonction Vercel : reçoit la conversation, appelle le modèle, renvoie la réponse.
const MAX_CONTEXT = 20, MAX_CHARS = 8000, MAX_TOKENS = 1024;
const RATE = parseInt(process.env.RATE_LIMIT_PER_MINUTE) || 15;
const SYSTEM = 'Tu es un assistant conversationnel clair et honnête. Tu réponds en français par défaut. Si tu ne sais pas, tu le dis.';
const hits = new Map(); // limite « au mieux » : la mémoire n'est pas partagée entre instances

class UserError extends Error { constructor(m, status = 500) { super(m); this.status = status; } }

async function askAnthropic(messages) {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) throw new UserError('Aucune clé API configurée. Ajoutez ANTHROPIC_API_KEY dans les variables d’environnement Vercel, ou utilisez PROVIDER=websearch.', 503);
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model: process.env.ANTHROPIC_MODEL || 'claude-sonnet-5-5', max_tokens: MAX_TOKENS, system: SYSTEM, messages }),
  });
  if (res.status === 401) throw new UserError('Clé API refusée par Anthropic.', 502);
  if (res.status === 429) throw new UserError('Limite de requêtes atteinte. Réessayez dans un instant.', 429);
  if (!res.ok) throw new UserError(`Le modèle a renvoyé une erreur (${res.status}). Vérifiez votre crédit et le nom du modèle.`, 502);
  const d = await res.json();
  return { text: d.content.filter((b) => b.type === 'text').map((b) => b.text).join(''), truncated: d.stop_reason === 'max_tokens' };
}

async function askWebSearch(messages) {
  const q = messages.at(-1).content.slice(0, 200);
  const u = new URL('https://fr.wikipedia.org/w/api.php');
  Object.entries({ action: 'query', format: 'json', generator: 'search', gsrsearch: q, gsrlimit: '3', prop: 'extracts|info',
    exintro: '1', explaintext: '1', exsentences: '3', inprop: 'url' }).forEach(([k, v]) => u.searchParams.set(k, v));
  let res;
  try { res = await fetch(u, { headers: { 'User-Agent': 'MonIA/1.0 (projet personnel)' } }); }
  catch { throw new UserError('Impossible de joindre le web depuis le serveur.', 502); }
  if (!res.ok) throw new UserError(`La recherche web a échoué (code ${res.status}).`, 502);
  const pages = Object.values((await res.json()).query?.pages || {}).sort((a, b) => a.index - b.index).filter((p) => p.extract?.trim());
  if (!pages.length) return { text: `J’ai cherché « ${q} » sur le web, sans rien trouver d’utile. Essayez d’autres mots.`, truncated: false };
  return { text: `J’ai cherché « ${q} » sur le web. Voici ce que j’ai trouvé :\n\n` +
    pages.slice(0, 2).map((p) => `**${p.title}**\n${p.extract.trim()}\n[Source](${p.fullurl})`).join('\n\n') +
    '\n\n*Je n’ai pas de vrai modèle de langage : je recopie des extraits sans les comprendre.*', truncated: false };
}

const providers = { anthropic: askAnthropic, websearch: askWebSearch };

export default async function handler(req, res) {
  try {
    if (req.method !== 'POST') throw new UserError('Méthode non autorisée.', 405);
    const code = process.env.ACCESS_CODE;
    if (code && req.headers['x-access-code'] !== code) throw new UserError('Code d’accès requis ou invalide.', 401);
    const ip = String(req.headers['x-forwarded-for'] || 'inconnu').split(',')[0].trim();
    const now = Date.now(), list = (hits.get(ip) || []).filter((t) => now - t < 60000);
    if (list.length >= RATE) throw new UserError('Trop de messages. Réessayez dans une minute.', 429);
    list.push(now); hits.set(ip, list);

    const PROVIDER = process.env.PROVIDER || 'websearch';
    const ask = providers[PROVIDER];
    if (!ask) throw new UserError(`PROVIDER inconnu : « ${PROVIDER} » (anthropic ou websearch).`, 503);
    const input = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
    const raw = input?.messages;
    if (!Array.isArray(raw) || !raw.length) throw new UserError('Aucun message reçu.', 400);
    const messages = raw.slice(-MAX_CONTEXT).map((m) => ({ role: m?.role, content: m?.content }));
    for (const m of messages) {
      if (!['user', 'assistant'].includes(m.role) || typeof m.content !== 'string' || !m.content.trim()) throw new UserError('Message invalide.', 400);
      if (m.content.length > MAX_CHARS) throw new UserError(`Message trop long (maximum ${MAX_CHARS} caractères).`, 400);
    }
    while (messages.length && messages[0].role !== 'user') messages.shift();
    if (!messages.length || messages.at(-1).role !== 'user') throw new UserError('Le dernier message doit venir de l’utilisateur.', 400);
    const { text, truncated } = await ask(messages);
    if (!text.trim()) throw new UserError('Le modèle n’a rien répondu. Réessayez.', 502);
    res.status(200).json({ reply: text, truncated });
  } catch (e) {
    if (!(e instanceof UserError)) console.error(e);
    res.status(e.status || 500).json({ error: e instanceof UserError ? e.message : 'Erreur interne du serveur.' });
  }
}
