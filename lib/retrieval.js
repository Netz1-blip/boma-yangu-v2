// lib/retrieval.js
// Boma Yangu AI v2 — Pinecone Retrieval Engine
// Primary: HuggingFace embedding → Pinecone vector search
// Fallback: keyword search (when HuggingFace is slow/unavailable)
// 
// Key change from v1:
//   v1 loaded boma-vectors.json into memory and searched locally
//   v2 queries Pinecone cloud — persistent, faster, no cold-start file load
//
// Change from retrieval v2.0:
//   getPineconeIndex() now uses v7 host-based targeting (describeIndex)
//   instead of deprecated string syntax pc.index('name')

import { Pinecone } from '@pinecone-database/pinecone';

// ── Constants ────────────────────────────────────────────────────────────────

const HF_API_URL      = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction";
const TOP_K           = 5;
const HF_TIMEOUT_MS   = 5000;
const HF_RETRIES      = 2;
const HF_RETRY_DELAY  = 400;

// ── Pinecone connection ───────────────────────────────────────────────────────
// Initialised once, reused across requests
// v7 requires host-based targeting via describeIndex

let _index = null;

async function getPineconeIndex() {
  if (_index) return _index;
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const indexModel = await pc.describeIndex('boma-yangu-v2');
  _index = pc.index({ host: indexModel.host });
  console.log('[retrieval] Pinecone index connected: ' + indexModel.host);
  return _index;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── HuggingFace embedding (with timeout + retry) ──────────────────────────────
// Kept exactly the same as v1 — same model, same logic, same fallback behaviour

async function embedQuery(queryText) {
  const hfToken = process.env.HF_TOKEN;
  if (!hfToken) throw new Error('HF_TOKEN not set.');

  let lastError;

  for (let attempt = 1; attempt <= HF_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), HF_TIMEOUT_MS);

    try {
      const response = await fetch(HF_API_URL, {
        method: 'POST',
        headers: {
          Authorization: 'Bearer ' + hfToken,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs: queryText,
          options: { wait_for_model: true },
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (response.status === 429 || response.status === 503) {
        console.warn('[retrieval] HF status ' + response.status + ' on attempt ' + attempt);
        if (attempt < HF_RETRIES) await sleep(HF_RETRY_DELAY);
        continue;
      }

      if (!response.ok) {
        const errText = await response.text();
        throw new Error('HF error [' + response.status + ']: ' + errText);
      }

      const result = await response.json();
      const embedding = Array.isArray(result[0]) ? result[0] : result;

      if (!Array.isArray(embedding) || embedding.length !== 384) {
        throw new Error('Bad embedding shape: ' + embedding?.length);
      }

      if (attempt > 1) console.log('[retrieval] HF succeeded on attempt ' + attempt);
      return embedding;

    } catch (err) {
      clearTimeout(timer);
      lastError = err;
      const reason = err.name === 'AbortError'
        ? 'timeout after ' + HF_TIMEOUT_MS + 'ms'
        : err.message;
      console.warn('[retrieval] Attempt ' + attempt + ' failed: ' + reason);
      if (attempt < HF_RETRIES) await sleep(HF_RETRY_DELAY);
    }
  }

  throw new Error('HF failed after ' + HF_RETRIES + ' attempts: ' + lastError?.message);
}

// ── Keyword fallback ──────────────────────────────────────────────────────────
// Used when HuggingFace is unavailable
// Searches against a small in-memory fallback set passed in from the caller
// For Pinecone we pass an empty array — returns empty, chat.js handles gracefully

function keywordFallback(query, candidates, topK) {
  console.log('[retrieval] Using keyword fallback.');

  const stopWords = new Set([
    'a','an','the','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should',
    'i','you','he','she','it','we','they','me','him','her','us','them',
    'and','or','but','so','yet','for','nor','as','at','by','to','of',
    'in','on','with','from','into','about','what','how','when','where',
    'which','who','that','this','these','those','my','your','our','their',
    'na','ya','wa','za','la','ni','si','kwa','katika','au','pia','hii',
    'hizi','hiyo','hilo','je','nini','wapi','jinsi','wakati','ikiwa',
  ]);

  const queryWords = query.toLowerCase()
    .split(/\s+/)
    .filter(w => w.length > 2 && !stopWords.has(w));

  if (queryWords.length === 0 || candidates.length === 0) return [];

  const scored = candidates.map(chunk => {
    const text = chunk.text.toLowerCase();
    let score = 0;
    for (const word of queryWords) {
      const count = (text.match(new RegExp(word, 'g')) || []).length;
      score += count;
    }
    return { ...chunk, score: score / (queryWords.length * 10), method: 'keyword' };
  });

  return scored
    .filter(c => c.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ── Main retrieve function ────────────────────────────────────────────────────
// Called by api/chat.js — signature identical to v1
// opts.county is passed from the chat UI county selector

export async function retrieve(query, opts = {}) {
  const { topK = TOP_K, county = null } = opts;

  // getPineconeIndex is now async (needs await for describeIndex on first call)
  const index = await getPineconeIndex();

  try {
    // Step 1: Embed the user's question with HuggingFace
    const queryVector = await embedQuery(query);

    // Step 2: Build Pinecone query options
    const queryOptions = {
      vector:          queryVector,
      topK:            topK,
      includeMetadata: true,
    };

    // Step 3: Add county filter if user selected a county
    // This filters to chunks tagged for that county OR national chunks
    if (county && county !== 'all') {
      queryOptions.filter = {
        $or: [
          { county: { $eq: county.toLowerCase() } },
          { county: { $eq: 'national' } },
        ],
      };
    }

    // Step 4: Query Pinecone
    const results = await index.query(queryOptions);

    console.log('[retrieval] Pinecone returned ' + results.matches.length + ' matches');

    // Step 5: Return in same format as v1 so api/chat.js needs no changes
    return results.matches.map(match => ({
      text:   match.metadata.text,
      source: match.metadata.source || 'unknown',
      scope:  match.metadata.scope  || 'national',
      county: match.metadata.county || null,
      score:  match.score,
      method: 'vector',
    }));

  } catch (err) {
    console.error('[retrieval] Pinecone query failed: ' + err.message);
    // Return empty array — api/chat.js and formatContext handle this gracefully
    return [];
  }
}

// ── formatContext — unchanged from v1 ─────────────────────────────────────────
// api/chat.js calls this to build the context string for the LLM prompt

export function formatContext(chunks) {
  if (!chunks || chunks.length === 0) {
    return 'NO_KB_MATCH: No relevant content found in the knowledge base. The model MUST NOT invent an answer. Say you do not have that information and direct the user to bomayangu.go.ke or call 0700 832 832.';
  }

  const method = chunks[0]?.method === 'keyword' ? ' [keyword fallback]' : '';
  console.log('[retrieval] Formatting ' + chunks.length + ' chunks' + method);

  return chunks
    .map((c, i) => {
      const label = c.county
        ? '[' + (c.scope || 'NATIONAL').toUpperCase() + ' - ' + c.county + ']'
        : '[' + (c.scope || 'NATIONAL').toUpperCase() + ']';
      return '--- Source ' + (i + 1) + ': ' + c.source + ' ' + label + ' ---\n' + c.text;
    })
    .join('\n\n');
}