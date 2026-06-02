Boma Yangu AI v2 is a production RAG (Retrieval-Augmented Generation) system that helps Kenyan citizens navigate the Affordable Housing Programme (AHP) and National Housing Development Fund (NHDF).
Users ask questions in English or Kiswahili — the AI retrieves the most relevant information from a curated knowledge base of 62 official documents and answers with source citations, in the user's language.
This is v2 — an upgraded pipeline over the original project. The core change: vectors moved from an in-memory JSON file to a persistent Pinecone cloud index, and Cerebras was replaced with Groq.

Live Demo
https://boma-yangu-v2.vercel.app
Try asking:

"How do I register for Boma Yangu?"
"What is the housing levy rate?"
"Nahitaji nyumba Nairobi, nianze wapi?"
"What projects are available in Mombasa?"


Stack
LayerV1 (original)V2 (this repo)EmbeddingsHuggingFace all-MiniLM-L6-v2HuggingFace all-MiniLM-L6-v2Vector storagedata/boma-vectors.json in-memoryPinecone cloud index (persistent)LLMCerebras gpt-oss-120bGroq llama-3.3-70b-versatileHostingVercel (free)Vercel (free)Cost$0$0

Architecture
User question (English or Kiswahili)
           │
           ▼
  HuggingFace Inference API
  sentence-transformers/all-MiniLM-L6-v2
  → converts question to 384-dimensional vector
           │
           ▼
  Pinecone boma-yangu-v2 index
  → cosine similarity search across 601 vectors
  → returns top 5 most relevant knowledge chunks
           │
           ▼
  api/chat.js — builds system prompt with context
           │
           ▼
  Groq llama-3.3-70b-versatile
  → answers using ONLY retrieved context
  → matches user language (EN/SW)
  → cites official government sources
           │
           ▼
  Answer with source URL + trust signal
Request path:

User sends message from index.html → POST /api/chat with messages and optional county
api/chat.js extracts last user message, calls retrieve(query, { topK: 5, county })
lib/retrieval.js embeds query via HuggingFace (with retry + timeout handling)
Top 5 chunks retrieved from Pinecone with optional county metadata filter
Chunks formatted into context block, injected into system prompt
Groq returns answer → sent to client as { reply }


Repository Structure
boma-yangu-v2/
├── index.html               # Main chat application (single-page)
├── eligibility.html         # Eligibility checker (client-side only)
├── vercel.json              # URL rewrites (/eligibility → eligibility.html)
├── package.json             # Node deps — pinecone, groq-sdk, dotenv
│
├── api/
│   └── chat.js              # Vercel serverless: RAG pipeline + Groq
├── lib/
│   └── retrieval.js         # Pinecone vector search + HF embedding + keyword fallback
├── script/
│   ├── buildVectors.js      # Offline: KB markdown → HF embeddings → JSON
│   └── uploadToPinecone.js  # One-time: push boma-vectors.json → Pinecone
├── data/
│   └── boma-vectors.json    # 601 pre-computed vectors (384 dims each)
│
└── knowledge/               # Source of truth — 62 Markdown documents
    ├── core/                # Programme facts, levy, allocation, legal, FAQ
    ├── Citizens/            # Employed, self-employed, diaspora, civil servants
    ├── Regions/             # County and regional project context
    ├── Context/             # Tone, culture, Sheng, scam navigation
    └── security/            # Scam patterns and warnings

Key Features
FeatureDetailsBilingualDetects English vs Kiswahili — replies in the user's languageRAG pipelineRetrieves top 5 KB chunks before every answer — no hallucination from thin airPersistent vectors601 vectors in Pinecone — survive deployments, cold starts, everythingCounty filterSidebar lets users filter by county — Pinecone metadata filter scopes retrievalKeyword fallbackIf HuggingFace times out, keyword search keeps the app answeringSource citationsEvery answer cites the relevant official government URLScam warningsProminent messaging — registration is FREE, official portal onlyEligibility checkerClient-side only — income band, Nairobi projects, personalised next stepsDark modeFull dark/light theme toggle

Environment Variables
VariableRequiredPurposeHF_TOKENYesHuggingFace Inference API — embedding queriesGROQ_API_KEYYesGroq chat completions — LLM answersPINECONE_API_KEYYesPinecone vector database — retrieval
Local file: .env.local (gitignored — never commit keys)

Quick Start
Prerequisites

Node.js 18+
Vercel CLI: npm install -g vercel
API keys: HuggingFace, Groq, Pinecone (all free tier)

Local development
bash# Clone the repo
git clone https://github.com/Netz1-blip/boma-yangu-v2.git
cd boma-yangu-v2

# Install dependencies
npm install

# Add your keys to .env.local
cp .env.example .env.local
# Edit .env.local with your keys

# Link to Vercel (first time only)
npx vercel link

# Pull env vars
npx vercel env pull .env.local

# Start local dev server
npx vercel dev
Open http://localhost:3000
Upload vectors to Pinecone (one time only)
If setting up a fresh Pinecone index:
bash# Create index in Pinecone dashboard first:
# Name: boma-yangu-v2 | Dimensions: 384 | Metric: cosine | AWS us-east-1

# Then push all 601 vectors
node script/uploadToPinecone.js
This reads data/boma-vectors.json (pre-computed embeddings) and uploads to Pinecone. Run once — vectors persist permanently.
Deploy to production
bashvercel --prod

Retrieval Design
Why Pinecone over local JSON
ConcernJSON (v1)Pinecone (v2)Cold startLoads entire file into RAMInstant connectionPersistenceResets on function restartPermanentAdding documentsRequires rebuild + redeployPush new vectors anytimeMulti-clientOne JSON per deploymentNamespaces — isolated per clientScaleSlow at 10k+ vectorsBuilt for millions
Embedding model
sentence-transformers/all-MiniLM-L6-v2 — outputs 384-dimensional vectors. The same model embeds both the knowledge base documents (at build time) and user questions (at runtime). This is critical — different models produce incompatible vector spaces.
County filtering
Vectors are tagged with county metadata at upload time. When a user selects a county in the UI, Pinecone filters to chunks tagged for that county OR national:
javascriptfilter: {
  $or: [
    { county: { $eq: county.toLowerCase() } },
    { county: { $eq: 'national' } },
  ]
}
Keyword fallback
If HuggingFace times out (free tier rate limits), lib/retrieval.js falls back to keyword scoring — word overlap between query and chunks. Not as precise as vector search but keeps the app answering under all conditions.

Knowledge Base
62 Markdown documents across 5 categories:

core/ — Programme rules, housing levy, eligibility, allocation, tenant purchase scheme, legal framework, FAQ
Citizens/ — Employed workers, self-employed, informal/jua kali, civil servants, diaspora, youth
Regions/ — Nairobi, Mombasa, Kisumu, Nakuru, Kakamega, Kiambu, Kajiado, Eldoret
Context/ — Kenyan tone, Sheng language, scam navigation, cultural context
security/ — Scam patterns, fraud warnings, official channel verification

All documents include last_verified dates and source citations.
Rebuilding vectors (after KB edits)
bash# .env.local must contain HF_TOKEN
node script/buildVectors.js

# Then re-upload to Pinecone
node script/uploadToPinecone.js

# Deploy
vercel --prod

What Changed From V1
V2 is a targeted infrastructure upgrade. The knowledge base, UI, language handling, and system prompt are identical to v1.
Files changed:

lib/retrieval.js — Pinecone replaces local JSON search
api/chat.js — Groq replaces Cerebras (2 lines changed)
package.json — added @pinecone-database/pinecone, groq-sdk
script/uploadToPinecone.js — new one-time upload script

Files unchanged:

index.html — chat UI
eligibility.html — eligibility checker
knowledge/ — all 62 documents
data/boma-vectors.json — all 601 vectors
vercel.json — routing config


Official References
ResourceURLBoma Yangu portalhttps://www.bomayangu.go.keHousing & Urban Developmenthttps://www.housingandurban.go.keKRA (Housing Levy)https://www.kra.go.keNSSFhttps://www.nssf.or.keAffordable Housing Act 2024Government legal database

About The Builder
Built by Netz (Neithen Muhong) — RAG Engineer, Nairobi, Kenya.
Origin: Was an attaché at Kajiado Huduma Centre. Watched Kenyan citizens queue hours for basic government information. Built AI to fix it.
Core service: RAG systems — taking an organisation's existing data and turning it into AI their staff and customers can talk to directly.
LinkedIn: linkedin.com/in/neithen-muhong
Signature: Netz | RAG Engineer
Tagline: Building AI for Africa 🇰🇪
