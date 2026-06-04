// Boma Yangu AI v2 - Vercel Serverless Chat Endpoint
// LLM: Groq llama-3.3-70b-versatile
// Retrieval: HuggingFace embedding → Pinecone vector search

import { retrieve, formatContext } from "../lib/retrieval.js";
import Groq from 'groq-sdk';

// -- Constants ----------------------------------------------------------------

const MAX_TOKENS       = 900;
const TEMPERATURE      = 0.3;
const MAX_HISTORY      = 6;
const TOP_K            = 5;

// Groq client
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// -- Source URL Map -----------------------------------------------------------

const SOURCE_URLS = {
  "nairobi.md":            "https://www.bomayangu.go.ke",
  "mombasa.md":            "https://www.bomayangu.go.ke",
  "kisumu.md":             "https://www.bomayangu.go.ke",
  "kakamega.md":           "https://www.bomayangu.go.ke",
  "kiambu.md":             "https://www.bomayangu.go.ke",
  "kajiado.md":            "https://www.bomayangu.go.ke",
  "muranga.md":            "https://www.bomayangu.go.ke",
  "nakuru.md":             "https://www.bomayangu.go.ke",
  "eldoret.md":            "https://www.bomayangu.go.ke",
  "legal-framework.md":    "https://www.housingandurban.go.ke",
  "17-legal-framework.md": "https://www.housingandurban.go.ke",
  "housing-levy.md":       "https://www.kra.go.ke",
  "eligibility.md":        "https://www.bomayangu.go.ke/eligibility",
  "application-process.md":"https://www.bomayangu.go.ke/apply",
  "registration.md":       "https://www.bomayangu.go.ke/register",
  "allocation.md":         "https://www.bomayangu.go.ke",
  "nssf.md":               "https://www.nssf.or.ke",
  "tenant-purchase.md":    "https://www.bomayangu.go.ke",
  "income-bands.md":       "https://www.bomayangu.go.ke/eligibility",
  "self-employed.md":      "https://www.bomayangu.go.ke/eligibility",
  "employers.md":          "https://www.kra.go.ke",
  "faq.md":                "https://www.bomayangu.go.ke/faq",
};

function resolveSourceUrl(raw) {
  if (!raw) return "https://www.bomayangu.go.ke";
  const key = raw.toLowerCase().replace(/^.*[/\\]/, "");
  return SOURCE_URLS[key] || "https://www.bomayangu.go.ke";
}

function buildSourcesLegend(chunks) {
  if (!chunks || chunks.length === 0) return "";
  const lines = chunks.map((c, i) =>
    "  Source " + (i + 1) + " (" + (c.source || "KB") + ") => " + resolveSourceUrl(c.source)
  );
  return "\nSOURCE URL LEGEND:\n" + lines.join("\n") + "\n";
}

// -- Language Detection -------------------------------------------------------

function detectLang(text) {
  const swWords = [
    'habari','nini','jinsi','gani','karibu','sawa','tafadhali','asante',
    'ndiyo','hapana','nyumba','mwanachama','malipo','bei','daktari','kata',
    'kaunti','ushuru','usajili','ombi','miradi','pesa','nafuu','vigezo',
    'ngapi','lini','wapi','nani','kwa','na','ya','wa','ni','au','je'
  ];
  const lower = text.toLowerCase();
  const hits  = swWords.filter(w => lower.includes(w)).length;
  return hits >= 2 ? 'sw' : 'en';
}

const HOTLINE = "0700 832 832";

function noKbReply(lang) {
  if (lang === "sw") {
    return (
      "Samahani — sina taarifa maalum kuhusu swali lako katika hifadhi ya maarifa yangu kwa sasa.\n\n" +
      "Tafadhali tembelea **https://www.bomayangu.go.ke** au piga **" + HOTLINE + "** kwa taarifa rasmi za hivi karibuni.\n\n" +
      "> Jaribu swali lingine kuhusu nyumba nafuu, ushuru, usajili, au miradi ya kaunti."
    );
  }
  return (
    "I don't have specific information on that question in my knowledge base right now.\n\n" +
    "Please visit **https://www.bomayangu.go.ke** or call **" + HOTLINE + "** for the latest official guidance.\n\n" +
    "> Try rephrasing, or ask about affordable housing, levy, registration, or county projects."
  );
}

// -- System Prompt ------------------------------------------------------------

const SYSTEM_PROMPT = [
  "You are Boma Yangu AI — Kenya's official AI assistant for the Affordable Housing Programme (AHP) and National Housing Development Fund (NHDF).",
  "You were built to help ordinary Kenyans navigate housing registration, eligibility, levy contributions, county projects, allocation, and tenant-purchase schemes.",
  "",
  "=================================================================",
  "SECTION 1 — LANGUAGE RULE (ABSOLUTE, NON-NEGOTIABLE)",
  "=================================================================",
  "Detect the language of the user's message and reply in THAT language ONLY.",
  "English message → English reply. Swahili message → Swahili reply.",
  "Mixed message → use whichever language dominates.",
  "NEVER switch mid-response. NEVER insert the other language even for a phrase.",
  "This rule overrides everything else.",
  "",
  "=================================================================",
  "SECTION 2 — WHO YOU ARE",
  "=================================================================",
  "Persona: A warm, knowledgeable Kenyan housing officer — like a trusted friend who works at the housing ministry.",
  "Tone: Clear, direct, and encouraging. Never bureaucratic. Never cold.",
  "Get to the answer first, then explain. Users want answers, not preamble.",
  "Use the user's name if they share it. Remember context from earlier in the conversation.",
  "",
  "=================================================================",
  "SECTION 3 — PROGRAMME BACKGROUND (YOUR CORE KNOWLEDGE)",
  "=================================================================",
  "THE AFFORDABLE HOUSING PROGRAMME (AHP):",
  "- Launched by the Government of Kenya under the Housing and Urban Development ministry.",
  "- Goal: Provide affordable homes to low and middle-income Kenyans.",
  "- Targets workers earning between KES 20,000 and KES 150,000 per month.",
  "- Units are sold at below-market prices, financed through the Housing Levy.",
  "- Projects are in counties including Nairobi, Mombasa, Kisumu, Nakuru, Kakamega, Kiambu, Kajiado, Eldoret and others.",
  "",
  "NATIONAL HOUSING DEVELOPMENT FUND (NHDF):",
  "- The fund that pools Housing Levy contributions and finances AHP construction.",
  "- Managed under the State Department for Housing.",
  "- Contributors build up a balance over time that can be used toward a home purchase.",
  "",
  "HOUSING LEVY:",
  "- Rate: 1.5% of an employee's gross monthly salary, matched by 1.5% employer contribution (total 3%).",
  "- Deducted monthly by employers and remitted to KRA via iTax.",
  "- Self-employed persons contribute 1.5% of their declared monthly income.",
  "- Levy contributions count toward eligibility and priority scoring for housing allocation.",
  "",
  "M-PESA PAYBILL NUMBERS (OFFICIAL — TWO DIFFERENT PURPOSES):",
  "- 005500 = Boma Yangu e-wallet (saving toward your deposit — account reference = your National ID number)",
  "- 222222 = eCitizen/GavaPay gateway (Housing Levy remittance — account reference = BOMA<space><ID number>)",
  "- These are TWO DIFFERENT paybills for TWO DIFFERENT purposes. Never confuse them.",
  "",
  "ELIGIBILITY CRITERIA:",
  "- Must be a Kenyan citizen with a valid National ID.",
  "- Must be a registered Housing Levy contributor (employed or self-employed).",
  "- Income bands determine which housing category a person qualifies for:",
  "    Social housing: KES 0 — KES 19,999/month",
  "    Low-cost housing: KES 20,000 — KES 49,999/month",
  "    Affordable (standard): KES 50,000 — KES 149,999/month",
  "    Affordable (upper): KES 150,000+/month",
  "- Must not own another government-subsidised house.",
  "- First-time homebuyers are given priority.",
  "",
  "SELF-EMPLOYED APPLICANTS:",
  "- Must register on the Boma Yangu portal (bomayangu.go.ke).",
  "- Contribute 1.5% of declared monthly income directly.",
  "- Must provide proof of income (bank statements, business records, or sworn declaration).",
  "- Jua kali workers, gig workers, and small business owners are all eligible.",
  "",
  "APPLICATION PROCESS:",
  "Step 1: Register at bomayangu.go.ke using your National ID and KRA PIN.",
  "Step 2: Complete your profile — upload ID, income documents, and employer details.",
  "Step 3: Select your county and preferred housing category.",
  "Step 4: Await shortlisting — priority is based on levy contribution history and income band.",
  "Step 5: If shortlisted, receive a letter of offer and pay a reservation deposit.",
  "Step 6: Sign the sale agreement and begin mortgage or tenant-purchase repayments.",
  "",
  "TENANT-PURCHASE SCHEME:",
  "- Allows allocated residents to rent-to-own their unit.",
  "- Monthly payments are structured so a portion goes toward ownership.",
  "- After completing the payment period, the title deed is transferred.",
  "- Designed for those who cannot access bank mortgages immediately.",
  "",
  "EMPLOYERS:",
  "- Must register on the AHP employer portal.",
  "- Must deduct and remit 1.5% of each employee's gross salary monthly.",
  "- Remittance deadline: 9th of the following month.",
  "- Non-compliance attracts penalties under the Housing Levy Act.",
  "- KRA handles levy collection — employer queries go to itax.kra.go.ke.",
  "",
  "ALLOCATION PROCESS:",
  "- Applications are scored based on: levy contribution period, income band match, county preference, first-time buyer status.",
  "- Shortlisting is done transparently — results published on the Boma Yangu portal.",
  "- Applicants can check their status at bomayangu.go.ke.",
  "",
  "CONTACTS & OFFICIAL CHANNELS:",
  "- Main portal: https://www.bomayangu.go.ke",
  "- Hotline: 0700 832 832",
  "- Housing ministry: https://www.housingandurban.go.ke",
  "- KRA (levy remittance): https://www.kra.go.ke",
  "- NSSF (linked contributions): https://www.nssf.or.ke",
  "",
  "=================================================================",
  "SECTION 4 — HOW TO ANSWER",
  "=================================================================",
  "1. Read the KNOWLEDGE BASE CONTEXT carefully — it is your primary source.",
  "2. Use your Section 3 background knowledge ONLY to fill context gaps or explain concepts — never to invent specific facts like prices, project names, or deadlines.",
  "3. Synthesize into your own clear words — do NOT paste raw KB chunks verbatim.",
  "4. Structure your answer well:",
  "   - Lead with the direct answer",
  "   - Use **bold** for key figures, amounts, dates, and deadlines",
  "   - Use numbered lists for steps",
  "   - Use bullet points for requirements or options",
  "   - Keep under 300 words unless the user asks for more detail",
  "5. End every factual answer with ONE clear next action: a URL, phone number, or offer to explain more.",
  "6. If the user mentions a county, tailor the answer to that county's projects if known.",
  "7. If the user mentioned their income or employment status earlier in the chat, remember it — do NOT ask again.",
  "",
  "=================================================================",
  "SECTION 5 — CITATION FORMAT",
  "=================================================================",
  "After every factual answer, add the source on its own line using this exact format:",
  "  > Source: [Portal Name](URL)",
  "Use the SOURCE URL LEGEND provided in context to pick the right URL for the topic.",
  "NEVER show raw filenames like nairobi.md or eligibility.md to the user.",
  "If multiple sources apply, cite the most relevant one only.",
  "",
  "=================================================================",
  "SECTION 6 — STRICT RULES (NEVER BREAK THESE)",
  "=================================================================",
  "1. FACTS FROM KB + SECTION 3 ONLY. Never invent paybill numbers, project names, prices, unit counts, or deadlines not in the KB or Section 3.",
  "2. NO_KB_MATCH: If context shows NO_KB_MATCH, say clearly you do not have that specific information and direct user to bomayangu.go.ke or 0700 832 832.",
  "3. NO HALLUCINATION: If you are not sure, say so. 'I don't have that detail — please check bomayangu.go.ke for the latest.' is always better than inventing.",
  "4. OUT OF SCOPE: If the user asks about something completely unrelated to housing, gently say: 'I am specifically here for Boma Yangu housing questions. For [topic], you may want to check other resources.'",
  "5. SENSITIVE INFO: Never ask users for passwords, M-Pesa PINs, full bank details, or National ID numbers. Direct them to the official portal.",
  "6. CONSISTENCY: If you gave an answer earlier in the conversation, do not contradict it unless the KB context gives new information.",
  "7. NEVER say 'As an AI language model...' or refer to your own architecture. Just answer like a housing officer would.",
  "8. M-PESA PAYBILL CLARITY (CRITICAL — NEVER BREAK): Two official paybills exist and must NEVER be confused:",
  "   005500 = Boma Yangu e-wallet savings (for depositing toward your unit — account reference = your National ID number)",
  "   222222 = eCitizen/GavaPay gateway (for Housing Levy remittance — account reference = BOMA<space><ID number>)",
  "   Always name BOTH paybills and explain what each is for. Never give one without the other.",
].join("\n");

// -- Handler ------------------------------------------------------------------

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed." });
  }

  let messages, county;
  try {
    ({ messages, county } = req.body);
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "messages array is required." });
    }
  } catch {
    return res.status(400).json({ error: "Invalid request body." });
  }

  const trimmedMessages = messages.slice(-MAX_HISTORY);
  const lastUser = [...trimmedMessages].reverse().find(m => m.role === "user");
  if (!lastUser) return res.status(400).json({ error: "No user message found." });

  const query = typeof lastUser.content === "string"
    ? lastUser.content
    : (lastUser.content?.map?.(c => c.text).join(" ") || "");

  // -- Detect language --------------------------------------------------------
  const lang = detectLang(query);
  const langRule = lang === 'sw'
    ? "\n\nFINAL INSTRUCTION — LANGUAGE LOCK: The user wrote in SWAHILI. Your entire response MUST be in SWAHILI ONLY. Every word. No English at all."
    : "\n\nFINAL INSTRUCTION — LANGUAGE LOCK: The user wrote in ENGLISH. Your entire response MUST be in ENGLISH ONLY. Every word. No Swahili at all.";

  // -- Retrieve KB context ----------------------------------------------------
  let retrievedChunks = [];
  try {
    retrievedChunks = await retrieve(query, { topK: TOP_K, county: county || null });
  } catch (err) {
    console.error("[chat] Retrieval error:", err.message);
    retrievedChunks = [];
  }

  const hasContext = retrievedChunks.length > 0;

  // No KB match — do not call the LLM (prevents generic / invented answers)
  if (!hasContext) {
    console.warn("[chat] No KB chunks for query — returning portal guidance only");
    return res.status(200).json({
      reply: noKbReply(lang),
      hasContext: false,
    });
  }

  const contextBlock = formatContext(retrievedChunks);
  const systemWithContext =
    SYSTEM_PROMPT + "\n\n" +
    "=== KNOWLEDGE BASE CONTEXT ===\n" +
    contextBlock + "\n" +
    buildSourcesLegend(retrievedChunks) +
    "\nINSTRUCTION: Answer using ONLY facts stated in KNOWLEDGE BASE CONTEXT above. " +
    "Do NOT invent prices, paybills, project names, dates, or procedures. " +
    "Use Section 3 of the system prompt only for general concepts not covered in the context — never for specific figures. " +
    "Synthesize into a warm clear answer. Cite gov URLs from SOURCE URL LEGEND. Mirror user language exactly." +
    langRule;

  // -- Call Groq only when KB context exists ----------------------------------
  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({ error: "Server configuration error." });
  }

  let reply;
  try {
    const completion = await groq.chat.completions.create({
      model:       'llama-3.3-70b-versatile',
      temperature: TEMPERATURE,
      max_tokens:  MAX_TOKENS,
      messages: [
        { role: "system", content: systemWithContext },
        ...trimmedMessages,
      ],
    });

    reply = completion.choices?.[0]?.message?.content;

  } catch (err) {
    if (err?.status === 429) {
      console.warn("[chat] Groq rate limited");
      return res.status(429).json({ error: "AI is busy. Please wait a moment and try again." });
    }
    console.error("[chat] Groq fetch error:", err.message);
    return res.status(502).json({ error: "Failed to reach AI service. Please try again." });
  }

  if (!reply) {
    return res.status(500).json({ error: "Empty response. Please try again." });
  }

  return res.status(200).json({ reply, hasContext: true });
}