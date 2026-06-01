import { readFileSync } from 'fs';
import { Pinecone } from '@pinecone-database/pinecone';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

if (!process.env.PINECONE_API_KEY) {
  console.error('❌ PINECONE_API_KEY not found in .env.local');
  process.exit(1);
}

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

// v7 requires targeting by host — get it from the index description
console.log('🔗 Connecting to Pinecone...');
const indexModel = await pc.describeIndex('boma-yangu-v2');
const index = pc.index({ host: indexModel.host });
console.log('✅ Connected to index host:', indexModel.host, '\n');

console.log('📖 Reading data/boma-vectors.json...');
const raw     = readFileSync('data/boma-vectors.json', 'utf-8');
const vectors = JSON.parse(raw);
console.log(`✅ Found ${vectors.length} vectors\n`);

const valid = vectors.filter(item => Array.isArray(item.embedding) && item.embedding.length === 384);
console.log(`✅ Valid vectors: ${valid.length}\n`);

const BATCH_SIZE = 100;
let uploaded = 0;

for (let i = 0; i < valid.length; i += BATCH_SIZE) {
  const batch = valid.slice(i, i + BATCH_SIZE);

  // v7 API: upsert takes { records: [...] }
  await index.upsert({
    records: batch.map((item, batchIndex) => ({
      id:     `chunk-${i + batchIndex}`,
      values: item.embedding,
      metadata: {
        text:   item.text   || '',
        source: item.source || 'unknown',
        scope:  item.scope  || 'national',
        county: item.county || 'national',
      },
    })),
  });

  uploaded += batch.length;
  console.log(`📤 Uploaded ${uploaded}/${valid.length} vectors...`);
}

console.log('\n✅ Upload complete!');
console.log(`📊 ${uploaded} vectors now live in Pinecone.`);
console.log('🔍 Check Pinecone dashboard → Database → boma-yangu-v2');