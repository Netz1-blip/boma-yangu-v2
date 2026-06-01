import * as dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });

console.log('Checking env vars...');
console.log('GROQ_API_KEY:', process.env.GROQ_API_KEY ? '✅ Found' : '❌ Missing');
console.log('HF_TOKEN:', process.env.HF_TOKEN ? '✅ Found' : '❌ Missing');
console.log('PINECONE_API_KEY:', process.env.PINECONE_API_KEY ? '✅ Found' : '❌ Missing');

console.log('\nTesting Pinecone retrieval...');
import { retrieve } from './lib/retrieval.js';

const chunks = await retrieve('How do I register for Boma Yangu?', { topK: 3 });
console.log('Chunks returned:', chunks.length);
if (chunks.length > 0) {
  console.log('First chunk source:', chunks[0].source);
  console.log('First chunk score:', chunks[0].score);
  console.log('\n✅ Pinecone pipeline working!');
} else {
  console.log('❌ No chunks returned');
}