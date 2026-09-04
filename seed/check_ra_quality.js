const fs = require('fs')
const c = fs.readFileSync('C:/Users/rahee/OneDrive/Desktop/pte-platform-actual/pte-platform-main/frontend/src/data/offlineBank/raPassages500.ts', 'utf8')
const re = /id: (\d+), text: "([^"]*)", topic: "([^"]*)", difficulty: '([^']*)'/g
let m
const arr = []
while ((m = re.exec(c)) !== null) {
  arr.push({ id: +m[1], text: m[2], topic: m[3], diff: m[4], words: m[2].split(/\s+/).length })
}
console.log('total', arr.length)

// Check sentences don't end mid-sentence (no trailing words without period)
console.log("\n--- Passages that might be awkwardly cut (don't end with period) ---")
arr.filter(x => !/[.!?]$/.test(x.text.trim())).forEach(x => {
  console.log(`[${x.id}] (${x.words}w) ${x.text}`)
})

console.log('\n--- Shortest 25 passages (33-45 words) ---')
arr.sort((a, b) => a.words - b.words).slice(0, 25).forEach(x => {
  console.log(`[${x.id}] (${x.words}w, ${x.diff}) ${x.text}`)
  console.log('')
})

console.log('\n--- Sample of previously-trimmed long ones (>75w) ---')
arr.sort((a, b) => b.words - a.words).slice(0, 8).forEach(x => {
  console.log(`[${x.id}] (${x.words}w, ${x.diff}, ${x.topic}) ${x.text}`)
  console.log('')
})
