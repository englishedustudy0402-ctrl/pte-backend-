const fs = require('fs')
const c = fs.readFileSync('C:/Users/rahee/OneDrive/Desktop/pte-platform-actual/pte-platform-main/frontend/scripts/out/seed/src/data/offlineBank/raPassages500.js', 'utf8')
const re = /id: (\d+), text: '((?:[^'\\]|\\.)*)', topic: '([^']*)', difficulty: '([^']*)'/g
let m
const arr = []
while ((m = re.exec(c)) !== null) {
  arr.push({ id: +m[1], text: m[2], topic: m[3], diff: m[4], words: m[2].split(/\s+/).length })
}
console.log('total', arr.length)

// Passages that are too long (>85 words) need trimming
const over = arr.filter(x => x.words > 85)
console.log('over 85 words:', over.length)
console.log('\n--- Over-long passages (first 60) ---')
over.slice(0, 60).forEach(x => {
  console.log(`[${x.id}] (${x.words}w, ${x.diff}, ${x.topic}) ${x.text}`)
  console.log('')
})

fs.writeFileSync('C:/Users/rahee/OneDrive/Desktop/pte-platform-actual/pte-platform-main/backend/seed/ra_original.json',
  JSON.stringify(arr, null, 1), 'utf8')
console.log('\nSaved original to ra_original.json')
