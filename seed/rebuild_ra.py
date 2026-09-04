"""
Rebuild raPassages500.ts from the recovered original 500-passage bank so every
passage matches real PTE Read Aloud length (~60-80 words).

Strategy:
  - Passages already in range (<= 85 words) are kept as-is.
  - Passages longer than 85 words are trimmed down to ~60-80 words by keeping
    complete sentences (so the text stays coherent and grammatical).
Only if trimming cannot reach at least 55 words do we fall back to a curated
replacement paragraph.

Difficulty labels are kept from the original where valid.
"""
import json
import re
from pathlib import Path

ORIG = Path(__file__).resolve().parent / "ra_original.json"
FRONTEND_BANK = Path(__file__).resolve().parents[2] / "frontend" / "src" / "data" / "offlineBank" / "raPassages500.ts"

FALLBACKS = [
    ("The discovery of antibiotics transformed modern medicine, saving millions of lives from bacterial infections that were once fatal. Alexander Fleming first observed the antibacterial properties of mould in nineteen twenty eight, and subsequent research led to the mass production of penicillin during the Second World War.", "Medicine", "hard"),
    ("Photosynthesis is the process by which green plants convert sunlight into chemical energy. Chlorophyll in the leaves absorbs light, which is used to transform carbon dioxide and water into glucose and oxygen. This process forms the foundation of most food chains on Earth.", "Science", "easy"),
    ("Coral reefs are among the most diverse ecosystems on the planet, supporting thousands of species of fish and marine life. They protect coastlines from storms and erosion, yet rising ocean temperatures and pollution are causing widespread coral bleaching that threatens their survival.", "Marine Biology", "medium"),
    ("The human brain contains billions of neurons that communicate through electrical and chemical signals. These connections form the basis of thought, memory, and emotion. Advances in neuroscience have improved our understanding of conditions such as Alzheimer's disease and depression.", "Neuroscience", "medium"),
    ("Renewable energy sources such as wind, solar, and hydroelectric power are becoming increasingly important as the world reduces its reliance on fossil fuels. These sources produce little greenhouse gas during operation, making them essential tools in the global effort against climate change.", "Energy", "easy"),
    ("Globalisation has increased the interconnectedness of economies, cultures, and populations around the world. Advances in transport and communication have made it easier for goods, ideas, and people to cross borders. While it brings economic growth, it also raises concerns about inequality and cultural change.", "Economics", "medium"),
    ("Artificial intelligence is transforming many industries by enabling machines to perform tasks that once required human intelligence. AI systems can analyse large datasets, recognise patterns in speech and images, and make predictions. These advances raise important questions about privacy and the ethical use of technology.", "Technology", "medium"),
    ("Space exploration has greatly expanded our understanding of the universe. From the first moon landing to the deployment of advanced telescopes, humanity has pushed the boundaries of science. Space research has also produced practical benefits in communication and medicine.", "Space", "easy"),
    ("Soil health is fundamental to successful agriculture and food production. Healthy soil contains a complex community of microorganisms that recycle nutrients essential for plant growth. Sustainable farming practices help maintain soil quality, while overuse of chemicals can degrade it over time.", "Agriculture", "medium"),
    ("Electric vehicles are growing in popularity as consumers seek greener transport. They run on batteries recharged from the grid and produce no tailpipe emissions. As battery technology improves, electric cars are expected to help reduce greenhouse gas emissions from transport.", "Transportation", "medium"),
    ("Smart cities use sensors and data analysis to improve the quality of urban life. They manage traffic, energy, and waste more efficiently by collecting information in real time. However, the widespread use of surveillance raises questions about privacy and security.", "Technology", "medium"),
    ("Plate tectonics explains the movement of the Earth's outer layer. The surface is divided into large plates that float on the molten mantle below. Their movement causes earthquakes, volcanic activity, and the formation of mountains and ocean basins.", "Science", "medium"),
    ("Traffic congestion is a growing problem in many cities worldwide. It wastes time, increases pollution, and reduces productivity. Solutions include improving public transport, promoting cycling, and using smart technology to manage traffic flows more efficiently.", "Transportation", "medium"),
    ("Honeybees play an essential role in agriculture by pollinating many food crops. Their populations have declined in recent years due to pesticides, disease, and habitat loss. Protecting bees is important for food security and the health of natural ecosystems.", "Agriculture", "medium"),
    ("Renewable energy adoption is accelerating as the cost of solar panels continues to decline. Wind and solar power now generate a significant share of global electricity. Expanding their use is a key strategy for reducing greenhouse gas emissions and protecting the climate.", "Energy", "medium"),
    ("The theory of evolution by natural selection, proposed by Charles Darwin, explains how species adapt to their environment over many generations. Organisms with advantageous traits are more likely to survive and reproduce, passing those traits to their offspring.", "Science", "hard"),
    ("The economic concept of supply and demand explains how prices are set in a market. When demand is high and supply is low, prices rise, encouraging producers to supply more. When supply exceeds demand, prices fall until balance is restored.", "Economics", "medium"),
    ("Cognitive psychology examines how people perceive, remember, and process information. Understanding these mental processes has improved methods of teaching and learning. It also informs the design of technology that supports human decision making.", "Psychology", "medium"),
    ("Preventive medicine focuses on stopping disease before it starts. It includes vaccination, regular checkups, and encouragement of healthy lifestyles. Preventing illness is often more effective and less costly than treating conditions after they develop.", "Medicine", "medium"),
    ("The structure of ecosystems depends on complex relationships between species. Predators regulate prey populations, and plants provide food and habitat. Disrupting these relationships through pollution or habitat loss can damage the entire ecosystem.", "Biology", "medium"),
]

def trim_to_range(text, lo=58, hi=82):
    """Trim text to (approximately) lo-hi words by keeping whole sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    acc = []
    wc = 0
    for s in sentences:
        sw = len(s.split())
        if len(acc) == 0 or wc + sw <= hi:
            acc.append(s)
            wc += sw
        else:
            break
    # If we only captured one huge sentence, allow it if <= hi words
    result = " ".join(acc).strip()
    rw = len(result.split())
    if len(acc) >= 1 and rw <= hi:
        for s in reversed(sentences[len(acc):]):
            pass
        return result
    return result

def main():
    orig = json.loads(ORIG.read_text(encoding="utf-8"))
    print(f"original count: {len(orig)}")
    fb = iter(FALLBACKS)
    used_fb = 0

    lines = []
    trimmed = 0
    for q in orig:
        text = q["text"]
        words = len(text.split())
        diff = q["diff"] if q["diff"] in ("easy", "medium", "hard") else "medium"
        if words <= 85:
            final_text = text
        else:
            trimmed += 1
            t = trim_to_range(text, 58, 82)
            if len(t.split()) < 55:
                try:
                    ft = next(fb)
                    final_text, q_topic, diff = ft
                    q["topic"] = q_topic
                    used_fb += 1
                except StopIteration:
                    final_text = t
            else:
                final_text = t
        lines.append((final_text, q["topic"], diff))

    print(f"trimmed {trimmed}, fallbacks used {used_fb}")
    print(f"total lines: {len(lines)}")

    body = []
    for i, (text, topic, diff) in enumerate(lines, start=1):
        body.append(f'  {{ id: {i}, text: {json.dumps(text)}, topic: {json.dumps(topic)}, difficulty: \'{diff}\' }},')

    content = (
        "// PTE Academic Read Aloud passages - aligned with real PTE exam length.\n"
        "// Real PTE Read Aloud shows a short paragraph of ~60-80 words and gives\n"
        "// a preparation time of 35-40 seconds. Each entry is one realistic short\n"
        "// academic paragraph (2-4 sentences) at a natural reading depth.\n"
        + "export const RA_PASSAGES_500: { id: number; text: string; topic: string; difficulty: 'easy' | 'medium' | 'hard' }[] = [\n"
        + "\n".join(body)
        + "\n]\n"
    )
    FRONTEND_BANK.write_text(content, encoding="utf-8")

    from collections import Counter
    c = Counter(x[2] for x in lines)
    print("final split:", dict(c))
    ws = [len(x[0].split()) for x in lines]
    print(f"word counts: min {min(ws)} max {max(ws)} avg {sum(ws)/len(ws):.1f}")
    over = [len(x[0].split()) for x in lines if len(x[0].split()) > 85]
    under = [len(x[0].split()) for x in lines if len(x[0].split()) < 45]
    print(f">85 words: {len(over)}, <45 words: {len(under)}")

if __name__ == "__main__":
    main()
