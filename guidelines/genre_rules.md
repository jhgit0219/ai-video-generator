# Genre & Style Rules (v1.0)

**Match visual style to content genre for appropriate image search results.**

## Genre Classifications

### Documentary/Historical
**When to use:**
- Real historical events, figures, or periods
- Modern documentaries, science, nature
- Archival or realistic content

**Visual style:**
- "photorealistic documentary footage"
- "historical archival photography"
- "black and white archival photo" (for pre-1960s)

**Search qualifiers:**
- ["photorealistic", "documentary"]
- ["archival", "historical photograph"]
- ["period photograph", "authentic"]

---

### Realistic Fiction
**When to use:**
- Fictional stories set in realistic modern/historical settings
- No magic, no sci-fi technology
- Drama, thriller, contemporary fiction

**Visual style:**
- "cinematic film still"
- "realistic photography"
- "dramatic cinematic lighting"

**Search qualifiers:**
- ["cinematic", "film quality"]
- ["dramatic lighting", "realistic"]

---

### Fantasy Fiction
**When to use:**
- Magical elements, mythical creatures
- Medieval/fantasy settings
- Clearly fictional characters (Mia, Whiskers, dragons, wizards)

**Visual style:**
- "illustrated storybook art"
- "fantasy concept art"
- "painted illustration"

**Search qualifiers:**
- ["illustrated", "storybook art"]
- ["fantasy art", "concept art"]
- ["painted", "artistic rendering"]

**CRITICAL:** Fantasy content should use illustrated/artistic style, NOT photorealistic!
- ✓ "illustrated young girl with cat"
- ✗ "photorealistic young girl with cat" (wrong style for fantasy)

---

### Science Fiction
**When to use:**
- Futuristic technology
- Space travel, robots, aliens
- Cyberpunk, dystopian settings

**Visual style:**
- "sci-fi concept art"
- "futuristic digital art"
- "3D rendered sci-fi"

**Search qualifiers:**
- ["futuristic", "concept art"]
- ["sci-fi rendering", "digital art"]
- ["cyberpunk", "3D render"]

---

### Animated/Stylized
**When to use:**
- Clearly animated or cartoon-style stories
- Highly stylized artistic narratives

**Visual style:**
- "animated illustration"
- "stylized art"
- "cartoon rendering"

**Search qualifiers:**
- ["animated", "cartoon"]
- ["stylized", "artistic"]

---

## Style Matching Examples

### Example 1: Mia and Whiskers Story
- **Content:** Fictional child discovers magical map with guardian cat
- **Genre:** `fiction_fantasy`
- **Style:** "illustrated storybook art"
- **Qualifiers:** ["illustrated", "storybook"]
- **Why:** Fictional names + magical elements → fantasy → illustrated style

### Example 2: Historical Documentary
- **Content:** "In 1945, soldiers landed on Normandy beach"
- **Genre:** `documentary_historical`
- **Style:** "historical archival photography, black and white"
- **Qualifiers:** ["archival", "historical", "WWII era"]
- **Why:** Real historical event → archival photographic style

### Example 3: Transformers Story
- **Content:** "Optimus Prime transformed into a truck"
- **Genre:** `fiction_scifi`
- **Style:** "sci-fi concept art, 3D rendered"
- **Qualifiers:** ["futuristic", "3D render", "concept art"]
- **Why:** Robots + transformation tech → sci-fi → digital art style

---

## Common Mistakes

### ❌ Mistake 1: Photorealistic fantasy
**Wrong:** "photorealistic young girl holding magical map"
- Problem: Fantasy content styled as documentary
- Result: Won't find appropriate magical/illustrated images

**Correct:** "illustrated young girl holding ancient map, storybook art"

### ❌ Mistake 2: Illustrated historical
**Wrong:** "illustrated soldiers landing on Normandy beach, storybook art"
- Problem: Real historical event treated as fiction
- Result: Disrespectful, historically inaccurate

**Correct:** "soldiers landing on Normandy beach, historical photograph, 1945"

### ❌ Mistake 3: Genre mixing
**Wrong:** "documentary footage of dragon flying over castle"
- Problem: Mixing documentary style with fantasy subject
- Result: No search results (dragons don't exist in documentary footage!)

**Correct:** "dragon flying over medieval castle, fantasy concept art"

---

## Priority Rule

**Visual style MUST match genre, not just subject matter.**

Genre determines style → Style determines search qualifiers → Qualifiers determine image sources
