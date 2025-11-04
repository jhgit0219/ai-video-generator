# Content Guidelines for Image Search

**Version:** v1.0
**Hash:** 0963512d
**Status:** Reference documentation (see `guidelines/` for modular injection)

---

## ⚠️ Important: For Developers

**DO NOT inject this entire file into LLM prompts!**

This comprehensive document (~2,700 tokens) is for reference and documentation only.

For runtime LLM injection, use the modular guidelines in `guidelines/`:
- `subject_rules.md` (~650 tokens) - For content analysis and subject extraction
- `genre_rules.md` (~700 tokens) - For genre classification
- `forbidden_mods.md` (~800 tokens) - For query refinement validation
- `priority_rules.md` (~1150 tokens) - For image ranking

**Usage:**
```python
from utils.guideline_loader import get_cached_guidelines

# For ContentAnalyzer
guidelines = get_cached_guidelines('content_analyzer')  # ~650 tokens

# For query refinement
guidelines = get_cached_guidelines('query_refinement')  # ~1750 tokens

# For image ranking
guidelines = get_cached_guidelines('ranking')  # ~1150 tokens

# For ultra-concise system prompts
guidelines = get_cached_guidelines('summary')  # ~50 tokens
```

---

## Full Documentation (Reference Only)

This file contains critical constraints and design decisions for the AI video generation pipeline.
These guidelines MUST be followed by all LLM agents involved in content analysis and query refinement.

## Subject Preservation Rules

### 1. Preserve Subject Characteristics (CRITICAL)

When extracting subjects from story content, preserve ALL distinguishing attributes:

**Character Attributes to Preserve:**
- **Gender**: "girl", "boy", "woman", "man" (NOT generic "person" or "child")
- **Age**: "young girl", "elderly man", "teenage boy", "toddler"
- **Species**: "cat", "dog", "dragon" (be specific, not "animal")
- **Appearance**: "orange cat", "tabby cat", "long-haired girl", "bearded man"
- **Role/Identity**: "guardian", "explorer", "warrior", "queen"

**Examples:**

✓ CORRECT: "young girl" (preserves gender + age)
✗ WRONG: "child" (loses gender)
✗ WRONG: "person" (loses gender + age)

✓ CORRECT: "orange tabby cat" (preserves species + appearance)
✗ WRONG: "cat" (loses appearance details)
✗ WRONG: "animal" (loses species)

✓ CORRECT: "elderly wizard with long beard"
✗ WRONG: "wizard" (loses age + appearance)
✗ WRONG: "man" (loses role)

### 2. Context-Specific Subject Extraction

When analyzing story transcripts, extract subjects based on:

**Named Characters:**
- "Mia" → "young girl" (infer from context: village, child-appropriate story, feminine name)
- "Whiskers" → "cat" (guardian cat mentioned in story)
- "King Arthur" → "medieval king" or "armored knight"
- "Optimus Prime" → "robot" or "transformer" (for sci-fi content)

**Inference Rules:**
- Child-appropriate story context → subjects are likely young
- Fantasy/adventure stories → characters may have special roles (guardian, hero, etc.)
- Historical narratives → use period-appropriate descriptors
- Sci-fi content → use genre-appropriate terms (robot, alien, spaceship)

### 3. Query Refinement Constraints

When refining visual search queries, you MUST:

1. **Never remove subject attributes** - rephrase if needed, but keep gender/age/appearance
   - ✓ "young girl holding map" → "female child with treasure map"
   - ✗ "young girl holding map" → "person with map"

2. **Never swap genders** - male pirates are NOT acceptable for female child characters
   - ✗ "young girl" → "pirate" (pirates default to male in stock images)
   - ✓ "young girl" → "girl explorer" or "girl adventurer"

3. **Avoid gender-ambiguous occupations** unless explicitly intended:
   - "pirate", "knight", "warrior", "explorer" → often return male images
   - Use: "girl pirate", "female knight", "woman warrior", "girl explorer"

4. **Preserve species in multi-character scenes:**
   - ✓ "girl and cat near ancient map"
   - ✗ "girl with pet" (loses cat specificity)
   - ✗ "person and animal" (loses both)

### 4. Genre-Specific Style Constraints

**Fiction/Fantasy Content:**
- Prefer: "illustrated", "storybook art", "fantasy art", "painted style"
- Avoid: "photorealistic", "stock photo", unless story demands it
- Characters can be stylized/illustrated

**Documentary/Historical Content:**
- Prefer: "photorealistic", "historical photograph", "archival footage"
- Avoid: "cartoon", "illustrated", "fantasy art"
- Use period-appropriate descriptors

**Sci-Fi Content:**
- Prefer: "futuristic", "concept art", "3D rendered"
- Accept: mix of illustrated and photorealistic
- Use genre terms: "robot", "spacecraft", "alien", "cyberpunk"

### 5. Forbidden Query Modifications

**NEVER do these during query refinement:**

1. Remove named subjects (cat, girl, map, etc.)
2. Swap genders (male for female, female for male)
3. Change age groups (child → adult, elderly → young)
4. Genericize species (cat → pet, dragon → creature)
5. Remove critical objects (map, sword, crown, etc.)
6. Change genre style (fantasy → photorealistic for fantasy stories)

### 6. Story Context Examples

**Example 1: Mia and Whiskers Story**
- Story type: Fantasy/Adventure, child-appropriate
- Main character: "Mia" = young girl (female child)
- Companion: "Whiskers" = cat (guardian cat)
- Critical objects: ancient map, treasure
- Visual style: Illustrated/fantasy art (NOT photorealistic stock photos)

**Required subjects per segment:**
- Segment 1 (map discovery): ["young girl", "ancient map"]
- Segment 2 (cat reveal): ["cat", "glowing eyes"]
- Segment 3 (guardian reveal): ["young girl", "cat", "map"]

**Forbidden modifications:**
- "young girl" → "child" (loses gender)
- "young girl" → "pirate" (implies male, wrong age)
- "cat" → "pet" (loses species)
- "ancient map" → "document" (loses specificity)

**Example 2: Transformers Story (Hypothetical)**
- Story type: Sci-fi/Action
- Main character: "Optimus Prime" = robot leader
- Setting: Futuristic city, war-torn landscape
- Visual style: 3D rendered, concept art

**Required subjects:**
- Segment 1: ["robot", "futuristic city"]
- Segment 2: ["transforming robot", "vehicle"]

**Forbidden modifications:**
- "robot" → "machine" (too generic)
- "Optimus Prime" → "person" (wrong species!)

### 7. Quality Over CLIP Score

**Priority Order:**
1. Subject preservation (gender, age, species, attributes) - HIGHEST PRIORITY
2. Story accuracy (correct genre, style, context)
3. CLIP similarity score - LOWEST PRIORITY

If an image has high CLIP score but wrong gender/age/species → REJECT
If an image has lower CLIP score but correct subjects → ACCEPT

**Example:**
- Image A: Male pirate with map (CLIP: 0.85) → REJECT (wrong gender)
- Image B: Young girl with map (CLIP: 0.65) → ACCEPT (correct subject)

### 8. Requery Strategy

If initial scraping returns wrong subjects (e.g., male instead of female):

1. **Add explicit gender/age terms:** "young girl" → "female child"
2. **Remove ambiguous occupations:** "explorer" → "girl with map"
3. **Add visual modifiers:** "girl", "female", "woman", "young"
4. **Try alternative phrasings:** "adventurous girl" instead of "girl adventurer"

**Do NOT:**
- Remove the subject requirement
- Accept "close enough" with wrong attributes
- Prioritize CLIP score over subject accuracy

---

## Implementation Notes

This file should be loaded and injected into:
1. ContentAnalyzer agent system prompts
2. Director Agent query refinement prompts
3. Any LLM involved in visual query generation or modification

**Usage in code:**
```python
# Load guidelines
with open("CONTENT_GUIDELINES.md", "r") as f:
    guidelines = f.read()

# Inject into system prompt
system_prompt = f"{base_prompt}\n\n# Content Guidelines\n{guidelines}"
```
