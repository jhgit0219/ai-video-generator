# Priority & Ranking Rules (v1.0)

**When conflicts arise, follow this decision hierarchy.**

## Priority Hierarchy (Highest to Lowest)

### 1. Subject Preservation (HIGHEST PRIORITY)
**All subject attributes (gender, age, species, appearance) must be preserved.**

**Rationale:** Story accuracy and character consistency are non-negotiable.

**Example:**
- Image A: Male pirate with map (CLIP: 0.85) → **REJECT** (wrong gender)
- Image B: Young girl with map (CLIP: 0.65) → **ACCEPT** (correct subject)

**Decision:** Choose Image B despite lower CLIP score.

---

### 2. Story Accuracy
**Genre, style, and narrative context must match the content.**

**Rationale:** Photorealistic images of fantasy content are jarring and inappropriate.

**Example:**
- Image A: Photo of real girl reading (CLIP: 0.80, photorealistic) → **REJECT** (wrong style for fantasy)
- Image B: Illustrated girl with cat (CLIP: 0.70, storybook art) → **ACCEPT** (correct style)

**Decision:** Choose Image B for fantasy content.

---

### 3. CLIP Similarity Score (LOWEST PRIORITY)
**Image-to-text relevance score from CLIP model.**

**Rationale:** Useful for ranking AFTER subject and style filters are satisfied.

**When to use:**
- All candidates have correct subjects → use CLIP to pick best composition
- All candidates match genre style → use CLIP to pick most relevant scene

**When NOT to prioritize:**
- CLIP score high but wrong gender → **REJECT**
- CLIP score high but wrong style → **REJECT**

---

## Quality Over Quantity

### Threshold Rules

**Minimum acceptable CLIP score:** 0.35
- Below this → consider requery with refined search terms
- Exception: If subject is preserved and style is correct, accept scores as low as 0.25

**Subject preservation threshold:** 100%
- Missing any required subject (girl, cat, map) → automatic rejection
- No exceptions

---

## Conflict Resolution

### Scenario 1: High CLIP, Wrong Subject
**Image:** Man with treasure map (CLIP: 0.90)
**Required:** Young girl with map

**Decision:** REJECT
**Reason:** Subject preservation > CLIP score

---

### Scenario 2: Low CLIP, Correct Subject
**Image:** Illustrated girl with cat (CLIP: 0.40)
**Required:** Young girl with cat

**Decision:** ACCEPT
**Reason:** Subject correct + style matches genre (fantasy) → acceptable

---

### Scenario 3: Partial Subject Match
**Image:** Girl holding book (CLIP: 0.75)
**Required:** Young girl with ancient map

**Decision:** REJECT (missing critical object)
**Reason:** Map is story-critical → cannot substitute "book"

**Requery:** Try more specific terms like "girl treasure map parchment"

---

### Scenario 4: Genre Style Mismatch
**Image:** Photorealistic cat portrait (CLIP: 0.85)
**Required:** Cat with glowing eyes (fantasy genre)

**Decision:** CONDITIONAL
- If no illustrated options → accept as fallback
- If illustrated options available → prefer illustrated

---

## Requery Strategy

When initial search yields poor results:

### Step 1: Check subject presence
- If subjects missing → **ADD** explicit subject terms to query
- Example: "adventure scene" → "girl with cat holding map"

### Step 2: Check gender specificity
- If getting wrong gender → **ADD** gender explicitly
- Example: "explorer with map" → "girl explorer with treasure map"

### Step 3: Check style qualifiers
- If getting wrong visual style → **ADD** genre style
- Example: "cat glowing eyes" → "cat glowing eyes, illustrated fantasy art"

### Step 4: Simplify language
- If no results at all → **REMOVE** rare/complex terms
- Example: "mystical feline guardian" → "cat watching over girl"

### DO NOT:
- ❌ Remove subjects to get higher scores
- ❌ Swap genders to get more results
- ❌ Genericize species to broaden search
- ❌ Change genre style to match available images

---

## Scoring Rubric

### Final Image Score Calculation

**Components:**
1. **Subject Match** (0-50 points)
   - All subjects present with correct attributes: 50 pts
   - Missing one subject: 0 pts (automatic disqualification)
   - Wrong gender/age/species: 0 pts (automatic disqualification)

2. **Style Match** (0-25 points)
   - Perfect genre match (fantasy→illustrated): 25 pts
   - Acceptable match (fantasy→artistic): 15 pts
   - Style mismatch (fantasy→photo): 5 pts

3. **CLIP Similarity** (0-25 points)
   - CLIP > 0.70: 25 pts
   - CLIP 0.50-0.70: 15 pts
   - CLIP 0.35-0.50: 10 pts
   - CLIP < 0.35: 5 pts

**Minimum acceptable total:** 60/100

**Example Scores:**

| Image | Subject | Style | CLIP | Total | Decision |
|---|---|---|---|---|---|
| A: Girl + cat + map, illustrated | 50 | 25 | 15 (0.60) | 90 | ✅ ACCEPT |
| B: Girl + cat (no map), illustrated | 0 | 25 | 25 (0.85) | N/A | ❌ REJECT (missing subject) |
| C: Boy + cat + map, illustrated | 0 | 25 | 20 (0.70) | N/A | ❌ REJECT (wrong gender) |
| D: Girl + cat + map, photorealistic | 50 | 5 | 25 (0.85) | 80 | ⚠️ CONDITIONAL (style mismatch) |

---

## Edge Cases

### No perfect matches found
**Scenario:** 5 images scraped, none have all required subjects

**Action:**
1. Requery with more explicit terms (iteration 1)
2. If still failing, try alternative phrasings (iteration 2)
3. If 3 iterations fail, flag for manual review
4. DO NOT lower subject requirements to force a match

---

### Conflicting attributes
**Scenario:** Query says "young girl" but visual description says "adult woman"

**Resolution:**
- Trust the **transcript/story content** over visual_description field
- Visual_description may have errors
- If character name is "Mia" in child story → must be "young girl"

---

### Ambiguous gender
**Scenario:** Character name is gender-neutral (e.g., "Alex")

**Resolution:**
- Check story context for pronouns (he/she/they)
- If pronouns present → use corresponding gender term
- If truly ambiguous → use "child" only as last resort
- Prefer inference from context over generic terms
