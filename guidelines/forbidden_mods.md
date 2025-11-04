# Forbidden Query Modifications (v1.0)

**CRITICAL: These modifications are NEVER allowed during query refinement.**

## Absolute Prohibitions

### 1. Never Remove Named Subjects

If the original query contains specific subjects, they MUST appear in the refined query.

**Examples:**

❌ **Original:** "young girl and cat with ancient map"
❌ **Refined:** "ancient map with golden light" ← **FORBIDDEN** (removed girl and cat)
✅ **Correct:** "young girl and cat near ancient map, golden light"

❌ **Original:** "Mia's cat glowing eyes map treasure"
❌ **Refined:** "person looking at document" ← **FORBIDDEN** (genericized everything)
✅ **Correct:** "young girl with cat, glowing eyes, ancient treasure map"

---

### 2. Never Swap Genders

Changing character gender is a critical error.

**Examples:**

❌ **Original:** "young girl exploring forest"
❌ **Refined:** "boy hiking through woods" ← **FORBIDDEN** (gender swap)
✅ **Correct:** "girl walking through forest"

❌ **Original:** "female scientist in laboratory"
❌ **Refined:** "scientist in lab" ← **FORBIDDEN** (removed gender, defaults to male)
✅ **Correct:** "woman scientist in laboratory"

---

### 3. Never Change Age Groups

Don't turn children into adults or vice versa.

**Examples:**

❌ **Original:** "young girl reading map"
❌ **Refined:** "woman studying cartography" ← **FORBIDDEN** (child → adult)
✅ **Correct:** "girl looking at treasure map"

❌ **Original:** "elderly wizard casting spell"
❌ **Refined:** "young mage with staff" ← **FORBIDDEN** (elderly → young)
✅ **Correct:** "old wizard performing magic"

---

### 4. Never Genericize Species

Specific animals must stay specific.

**Examples:**

❌ **Original:** "cat with glowing eyes watching girl"
❌ **Refined:** "pet observing child" ← **FORBIDDEN** (cat → pet, girl → child)
✅ **Correct:** "cat with bright eyes near young girl"

❌ **Original:** "dragon flying over medieval castle"
❌ **Refined:** "creature above fortress" ← **FORBIDDEN** (dragon → creature)
✅ **Correct:** "dragon soaring over stone castle"

---

### 5. Never Remove Critical Objects

Story-essential items cannot be dropped.

**Examples:**

❌ **Original:** "girl holding ancient treasure map"
❌ **Refined:** "young girl in adventure scene" ← **FORBIDDEN** (map removed)
✅ **Correct:** "girl with old parchment map"

❌ **Original:** "warrior wielding glowing sword"
❌ **Refined:** "armored knight in battle" ← **FORBIDDEN** (sword removed)
✅ **Correct:** "knight holding luminous blade"

---

### 6. Never Change Genre Style

Don't mix incompatible visual styles.

**Examples:**

❌ **Original:** "illustrated storybook girl with magical cat"
❌ **Refined:** "photorealistic child with pet" ← **FORBIDDEN** (fantasy → documentary)
✅ **Correct:** "illustrated young girl with mystical cat, storybook art"

❌ **Original:** "historical photograph of WWII soldiers"
❌ **Refined:** "illustrated military scene, artistic rendering" ← **FORBIDDEN** (historical → fictional)
✅ **Correct:** "archival photo of soldiers, 1944, black and white"

---

## Acceptable Modifications

These changes ARE allowed:

### ✅ Rephrase (preserving meaning)
- "mystical cat by her side" → "cat standing near girl"
- "Mia follows map" → "girl holding treasure map"
- "guardian cat" → "cat watching over child"

### ✅ Add descriptive details
- "young girl" → "young girl with long hair"
- "cat" → "cat with glowing amber eyes"
- "map" → "ancient parchment map with golden markings"

### ✅ Simplify language (preserving attributes)
- "the feline companion known as Whiskers" → "cat"
- "a young female child named Mia" → "young girl"

### ✅ Add visual style qualifiers
- "girl with map" → "girl with map, illustrated storybook art"
- "cat glowing eyes" → "cat with bright eyes, fantasy art style"

---

## Validation Checklist

Before accepting a refined query, verify:

- [ ] All named subjects present (girl, cat, map, etc.)
- [ ] Gender preserved (girl → girl, not child/person)
- [ ] Age preserved (young → young, elderly → elderly)
- [ ] Species preserved (cat → cat, not pet/animal)
- [ ] Critical objects present (map, sword, artifact)
- [ ] Genre style maintained (fantasy → illustrated, historical → photo)

If ANY checkbox fails → **REJECT THE REFINED QUERY**

---

## Common Rationalization Traps

LLMs often try to justify forbidden modifications. Reject these arguments:

### ❌ "More searchable"
**Claim:** "I changed 'young girl' to 'child' because it's more searchable"
**Response:** Subject preservation > search volume. Use "young girl" even if fewer results.

### ❌ "Better CLIP scores"
**Claim:** "I removed 'cat' because images without cats scored higher"
**Response:** Story accuracy > CLIP scores. The cat MUST be in the image.

### ❌ "Simpler is better"
**Claim:** "I simplified 'young girl' to 'person' for clarity"
**Response:** Simplification that loses attributes is forbidden. Use "young girl".

### ❌ "Same meaning"
**Claim:** "'Child' means the same as 'young girl'"
**Response:** No—"child" loses gender. They are not equivalent.

---

## Severity Levels

| Violation | Severity | Action |
|---|---|---|
| Removed subject (girl, cat, map) | CRITICAL | Immediate rejection |
| Gender swap (girl → boy) | CRITICAL | Immediate rejection |
| Genericized species (cat → pet) | HIGH | Reject and retry |
| Genericized gender (girl → child) | HIGH | Reject and retry |
| Genre style mismatch | MEDIUM | Warn and retry |
| Minor rephrasing | LOW | Accept if meaning preserved |
