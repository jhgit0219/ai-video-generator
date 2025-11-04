# Subject Preservation Rules (v1.0)

**CRITICAL: Always preserve gender, age, species, and appearance attributes.**

## 1. Character Attribute Preservation

When extracting subjects from story content, preserve ALL distinguishing attributes:

### Gender
- ✓ CORRECT: "young girl", "woman", "boy", "man"
- ✗ WRONG: "child" (loses gender), "person" (loses gender + age)

### Age
- ✓ CORRECT: "young girl", "elderly man", "teenage boy", "toddler"
- ✗ WRONG: "person" (loses age), "adult" (too vague)

### Species
- ✓ CORRECT: "cat", "dog", "dragon", "robot"
- ✗ WRONG: "pet" (loses species), "animal" (loses species), "creature" (too vague)

### Appearance
- ✓ CORRECT: "orange tabby cat", "long-haired girl", "bearded wizard"
- ✗ WRONG: "cat" only (loses appearance), "girl" only (loses appearance)

### Role/Identity
- ✓ CORRECT: "guardian cat", "explorer girl", "warrior queen"
- ✗ WRONG: Dropping role when it's story-critical

## 2. Named Character Conversion

Convert fictional names to generic descriptors **while preserving all attributes**:

### Inference from Context
- **Feminine names** in child stories → "young girl" (not "child")
  - Example: "Mia" → "young girl" or "female child"

- **Guardian animals** → species + role
  - Example: "Whiskers the guardian" → "cat" or "guardian cat"

- **Historical figures** → role + era
  - Example: "King Arthur" → "medieval king" or "armored knight"

- **Sci-fi characters** → species/type + descriptor
  - Example: "Optimus Prime" → "transforming robot" or "robot leader"

### Multi-Attribute Preservation
When a character has multiple key attributes, preserve them all:
- "Mia with glowing map" → "young girl" + "ancient map with glow"
- "Whiskers' glowing eyes" → "cat with glowing eyes"

## 3. Forbidden Generic Terms

**NEVER use these terms when specific attributes are known:**

| Generic Term | Why It's Wrong | Use Instead |
|---|---|---|
| "child" | Loses gender | "young girl" or "boy" |
| "person" | Loses gender + age | "woman", "man", "girl", "boy" |
| "pet" | Loses species | "cat", "dog" (specific animal) |
| "animal" | Loses species | "cat", "dog", "dragon" |
| "document" | Loses object type | "map", "scroll", "letter" |
| "item" | Completely vague | Specific object name |

## 4. Gender-Specific Occupations

Avoid occupations that default to wrong gender in stock imagery:

### Problem Occupations
- "pirate" → defaults to **male** in search results
- "knight" → defaults to **male**
- "warrior" → defaults to **male**
- "explorer" → defaults to **male**
- "adventurer" → defaults to **male**

### Solutions
If character is **female**:
- ✓ "girl pirate", "female pirate", "girl explorer"
- ✗ "pirate" alone (will return male images)

If character is **male**:
- ✓ "boy knight", "male warrior"
- Can use standalone if gender isn't story-critical

## 5. Priority Hierarchy

When conflicts arise, follow this order:

1. **Subject preservation** (gender, age, species, attributes) - HIGHEST
2. **Story accuracy** (correct genre, style, context)
3. **Search popularity** (common vs rare terms) - LOWEST

**Example:**
- Specific term with low search volume > Generic term with high search volume
- "young girl with map" (specific, lower volume) > "child reading" (generic, higher volume)
