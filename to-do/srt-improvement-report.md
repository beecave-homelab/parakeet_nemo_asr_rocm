**Role: Subtitle & Accessibility Expert**

---

## 📝 Subtitle Improvement Report

**Project Goal**: Improve `.srt` subtitle readability using the provided ASR post-processor logic while preserving all original words.

---

### ✅ General Observations

| Aspect                        | Observation                                                                    | Status               |
| ----------------------------- | ------------------------------------------------------------------------------ | -------------------- |
| **Word Accuracy**             | Original wording is preserved as expected.                                     | ✅ Good               |
| **Line Length**               | Most lines respect the 42-character limit.                                     | ✅ Acceptable         |
| **CPS (Chars/sec)**           | Generally below 17 CPS, with some borderline segments.                         | ⚠️ Needs Spot Checks |
| **Natural Sentence Grouping** | Many cues break mid-clause or mid-sentence, reducing readability.              | ❌ Needs Improvement  |
| **Cue Duration**              | Many cues are under 1s or over 6s, especially in spontaneous speech.           | ❌ Needs Adjustment   |
| **Gap Enforcement**           | Some cues appear stacked (no 2-frame buffer).                                  | ❌ Minor Issue        |
| **Line Wrapping**             | Greedy wrapping is mostly correct, but a few cues use unnecessary hard breaks. | ⚠️ Needs Refinement  |

---

### 🔍 Specific Problem Patterns & Examples

#### 1. **Mid-Sentence Breaks (Merge Needed)**

**Example**:

```srt
1
00:00:04,000 --> 00:00:07,550
As you lift the curtain of this postcard
paradise, a very

2
00:00:07,630 --> 00:00:12,539
different reality hides in plain sight.
```

* **Issue**: Sentence broken awkwardly.
* **Suggestion**: Merge into one cue if within 6s and 84 chars.

---

#### 2. **Sub-Second Cues**

**Example**:

```srt
31
00:03:02,363 --> 00:03:02,792
like it's
```

* **Issue**: Duration < 1s, single fragment.
* **Suggestion**: Merge forward or backward for context and duration ≥ 1s.

---

#### 3. **Unnatural Timing on Interjections**

**Example**:

```srt
27
00:02:04,003 --> 00:02:07,473
Whoa! I think this is the bunker we're
coming
```

* **Issue**: "Whoa!" deserves a full line or pause.
* **Suggestion**: Break before "Whoa!", or stretch timing if possible.

---

#### 4. **Overlong Segments Without Logical Splits**

**Example**:

```srt
18
00:01:14,362 --> 00:01:32,711
Switzerland. We have some GPS coordinates of an
abandoned bunker that apparently is up this dirt
```

* **Issue**: Segment over 18s; natural sentence boundary occurs mid-way.
* **Suggestion**: Split after “Switzerland.” and re-time with proper gap.

---

#### 5. **Poor Clause Wraps**

**Example**:

```srt
14
00:00:55,910 --> 00:01:00,430
of the most remarkable ones that we know
of. On this journey, we'll get to open the
```

* **Issue**: Wrap splits at preposition; better as:

  ```
  of the most remarkable ones that we know of.
  On this journey, we'll get to open the
  ```

---

#### 6. **Long, Flat Segments Without Natural Breaks**

**Example**:

```srt
255–257
As we prepared for our second day...
...we were about to be surprised...
```

* **Issue**: Informational density too high for viewer to read easily.
* **Suggestion**: Introduce pauses and adjust wrapping for breathability.

---

### ✅ Suggestions for Refinement Rules

| Rule Type                     | Suggestion                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Clause Merging**            | Improve clause-boundary logic: recognize “that,” “which,” “but,” and conjunctions as soft pauses.                  |
| **Micro-segment Elimination** | Discard or merge any cue < 1.0s unless it is an intentional standalone reaction (“Whoa!”, “What?”, etc.).          |
| **Pause Inference**           | Introduce optional pause-based cue breaking based on intonation (if available via ASR metadata or voice activity). |
| **Narrative Grouping**        | Allow sentence groups to exceed 2 lines occasionally for documentary-style narration, provided they stay under 6s. |
| **Visual Readability**        | Ensure no line ends with "of," "to," "that," "and," etc. Move them to the next line.                               |
| **Cue Padding**               | Apply the `GAP_FRAMES` rule strictly after merge; avoid any hard cue collisions.                                   |

---

### ✅ Recommended Pipeline Tuning Parameters

| Setting                    | Suggested Value | Rationale                                             |
| -------------------------- | --------------- | ----------------------------------------------------- |
| `MIN_SEGMENT_DURATION_SEC` | 1.2             | Slightly higher to enforce better pacing for reading  |
| `MAX_CPS`                  | 17              | Keep as is (performant)                               |
| `GAP_FRAMES`               | 2 (25 fps)      | Keep strict to avoid overlapping cues                 |
| `MAX_SEGMENT_DURATION_SEC` | 5.5             | Reduce slightly for readability during fast narration |
| `CLAUSE_CHARS`             | `,;:… and`      | Add soft clause breaks on “and”                       |
| `WRAP_LIMIT`               | 42              | Keep, but add smarter wrap priority logic             |

---

### 📌 Summary

* 🟩 Your refiner logic is sound and produces readable blocks with a few exceptions.
* 🟥 Improvements are needed in cue merging at sentence and clause level.
* 🟨 Consider more semantic wrapping and stricter enforcement of duration ranges.
* ✅ A second pass on emotional/narrative pacing (for expressive speech) could further improve viewer experience.

Would you like a Python-based evaluation report script that applies these refinements on sample `.srt` files with before/after diffs?
