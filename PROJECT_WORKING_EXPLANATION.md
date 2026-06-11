# How the Redrob Ranker Works: A Simple Explanation

This document explains the architecture of our candidate ranker, how it satisfies the strict hackathon constraints, and how it works under the hood (written so even a 10-year-old can understand it!).

---

## 🚦 Understanding the Hackathon Rules (Constraints)

The hackathon has two very different stages with distinct rules:

1. **Pre-computation (Allowed to be Slow)**:
   * **The Rule**: The organizers allow us to pre-compute things before the final submission test (see the submission specification). This step is allowed to take a long time (like 40+ minutes) and can run on any machine.
   * **What we do here**: We run `embed.py` once. It reads the candidates, understands them, and saves their mathematical signatures to disk.

2. **The Reproduction / Ranking Step (Must be Under 5 Minutes)**:
   * **The Rule**: The code that the organizers actually run to test our submission (`rank.py`) must run on a standard **CPU only**, with **no internet access (no APIs)**, using **less than 16 GB of RAM**, and finish in **under 5 minutes** (300 seconds).
   * **What we do here**: We run `rank.py`. Because all the heavy thinking (embedding) was already saved to disk during the pre-computation step, `rank.py` only has to do quick math (adding and multiplying numbers). It finishes in just **118 seconds**, well below the 300-second limit, using very little RAM and zero internet!

---

## 🧸 The 10-Year-Old Child Analogy

Imagine you are a teacher looking for the best student to join a secret Science Club. You have a giant stack of **100,000 application letters**. 

The principal says: **"You must pick the top 100 students in less than 5 minutes!"**

If you try to read all 100,000 letters and think about them, you will fail. It would take weeks! Here is how you solve it:

### Phase 1: The Assistant (Pre-computation / `embed.py`)
Before the 5-minute timer starts, you hire an assistant (our AI Model, `MiniLM`). 
1. The assistant reads every letter.
2. Instead of writing summaries, the assistant converts each letter's vibe into a **secret code of 384 numbers** (this is called an **Embedding**). 
3. The assistant also converts the Science Club requirements into 384 numbers.
4. The assistant writes all these numbers in a big notebook (`artifacts/`) and leaves. This took the assistant 44 minutes, but they did it *yesterday*, so it doesn't count against your 5-minute test today!

### Phase 2: The Fast Test (Ranking / `rank.py`)
Now, the principal starts the 5-minute timer. You open the assistant's notebook:
1. For every student, you compare their 384 numbers to the Club's 384 numbers. Since you are just comparing lists of numbers, it takes a fraction of a millisecond per student!
2. You check simple rules: Is the student in the right grade? (Experience check). Did they lie about their skills? (Honeypot check). Are they active? (Behavior check).
3. You calculate a final score for each student.
4. You print the top 100 list.

Because you only did math with numbers instead of reading letters, you finish the entire stack of 100,000 in **118 seconds**!

---

## ⚙️ The Technical Working Details

### 1. The AI Model (Embeddings)
* We use a small, lightweight AI model called **`sentence-transformers/all-MiniLM-L6-v2`** which runs locally on the CPU (no cloud APIs needed).
* It converts English text (sentences about work experience, summaries) into a list of 384 decimal numbers (a **vector**).
* If two profiles describe similar experience (even using different words), their lists of numbers will be mathematically close to each other.

### 2. Cosine Similarity (Vibe Check)
* To check how similar a candidate is to the Job Description, we calculate the **dot product** of their 384-number lists. 
* This is extremely fast because it is performed using optimized math libraries (`numpy`) that can process millions of numbers per second.

### 3. Rules & Multipliers (Details Check)
Once we have the similarity score, we adjust it using custom rules:
* **The Structural Filter**: We look at concrete fields like `years_of_experience` (checking if they fit the 5-9 years window), check if their current title matches AI roles, and verify if their skills are backed by actual months of work.
* **The Behavioral Multiplier**: We check if the candidate is active, open to work, and responds to messages, scaling down candidates who are unreachable.
* **The Honeypot Guard**: We check for inconsistencies (e.g. starting their career in 2024 but claiming 10 years of experience). Inconsistent profiles get scored `0.0` immediately.
