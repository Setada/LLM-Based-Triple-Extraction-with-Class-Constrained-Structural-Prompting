# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import tiktoken
from typing import List
from openai import OpenAI

# -------------------- CONFIGURATION --------------------
api_key = os.getenv("ARK_API_KEY")
base_url = ""
model = " "

input_dir = r" "
output_dir = r" "
max_tokens = 4000

# -------------------- INITIALIZE CLIENT --------------------
client = OpenAI(api_key=api_key, base_url=base_url)


# -------------------- UTIL: SPLIT TEXT INTO TOKEN-LIMITED BLOCKS --------------------
def split_text_by_sentence_and_tokens(text: str, max_tokens: int, model: str) -> List[str]:
    parts = re.split(r'(\.\s+)', text)
    segments = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i + 1]
        segments.append(sentence.strip())
    if len(parts) % 2 != 0:
        segments.append(parts[-1].strip())

    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in segments:
        sentence_tokens = len(enc.encode(sentence))
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += sentence + " "
            current_tokens += sentence_tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_tokens = sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -------------------- PROMPT TEMPLATE --------------------
system_prompt = """You are an expert aviation safety incident analyst.

Your task is to extract structured knowledge from aviation incident reports in the form of (subject, relation, object) triples.

Only return output using the exact fields listed below. Do not include any other fields or commentary.

Here is the required JSON schema structure:
[
  {
    "subject": "",
    "relation": "",
    "object": "",
    "timestamp": "",
    "measurement": "",
    "crew_info": "",
    "event_phase": "",
    "event_status": "",
    "inferred": false
  }
]

Your output should capture:
1. Incident attributes (e.g., date, aircraft type, damage, injuries)
2. Key conditions or root causes (e.g., weather, visibility, time of day, maintenance issues)
3. Critical flight phases (e.g., touchdown, rollout)
4. Pilot or crew inputs (e.g., rudder command)
5. Recorded flight data if explicitly mentioned (e.g., FMC or CVR data)

Triple construction rules:
- The relation must contain only a verb or verb phrase (e.g., "engaged", "reduced", "deployed", "activated", "caused").
- The object must be a noun or noun phrase that the verb acts upon.
- Do not combine the verb and its object into a single relation field.
- If gerunds appear (e.g., "starting hydroplaning"), extract as:
  - relation: "starting"
  - object: "hydroplaning"

Attribute Enrichment:
When available, extend each triple with the following optional attributes:
- timestamp: Exact or relative time of the event or crew action.
- measurement: Numerical values such as speed, pitch angle, height, etc.
- crew_info: Crew qualifications, flight hours, licence types (only include if pilot or crew is mentioned, otherwise leave blank).
- event_phase: Relevant flight phase (e.g., "touchdown", "rollout").
- event_status: Event condition (e.g., "normal", "abnormal", "correction", "incident").

Infer Missing Links:
If critical events are described (e.g., "tyre burst") but important causal or phase transitions (e.g., "touchdown") are implied but not explicitly stated, generate logical inferred triples.
- Mark these with:
  "inferred": true
- Example reasoning chain:
  "tyre burst" → infer "contact with ground" → infer "touchdown"

Do not summarize or interpret the report text. Only extract and return structured triples in the schema above.

Output Format Example:
[
  {
    "subject": "left main gear tyre",
    "relation": "burst",
    "object": "",
    "timestamp": "",
    "measurement": "",
    "crew_info": "",
    "event_phase": "rollout",
    "event_status": "abnormal",
    "inferred": false
  },
  {
    "subject": "left gear",
    "relation": "contacted",
    "object": "runway",
    "timestamp": "",
    "measurement": "",
    "crew_info": "",
    "event_phase": "touchdown",
    "event_status": "inferred phase",
    "inferred": true
  }
]

"""


# -------------------- MAIN PROCESSING --------------------
for filename in os.listdir(input_dir):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")

    if os.path.exists(output_path):
        continue  # Skip already processed

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        chunks = split_text_by_sentence_and_tokens(file_content, max_tokens=max_tokens, model=model)

        with open(output_path, "w", encoding="utf-8") as out_f:
            for chunk in chunks:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Here is the input text:\n{chunk}"}
                    ]
                )
                out_f.write(response.choices[0].message.content + "\n")

        print(f"[✓] Output written to: {output_path}")

    except Exception as e:
        print(f"[✗] Error processing {filename}: {e}")
