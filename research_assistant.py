# research_assistant.py
from typing import List, Sequence, Dict, Any, Optional
from collections import Counter
import json, re, time, random #typing for type hints (List, Sequence, Dict, Any, Optional), collections.Counter (for Self-Consistency voting),
# re (for regex-based parsing), and random (for jittered backoff)

from mistralai import Mistral #Changed from mistralai import Mistral to from mistralai import Mistral and removed ChatMessage
from mistralai.models.sdkerror import SDKError

# ---------- Prompt templates (no reasoning disclosure) ----------
# method with global constants: COT_TEMPLATE, SELF_CONSISTENCY_TEMPLATE, REACT_TEMPLATE, and VERIFIER_TEMPLATE. This is a cleaner, more readable design pattern 
# for configuration data that doesn't change during runtime.
COT_TEMPLATE = """You are a careful research assistant.
You may reason privately, but DO NOT reveal your reasoning steps.
Return a concise answer with citations to provided snippets like [S1], [S2].

User question:
{query}

Context snippets:
{context}

Return exactly:
Final Answer: <one concise paragraph with citations>
"""
#The final template instructs the model to reason privately ("DO NOT reveal your reasoning steps") and return a concise, citable answer
SELF_CONSISTENCY_TEMPLATE = """You are a careful research assistant.
You may think privately. DO NOT reveal your inner steps.
Answer concisely with citations like [S1], [S2].

Question:
{query}

Context snippets:
{context}

Output exactly:
Final Answer: <one concise paragraph with citations>
"""
#It's similar to CoT but focuses on generating a concise, citable answer. It uses a higher temperature during generation to produce diverse answers, 
# and the explicit template structure is vital for extracting the final answer.
# ReAct: the model outputs ONLY a JSON object:
# - {"action":"search","argument":"focused subquery"}
# - {"action":"answer","final_answer":"concise answer with [S#] citations"}
REACT_TEMPLATE = """You are an agent that must follow strict JSON I/O.

You have tools:
- search(query): retrieves relevant snippets from a local index (TF-IDF over PDF chunks)

Rules:
- Think privately; do not output your thoughts.
- Output ONLY a single JSON object per turn.
- Valid actions:
  1) {{\"action\":\"search\",\"argument\":\"<focused subquery>\"}}
  2) {{\"action\":\"answer\",\"final_answer\":\"<concise answer with [S#] citations>\"}}

Current user question:
{query}

If you need more info, first choose action "search". If you have enough info, choose action "answer".
Respond with valid JSON only.
"""
#demanding a strict JSON output ({"action":"search", ...} or {"action":"answer", ...}

VERIFIER_TEMPLATE = """You are a strict verifier. Do not reveal your reasoning.
Given a question, a draft answer, and context snippets, check if the answer is well-supported.
If unsupported claims appear, propose a corrected improved_answer that stays within the context.

Return ONLY valid JSON with fields:
{{
  "verdict": "pass" | "fail",
  "issues": ["short description of any issue"],
  "improved_answer": "<if fail, a corrected concise answer with [S#] citations; otherwise repeat the original answer>"
}}

Question:
{query}

Draft answer:
{answer}

Context snippets:
{context}
"""
#This template instructs the model to act as a strict verifier and to return a structured JSON object (`{"verdict": "pass"

# ---------- small helpers ----------
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _normalize_for_vote(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\[s\d+\]", "", s)          # remove [S#]
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _jittered_backoff(n: int, base: float = 0.7, factor: float = 2.0, cap: float = 8.0) -> float:
    return min(cap, base * (factor ** n)) + random.uniform(0, 0.25)

# ---------- main class ----------
class ResearchGPTAssistant:
    """
    Advanced prompting:
      - chain_of_thought_reasoning()
      - self_consistency_generate()
      - react_research_workflow()
      - verify_and_edit_answer()
    """
    def __init__(self, config, document_processor): #Implementation of __init__ logic
        self.config = config #Renamed self.config to config for clarity
        self.dp = document_processor #Renamed self.doc_processor to self.dp for brevity

        api_key = (self.config.mistral_api_key or "").strip() # Fetch API key from config with fallback to empty string
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is missing in your .env file")

        self.client = Mistral(api_key=api_key) #Initialize Mistral client with the provided API key

        # Try these models in order. Remove ones your account can't access.
        self.model_chain: Sequence[str] = [
            getattr(self.config, "model_name", "mistral-large-latest"),
            "mistral-medium-latest",
            "mistral-small-latest",
        ]
        self.max_retries = 3
        self.token_budget = max(128, getattr(self.config, "max_tokens", 512))
        #Defined self.model_chain, self.max_retries, and self.token_budget: This implements a robust LLM call strategy,
        # allowing the assistant to fall back to smaller models if the preferred one is unavailable and setting a token limit for responses.
    # ---------- low-level chat helpers ----------
    def _chat_once(self, model: str, messages: List[Dict[str, str]], temperature: float) -> str: # 1. Handles a single chat completion request to the Mistral API.
        resp = self.client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.token_budget,
            stream=False,
        )
        # v1 SDK: choices[0].message.content may be str or list of chunks
        content = resp.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if "text" in chunk and isinstance(chunk["text"], str):
                        parts.append(chunk["text"])
                    elif "content" in chunk and isinstance(chunk["content"], str):
                        parts.append(chunk["content"])
            return "".join(parts).strip()
        return str(content).strip()
        #_chat_once: This method handles the actual single API call. It takes the model, messages, and temperature, sets the max_tokens (from configuration)
    def _chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str: # 1. Implements retry logic with exponential backoff and jitter for robustness against transient API errors.
        last_err: Optional[Exception] = None
        for model in self.model_chain:
            for attempt in range(self.max_retries):
                try:
                    return self._chat_once(model, messages, temperature)
                except SDKError as e:
                    txt = str(e).lower()
                    # handle capacity 429 with backoff
                    if ("status 429" in txt or "capacity" in txt) and attempt < self.max_retries - 1: #
                        sleep_s = _jittered_backoff(attempt) # backoff with jitter
                        print(f"[Warn] 429 on '{model}'. retry {attempt+1}/{self.max_retries} in {sleep_s:.1f}s") # log and retry
                        time.sleep(sleep_s) # wait before retrying
                        continue # retry
                    last_err = e
                    break
                except Exception as e:
                    last_err = e
                    break
        raise RuntimeError(f"Mistral chat failed: {last_err}")
    #_chat (with retry logic): This is the high-level calling method. It iterates through the self.model_chain (model fallback) and uses a for attempt in range(self.max_retries)
    # loop with an explicit backoff and jitter wait time (via _jittered_backoff)
    def _format_context(self, chunks: List[str]) -> str: # 1. Formats the retrieved document chunks into a clean, annotated string with [S#] citations,
        # truncating each chunk
        lines = []
        for i, ch in enumerate(chunks, start=1):
            lines.append(f"[S{i}] {ch[:900]}")
        return "\n\n".join(lines)
    #_format_context: This helper formats the retrieved document chunks into a clean, annotated string with [S#] citations, truncating each chunk
    #to 900 characters for brevity.
    # ---------- Step 5 baseline ----------
    def answer_simple_question(self, query: str, k: int = 5) -> str: # 1. Retrieves top-k relevant document chunks using the document processor's find_similar_chunks
        #method.
        top = self.dp.find_similar_chunks(query, top_k=min(k, 5))
        snippets_only = [t[1] for t in top]
        messages = [
            {"role": "user", "content": COT_TEMPLATE.format(query=query, context=self._format_context(snippets_only))}
        ]
        text = self._chat(messages, temperature=self.config.temperature)
        m = re.search(r"Final Answer:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        return m.group(0).strip() if m else text.strip()

    # ---------- Step 6: CoT-style (answer-only) ----------
    def chain_of_thought_reasoning(self, query: str, context_chunks: List[str]) -> str: # 1. Uses the provided context chunks to answer a question with chain-of-thought
        # reasoning, but only returns the final answer. Implement step-by-step reasoning. re.search extracts the "Final Answer:".
        messages = [
            {"role": "user", "content": COT_TEMPLATE.format(query=query, context=self._format_context(context_chunks))}
        ]
        text = self._chat(messages, temperature=self.config.temperature)
        m = re.search(r"Final Answer:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        return m.group(0).strip() if m else text.strip()

    # ---------- Step 6: Self-Consistency ----------
    def self_consistency_generate(self, query: str, context_chunks: List[str], num_attempts: int = 3) -> Dict[str, Any]: # 1. Generates multiple diverse answers to a
        # question using self-consistency, then selects the most common answer. Iterates num_attempts; varies temperature; employs _normalize_for_vote and Counter for 
        # consensus calculation.
        num_attempts = max(2, min(7, num_attempts))
        answers: List[str] = []
        norm: List[str] = []

        prompt = SELF_CONSISTENCY_TEMPLATE.format(query=query, context=self._format_context(context_chunks))
        for i in range(num_attempts):
            messages = [{"role": "user", "content": prompt}]
            txt = self._chat(messages, temperature=0.7 + 0.1 * i)  # vary temps slightly
            m = re.search(r"Final Answer:\s*(.*)", txt, flags=re.IGNORECASE | re.DOTALL)
            ans = (m.group(0).strip() if m else txt.strip())
            answers.append(ans)
            norm.append(_normalize_for_vote(ans))

        vote = Counter(norm)
        winner_norm, _ = vote.most_common(1)[0]
        winner_idx = norm.index(winner_norm)
        winner_answer = answers[winner_idx]

        return {
            "final_answer": winner_answer,
            "all_samples": answers,
            "consensus_strength": vote[winner_norm] / len(answers),
        }

    # ---------- Step 6: ReAct-style workflow ----------
    def react_research_workflow(self, query: str, max_steps: int = 4, search_k: int = 4) -> str: # 1. Implements a ReAct-style research workflow, alternating between
        # searching for information and answering the question based on gathered context.  Manages an iterative loop; utilizes strict JSON enforcement via REACT_TEMPLATE
        # and limits context history
        observation_context: List[str] = []
        for _ in range(max_steps):
            ctx = self._format_context(observation_context) if observation_context else "(no observations yet)"
            agent_prompt = REACT_TEMPLATE.format(query=query) + f"\n\nObservations so far:\n{ctx}\n"
            out = self._chat([{"role": "user", "content": agent_prompt}], temperature=0.2)

            obj = _extract_json(out)
            if not obj or "action" not in obj:
                return self.answer_simple_question(query, k=search_k)

            action = obj["action"].lower()
            if action == "search":
                subq = (obj.get("argument") or "").strip() or query
                hits = self.dp.find_similar_chunks(subq, top_k=min(search_k, 6))
                observation_context.extend([h[1] for h in hits])
                observation_context = observation_context[-8:]  # keep buffer small
                continue
            elif action == "answer":
                final_answer = obj.get("final_answer", "").strip()
                return final_answer or self.answer_simple_question(query, k=search_k)
            else:
                return self.answer_simple_question(query, k=search_k)

        # Max steps reached → best effort from gathered context
        return self.chain_of_thought_reasoning(query, observation_context[:4] if observation_context else [])

    # ---------- Step 6: Verifier ----------
    def verify_and_edit_answer(self, answer: str, query: str, context_chunks: List[str]) -> Dict[str, Any]: # 1. Verifies a given answer against context chunks and
        # suggests improvements if the answer is unsupported. 	Uses VERIFIER_TEMPLATE; fixed temperature=0.0 for deterministic judgment; relies on structured JSON 
        # parsing for verdict and correction.
        verifier_prompt = VERIFIER_TEMPLATE.format(
            query=query, answer=answer, context=self._format_context(context_chunks)
        )
        out = self._chat([{"role": "user", "content": verifier_prompt}], temperature=0.0)
        data = _extract_json(out) or {}

        verdict = data.get("verdict", "fail")
        improved = (data.get("improved_answer") or answer).strip()
        issues = data.get("issues", [])
        return {
            "verdict": verdict,
            "issues": issues,
            "final_answer": improved if verdict == "fail" else answer,
            "raw": data
        }
