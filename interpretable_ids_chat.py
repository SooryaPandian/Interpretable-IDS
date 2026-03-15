import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    source: str
    score: float
    text: str


class InterpretableIDSChat:
    """RAG + Ollama helper for IDS explanation and interactive follow-up Q/A."""

    def __init__(
        self,
        kb_dir: str,
        model: str = "llama3.2:latest",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.kb_dir = kb_dir
        self.model = model
        self.ollama_base_url = ollama_base_url.rstrip("/")

        self._chunks: List[Dict[str, str]] = []
        self._vectorizer = None
        self._tfidf_matrix = None
        self._label_to_file = {
            "normal": "normal.txt",
            "fuzzers": "fuzzers.txt",
            "fuzzer": "fuzzers.txt",
            "analysis": "analysis.txt",
            "backdoors": "backdoors.txt",
            "backdoor": "backdoors.txt",
            "dos": "dos.txt",
            "denialofservice": "dos.txt",
            "exploits": "exploits.txt",
            "exploit": "exploits.txt",
            "generic": "generic.txt",
            "reconnaissance": "reconnaissance.txt",
            "recon": "reconnaissance.txt",
            "shellcode": "shellcode.txt",
            "worms": "worms.txt",
            "worm": "worms.txt",
        }

        self._load_and_index_knowledge_base()

    def _enforce_family_consistency(self, text: str, pipeline_context: Dict) -> str:
        binary_label = str(pipeline_context.get("binary_label", "unknown")).lower()
        predicted_family = str(pipeline_context.get("attack_name", "Unknown"))

        if binary_label == "normal":
            return text

        aliases = {
            "Fuzzers": ["fuzzer", "fuzzers", "fuzzing"],
            "Analysis": ["analysis"],
            "Backdoors": ["backdoor", "backdoors"],
            "DoS": ["dos", "denial of service", "denial-of-service"],
            "Exploits": ["exploit", "exploits", "exploitation"],
            "Generic": ["generic"],
            "Reconnaissance": ["recon", "reconnaissance"],
            "Shellcode": ["shellcode"],
            "Worms": ["worm", "worms"],
            "Normal": ["normal", "benign"],
        }

        predicted_aliases = set(a.lower() for a in aliases.get(predicted_family, [predicted_family]))
        lowered = text.lower()

        conflicting = []
        for family, terms in aliases.items():
            if family == predicted_family:
                continue
            for term in terms:
                if re.search(rf"\b{re.escape(term)}\b", lowered):
                    conflicting.append(family)
                    break

        if conflicting:
            note = (
                "\n\nConsistency Note: Pipeline-predicted attack family is "
                f"`{predicted_family}`. Interpret evidence under this family unless new verified evidence appears."
            )
            return text + note

        return text

    def _resolve_expected_file(self, attack_name: str) -> str:
        label_key = self._normalize_label(attack_name)
        expected_file = self._label_to_file.get(label_key)

        if expected_file is None and label_key:
            for known_key, mapped_file in self._label_to_file.items():
                if known_key in label_key or label_key in known_key:
                    expected_file = mapped_file
                    break

        return expected_file

    def _load_and_index_knowledge_base(self) -> None:
        if not os.path.isdir(self.kb_dir):
            raise RuntimeError(f"Knowledge base directory not found: {self.kb_dir}")

        txt_files = [
            os.path.join(self.kb_dir, p)
            for p in os.listdir(self.kb_dir)
            if p.lower().endswith(".txt")
        ]

        if not txt_files:
            raise RuntimeError(f"No .txt files found in knowledge base: {self.kb_dir}")

        chunks: List[Dict[str, str]] = []
        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                continue

            source = os.path.basename(file_path)
            for chunk in self._chunk_text(text):
                chunks.append({"source": source, "text": chunk})

        if not chunks:
            raise RuntimeError("Knowledge base is empty after chunking.")

        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._tfidf_matrix = self._vectorizer.fit_transform([c["text"] for c in self._chunks])

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 900) -> List[str]:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        chunks: List[str] = []
        current = ""

        for block in blocks:
            candidate = (current + "\n\n" + block).strip() if current else block
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(block) <= max_chars:
                    current = block
                else:
                    for i in range(0, len(block), max_chars):
                        chunks.append(block[i : i + max_chars])
                    current = ""

        if current:
            chunks.append(current)

        return chunks

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        idx = sims.argsort()[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for i in idx:
            results.append(
                RetrievedChunk(
                    source=self._chunks[i]["source"],
                    score=float(sims[i]),
                    text=self._chunks[i]["text"],
                )
            )
        return results

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z]", "", str(raw_label or "").strip().lower())
        return cleaned

    def retrieve_for_label(self, attack_name: str, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        expected_file = self._resolve_expected_file(attack_name)

        # If label is known, retrieve ONLY from that class document to avoid class leakage.
        if expected_file:
            label_indices = [
                i
                for i, ch in enumerate(self._chunks)
                if ch["source"].lower() == expected_file.lower()
            ]

            if label_indices:
                query_vec = self._vectorizer.transform([query])
                sims = cosine_similarity(query_vec, self._tfidf_matrix[label_indices]).flatten()
                local_idx = sims.argsort()[::-1][:top_k]
                return [
                    RetrievedChunk(
                        source=self._chunks[label_indices[i]]["source"],
                        score=float(sims[i]),
                        text=self._chunks[label_indices[i]]["text"],
                    )
                    for i in local_idx
                ]

        # Unknown label fallback: global retrieval.
        return self.retrieve(query=query, top_k=top_k)

    def _chat_ollama(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
        }

        req = urllib.request.Request(
            url=f"{self.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise RuntimeError(
                "Cannot reach Ollama. Ensure Ollama is running and model is pulled: "
                "`ollama pull llama3.2:latest`."
            ) from e

        data = json.loads(body)
        message = data.get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content

    @staticmethod
    def _context_to_text(pipeline_context: Dict) -> str:
        # Keep context compact but information rich.
        return json.dumps(pipeline_context, indent=2, default=str)

    def generate_initial_summary(self, pipeline_context: Dict) -> Tuple[str, List[RetrievedChunk]]:
        attack_name = str(pipeline_context.get("attack_name", "unknown"))
        attack_prob = pipeline_context.get("attack_probability", None)
        binary_label = str(pipeline_context.get("binary_label", "unknown"))

        query = f"{attack_name} intrusion detection behavior indicators mitigation impact"
        if binary_label == "normal":
            query = "normal benign traffic baseline behavior interpretation"
        if attack_prob is not None:
            query += f" probability {attack_prob}"

        target_label = attack_name if binary_label != "normal" else "normal"
        expected_file = self._resolve_expected_file(target_label)
        retrieved = self.retrieve_for_label(attack_name=target_label, query=query, top_k=4)
        kb_text = "\n\n".join(
            [f"[{r.source} | score={r.score:.4f}]\n{r.text}" for r in retrieved]
        )

        system_prompt = (
            "You are an IDS security analyst assistant. "
            "Use ONLY the provided pipeline evidence and retrieved RAG knowledge. "
            "Do not hallucinate unsupported facts. "
            "Write clear, actionable, concise analysis for SOC analysts. "
            "Never change the predicted attack family from pipeline context."
        )

        user_prompt = (
            "Create an interpretable report for this IDS prediction in a concise clinical/security-report style.\n\n"
            "Required sections:\n"
            "1) Final Decision Summary\n"
            "2) Binary and Multiclass Evidence\n"
            "3) SHAP Feature-Based Explanation\n"
            "4) Attack Context From Knowledge Base\n"
            "5) Recommended Next Investigation Steps\n\n"
            "Interpretation constraints:\n"
            "- Do not only restate raw values; interpret what they mean for risk.\n"
            "- Mention top SHAP features explicitly and connect them to sample feature values.\n"
            "- For normal classification, explain why indicators suggest benign traffic and what to monitor.\n"
            "- If a feature is notable (example: sttl around high/low extremes), call out its likely relevance.\n"
            "- Keep attack family consistent with `attack_name` from pipeline context.\n"
            "- Use clear analyst language and avoid vague statements.\n\n"
            "Pipeline Context:\n"
            f"{self._context_to_text(pipeline_context)}\n\n"
            "Retrieved Knowledge:\n"
            f"{kb_text}\n\n"
            f"Expected Knowledge File For This Prediction: {expected_file}"
        )

        summary = self._chat_ollama(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        summary = self._enforce_family_consistency(summary, pipeline_context)

        return summary, retrieved

    def chat_follow_up(
        self,
        pipeline_context: Dict,
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> Tuple[str, List[RetrievedChunk]]:
        attack_name = str(pipeline_context.get("attack_name", "unknown"))
        binary_label = str(pipeline_context.get("binary_label", "unknown"))
        target = attack_name if binary_label != "normal" else "normal"
        retrieved = self.retrieve_for_label(target, user_message, top_k=4)
        kb_text = "\n\n".join(
            [f"[{r.source} | score={r.score:.4f}]\n{r.text}" for r in retrieved]
        )

        system_prompt = (
            "You are an IDS security analyst assistant in a follow-up Q/A chat. "
            "Ground responses in the provided pipeline context + retrieved RAG snippets."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Pipeline Context:\n"
                    f"{self._context_to_text(pipeline_context)}\n\n"
                    "Retrieved Knowledge:\n"
                    f"{kb_text}"
                ),
            },
        ]

        # Keep last turns only to control prompt size.
        for msg in chat_history[-8:]:
            if msg.get("role") in {"user", "assistant"}:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        response = self._chat_ollama(messages=messages, temperature=0.2)
        response = self._enforce_family_consistency(response, pipeline_context)
        return response, retrieved
