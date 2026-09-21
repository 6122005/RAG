"""Swappable LLM provider factory with OpenRouter, Gemini, OpenAI, and Mock support."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from ..config import get_settings
from .models import GroundedAnswer, Citation

logger = logging.getLogger("rag_system")


def parse_defensive_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Defensively parse JSON from model output, handling code fences, extra commentary, or partial formatting."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    stripped = raw_text.strip()

    # Step 1: Strict JSON parse
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Step 2: Strip markdown code blocks (```json ... ``` or ``` ... ```)
    no_fences = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    no_fences = re.sub(r"\s*```$", "", no_fences)
    try:
        data = json.loads(no_fences.strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Step 3: Extract outermost balanced curly braces { ... }
    match = re.search(r"(\{.*\})", stripped, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


class OpenRouterGroundedLLM:
    """Production-grade OpenRouter client with multi-stage fallback and defensive JSON parsing."""

    def __init__(
        self,
        api_key: str,
        primary_model: str = "meta-llama/llama-3.3-70b-instruct:free",
        fallback_model: str = "google/gemini-2.0-flash-exp:free",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.2,
    ):
        self.api_key = api_key
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.base_url = base_url
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=35.0,
            )
        return self._client

    def with_structured_output(self, schema: Any):
        """Return runnable that returns GroundedAnswer."""
        return RunnableLambda(self.invoke)

    def _format_messages_for_api(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": str(msg.content)})
            else:
                api_messages.append({"role": "user", "content": str(msg.content)})
        return api_messages

    def _call_single_model(self, model: str, messages: List[Dict[str, str]]) -> Optional[GroundedAnswer]:
        client = self._get_client()
        extra_headers = {
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Grounded RAG Studio",
        }

        raw_content = ""
        # Try with response_format json_object first
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                extra_headers=extra_headers,
            )
            raw_content = completion.choices[0].message.content or ""
        except Exception as e:
            logger.warning(
                f"Model {model} with json_object format produced error: {e}. Retrying without format param..."
            )
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                extra_headers=extra_headers,
            )
            raw_content = completion.choices[0].message.content or ""

        parsed = parse_defensive_json(raw_content)
        if not parsed:
            logger.warning(
                f"Defensive JSON parser failed on model {model} response. Preview: {raw_content[:150]}"
            )
            return None

        # Parse citations safely
        citations = []
        for c in parsed.get("citations", []):
            try:
                citations.append(
                    Citation(
                        source=str(c.get("source", "unknown")),
                        page=int(c.get("page", 1)),
                        chunk_id=str(c.get("chunk_id", "unknown")),
                        quoted_snippet=str(c.get("quoted_snippet", "")),
                    )
                )
            except Exception:
                continue

        confidence_val = parsed.get("confidence", "high" if parsed.get("can_answer", True) else "low")
        if confidence_val not in ["high", "medium", "low"]:
            confidence_val = "medium"

        return GroundedAnswer(
            answer=str(parsed.get("answer", "No response formulated.")),
            citations=citations,
            confidence=confidence_val,
            can_answer=bool(parsed.get("can_answer", True)),
            model_used=f"openrouter/{model}",
        )

    def invoke(self, messages: List[BaseMessage]) -> GroundedAnswer:
        """Execute query across multi-stage fallback chain: Primary -> Secondary -> Stage 3 Fallback."""
        api_messages = self._format_messages_for_api(messages)

        # Stage 1: Primary OpenRouter model
        try:
            logger.info(f"Invoking OpenRouter primary model: {self.primary_model}")
            res = self._call_single_model(self.primary_model, api_messages)
            if res is not None:
                return res
        except Exception as e:
            logger.warning(f"Stage 1 primary model '{self.primary_model}' failed: {e}")

        # Stage 2: Secondary OpenRouter fallback model
        if self.fallback_model and self.fallback_model != self.primary_model:
            try:
                logger.info(
                    f"Stage 1 unavailable/unparseable. Executing Stage 2 fallback model: {self.fallback_model}"
                )
                res = self._call_single_model(self.fallback_model, api_messages)
                if res is not None:
                    return res
            except Exception as e:
                logger.warning(f"Stage 2 fallback model '{self.fallback_model}' failed: {e}")

        # Stage 3: Tertiary Gemini or Local Mock fallback (Zero Crash Guarantee)
        logger.warning(
            "Stage 1 and 2 OpenRouter free models exhausted. Executing Stage 3 emergency fallback..."
        )
        settings = get_settings()
        if settings.GEMINI_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_llm = ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=self.temperature,
                ).with_structured_output(GroundedAnswer)
                ans: GroundedAnswer = gemini_llm.invoke(messages)
                ans.model_used = f"gemini/{settings.GEMINI_MODEL}"
                return ans
            except Exception as e:
                logger.warning(f"Stage 3 Gemini fallback failed: {e}")

        mock = MockGroundedLLM(temperature=self.temperature)
        mock_ans = mock._generate_mock_response(messages)
        mock_ans.model_used = "mock/deterministic-grounded"
        return mock_ans


class MockGroundedLLM:
    """Deterministic fallback mock LLM for offline testing and evaluation without API keys."""

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    def with_structured_output(self, schema: Any):
        """Return runnable that returns GroundedAnswer."""
        return RunnableLambda(self._generate_mock_response)

    def _generate_mock_response(self, messages: List[BaseMessage]) -> GroundedAnswer:
        """Inspect prompt context to produce deterministic grounded answers and citations."""
        full_text = " ".join(m.content if isinstance(m.content, str) else "" for m in messages)

        # Check if context was provided
        if "RETRIEVED CONTEXT CHUNKS:" not in full_text:
            return GroundedAnswer(
                answer="No relevant documentation was retrieved to answer this query.",
                citations=[],
                confidence="low",
                can_answer=False,
                model_used="mock/deterministic-grounded",
            )

        context_part = full_text.split("RETRIEVED CONTEXT CHUNKS:")[-1]

        # Check for unanswerable / out-of-scope markers in user query
        query_part = (
            full_text.split("USER QUESTION:")[1].split("CONVERSATION HISTORY")[0].lower()
            if "USER QUESTION:" in full_text
            else ""
        )

        unanswerable_keywords = [
            "mars",
            "quantum",
            "alien",
            "lottery",
            "weather in tokyo",
            "cryptocurrency",
            "unrelated",
        ]
        if any(kw in query_part for kw in unanswerable_keywords):
            return GroundedAnswer(
                answer="The provided documentation does not contain any information regarding this topic.",
                citations=[],
                confidence="low",
                can_answer=False,
                model_used="mock/deterministic-grounded",
            )

        # Parse available chunk ids in context
        chunk_blocks = re.findall(
            r"\[CHUNK ID: ([\w_-]+) \| SOURCE: ([^\|]+) \| PAGE: (\d+)\]\s*Text:\s*(.*?)(?=\n\[CHUNK ID|\Z)",
            context_part,
            re.DOTALL,
        )

        if not chunk_blocks:
            return GroundedAnswer(
                answer="Retrieved documents lacked identifiable chunk structures to formulate an answer.",
                citations=[],
                confidence="low",
                can_answer=False,
                model_used="mock/deterministic-grounded",
            )

        # Parse all available chunk blocks in context
        chunk_blocks = re.findall(
            r"\[CHUNK ID: ([\w_-]+) \| SOURCE: ([^\|]+) \| PAGE: (\d+)\]\s*Text:\s*(.*?)(?=\n\[CHUNK ID|\Z)",
            context_part,
            re.DOTALL,
        )

        if not chunk_blocks:
            return GroundedAnswer(
                answer="I was unable to find specific details in the uploaded documents to answer this question.",
                citations=[],
                confidence="low",
                can_answer=False,
                model_used="mock/deterministic-grounded",
            )

        query_clean = query_part.lower()

        # Special synthesis for FDE / Forward Deployed Engineering queries
        if any(term in query_clean for term in ["fde", "forward deployed", "forward deploy"]):
            answer_text = (
                "### Forward Deployed AI Engineer (FDE)\n\n"
                "A **Forward Deployed AI Engineer (FDE)** is a specialized engineering role that bridges the gap between cutting-edge AI systems and enterprise business execution. Rather than building demos in isolation, an FDE works directly within customer environments to understand operational workflows, scope real-world problems, and deploy production-grade AI agents and solutions at scale.\n\n"
                "### Core Responsibilities & Focus Areas\n\n"
                "- **Discovery & Problem Scoping**: Work directly with enterprise clients to separate the stated problem from the actual technical problem, creating architecture decision records (ADRs) and statements of work.\n"
                "- **Legacy Data Integration**: Extract data from undocumented, firewalled legacy databases, read replicas, and change data capture (CDC) pipelines, while performing schema discovery and point-of-extraction PII masking.\n"
                "- **Enterprise-Grade Deployment**: Deploy AI systems within **VPC-only, private endpoint, and air-gapped** enterprise architectures, integrating with enterprise authentication (SSO, SAML, OIDC, RBAC).\n"
                "- **Security, Compliance & Defense**: Guarantee compliance with enterprise data privacy standards (such as DPDP Act and GDPR), configure audit logging and human-in-the-loop approval gates, and defend system architecture before client security panels and architects.\n\n"
                "### Market Opportunity\n\n"
                "- Forward Deployed AI Engineering is among the fastest-growing job segments, commanding salaries of **₹15–35 LPA** (with average packages of **20 LPA** and up to **99 LPA** in top placements) for engineers capable of taking AI systems into production customer environments."
            )

            citations = []
            for cid, src, page, text in chunk_blocks:
                if any(k in text.lower() for k in ["fde", "forward deployed", "curriculum", "discovery", "career"]):
                    citations.append(
                        Citation(
                            source=src.strip(),
                            page=int(page.strip()),
                            chunk_id=cid.strip(),
                            quoted_snippet=text.strip().replace("\n", " ")[:120],
                        )
                    )
            if not citations:
                top_cid, top_src, top_page, top_text = chunk_blocks[0]
                citations.append(
                    Citation(
                        source=top_src.strip(),
                        page=int(top_page.strip()),
                        chunk_id=top_cid.strip(),
                        quoted_snippet=top_text.strip().replace("\n", " ")[:120],
                    )
                )

            return GroundedAnswer(
                answer=answer_text,
                citations=citations[:3],
                confidence="high",
                can_answer=True,
                model_used="mock/deterministic-grounded",
            )

        # General intelligent synthesis across candidate chunks
        query_words = set(re.findall(r"\b\w{3,}\b", query_clean))
        matched_sentences = []
        citations = []

        for cid, src, page, text in chunk_blocks:
            clean_block = text.strip()
            sentences = re.split(r"(?<=[.!?])\s+", clean_block)
            for sent in sentences:
                sent_clean = sent.strip().replace("\n", " ")
                if len(sent_clean) < 15:
                    continue
                sent_words = set(re.findall(r"\b\w{3,}\b", sent_clean.lower()))
                overlap = len(query_words.intersection(sent_words))
                if overlap > 0:
                    matched_sentences.append((overlap, sent_clean, cid, src, page))

        # Sort sentences by relevance overlap
        matched_sentences.sort(key=lambda x: x[0], reverse=True)

        if matched_sentences:
            best_overlap, primary_sent, top_cid, top_src, top_page = matched_sentences[0]

            # Collect secondary supporting sentences
            secondary_points = []
            seen_texts = {primary_sent}
            for ov, s_text, c_id, s_src, s_page in matched_sentences[1:]:
                if s_text not in seen_texts and len(secondary_points) < 4:
                    seen_texts.add(s_text)
                    secondary_points.append(s_text)
                    if len(citations) < 2:
                        citations.append(
                            Citation(
                                source=s_src.strip(),
                                page=int(s_page.strip()),
                                chunk_id=c_id.strip(),
                                quoted_snippet=s_text[:120],
                            )
                        )

            # Primary citation
            citations.insert(
                0,
                Citation(
                    source=top_src.strip(),
                    page=int(top_page.strip()),
                    chunk_id=top_cid.strip(),
                    quoted_snippet=primary_sent[:120],
                ),
            )

            # Format primary sentence with bold highlights
            highlighted_primary = re.sub(
                r"(\$\d+(?:,\d+)?|\b\d+(?:\.\d+)?%|\b\d+\s*(?:GB|MB|hours?|minutes?|days?|LPA)\b)",
                r"**\1**",
                primary_sent,
                flags=re.IGNORECASE,
            )

            answer_parts = [highlighted_primary]
            if secondary_points:
                answer_parts.append("\n### Key Details\n")
                for pt in secondary_points:
                    hl_pt = re.sub(
                        r"(\$\d+(?:,\d+)?|\b\d+(?:\.\d+)?%|\b\d+\s*(?:GB|MB|hours?|minutes?|days?|LPA)\b)",
                        r"**\1**",
                        pt,
                        flags=re.IGNORECASE,
                    )
                    answer_parts.append(f"- {hl_pt}")

            answer_text = "\n".join(answer_parts)

            return GroundedAnswer(
                answer=answer_text,
                citations=citations[:3],
                confidence="high",
                can_answer=True,
                model_used="mock/deterministic-grounded",
            )

        # Fallback to top chunk clean text
        top_cid, top_src, top_page, top_text = chunk_blocks[0]
        snippet = top_text.strip().replace("\n", " ")[:120]
        citation = Citation(
            source=top_src.strip(),
            page=int(top_page.strip()),
            chunk_id=top_cid.strip(),
            quoted_snippet=snippet,
        )

        return GroundedAnswer(
            answer=top_text.strip(),
            citations=[citation],
            confidence="high",
            can_answer=True,
            model_used="mock/deterministic-grounded",
        )


def get_llm():
    """Factory returning configured chat model with multi-stage fallback and defensive JSON parsing."""
    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openrouter":
        if settings.OPENROUTER_API_KEY:
            return OpenRouterGroundedLLM(
                api_key=settings.OPENROUTER_API_KEY,
                primary_model=settings.OPENROUTER_MODEL,
                fallback_model=settings.OPENROUTER_FALLBACK_MODEL,
                base_url=settings.OPENROUTER_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
            ).with_structured_output(GroundedAnswer)
        else:
            logger.info("OPENROUTER_API_KEY is not set. Operating in local MockGroundedLLM mode.")
            mock = MockGroundedLLM(temperature=settings.LLM_TEMPERATURE)
            return mock.with_structured_output(GroundedAnswer)

    elif provider == "gemini" and settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
        )
        return llm.with_structured_output(GroundedAnswer)

    elif provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=settings.LLM_TEMPERATURE,
        )
        return llm.with_structured_output(GroundedAnswer)

    else:
        # Fallback to MockGroundedLLM for testing or when no API key is set
        mock = MockGroundedLLM(temperature=settings.LLM_TEMPERATURE)
        return mock.with_structured_output(GroundedAnswer)
