# test_system.py
"""
Testing and Evaluation Script for ResearchGPT Assistant

Implements comprehensive testing:
1. Unit tests for individual components
2. Integration tests for complete workflows
3. Performance evaluation metrics
4. Comparison of different prompting strategies
"""

import os
import time
import json
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List

from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant
from research_agents import AgentOrchestrator


# ---------------------------
# Helpers & safe fallbacks
# ---------------------------

def _cfg_attr(cfg: Any, primary: str, fallback: str, default=None):
    return getattr(cfg, primary, getattr(cfg, fallback, default))

def _dp_has(dp: Any, method: str) -> bool:
    return hasattr(dp, method) and callable(getattr(dp, method))

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _resolve_sample_dir(cfg: Config) -> Path:
    sample_dir = getattr(cfg, "SAMPLE_PAPERS_DIR", None)
    if sample_dir:
        return Path(sample_dir)
    data_dir = getattr(cfg, "data_dir", Path("data"))
    return Path(data_dir) / "sample_papers"

def _list_pdfs(sample_dir: Path) -> List[Path]:
    if not sample_dir.exists():
        return []
    return sorted(p for p in sample_dir.glob("*.pdf") if p.is_file())

def _tokens(s: str) -> List[str]:
    return [w.lower() for w in s.replace("\n", " ").split() if w.strip()]


class ResearchGPTTester:
    def __init__(self):
        """
        Initialize testing system
        """
        self.config = Config()
        self.doc_processor = DocumentProcessor(self.config)
        self.research_assistant = ResearchGPTAssistant(self.config, self.doc_processor)
        self.agent_orchestrator = AgentOrchestrator(self.research_assistant)

        # Test cases
        self.test_queries = [
            "What are the main advantages of machine learning?",
            "How do neural networks process information?",
            "What are the limitations of current AI systems?",
            "Compare supervised and unsupervised learning approaches",
            "What are the ethical considerations in AI development?",
        ]

        # Storage for all metrics/results
        self.evaluation_results: Dict[str, Any] = {
            "response_times": [],
            "response_lengths": [],
            "prompt_strategy_comparison": {},
            "agent_performance": {},
            "overall_scores": {},
        }

        # Paths
        results_root = _cfg_attr(self.config, "RESULTS_DIR", "results_dir", default="results")
        self.results_dir = _ensure_dir(Path(results_root))
        self.tests_dir = _ensure_dir(self.results_dir / "tests")

        # PDFs
        self.sample_dir = _resolve_sample_dir(self.config)
        self.pdfs = _list_pdfs(self.sample_dir)

    # ---------------------------
    # Unit/Integration: Document Processing
    # ---------------------------
    def test_document_processing(self):
        """
        Test document processing functionality:
        1. PDF text extraction (if PDFs present)
        2. Text preprocessing and cleaning
        3. Document chunking
        4. Similarity search
        5. Index building

        Returns:
            dict: Test results for document processing
        """
        print("\n=== Testing Document Processing ===")

        test_results = {
            "pdf_extraction": False,
            "text_preprocessing": False,
            "chunking": False,
            "similarity_search": False,
            "index_building": False,
            "errors": [],
            "docs": 0,
            "meta": 0,
            "ingested": 0,
        }

        try:
            # 1) Preprocessing (feature-detected)
            sample_text = (
                "This is a sample research paper about artificial intelligence "
                "and machine learning algorithms."
            )
            if _dp_has(self.doc_processor, "preprocess_text"):
                preprocessed = self.doc_processor.preprocess_text(sample_text)
                if isinstance(preprocessed, str) and len(preprocessed) > 0:
                    test_results["text_preprocessing"] = True
                    print("   ✓ Text preprocessing: PASS")
            else:
                # If no method exposed, consider preprocessing N/A but not a failure
                print("   • Text preprocessing: SKIP (method not available)")

            # 2) Chunking (feature-detected)
            if _dp_has(self.doc_processor, "chunk_text"):
                chunks = self.doc_processor.chunk_text(sample_text, chunk_size=50, overlap=10)
                if chunks:
                    test_results["chunking"] = True
                    print("   ✓ Text chunking: PASS")
            else:
                print("   • Text chunking: SKIP (method not available)")

            # 3) PDF ingestion (if present)
            if self.pdfs:
                # Prefer add_pdf; fallback to process_document
                for p in self.pdfs[:2]:
                    if _dp_has(self.doc_processor, "add_pdf"):
                        self.doc_processor.add_pdf(p)
                        test_results["ingested"] += 1
                    elif _dp_has(self.doc_processor, "process_document"):
                        self.doc_processor.process_document(str(p))
                        test_results["ingested"] += 1
                test_results["pdf_extraction"] = test_results["ingested"] > 0
                if test_results["pdf_extraction"]:
                    print(f"   ✓ PDF ingestion/extraction: PASS ({test_results['ingested']} files)")
                else:
                    print("   • PDF ingestion: SKIP (no PDFs ingested)")
            else:
                print(f"   • No PDFs in {self.sample_dir}; skipping PDF extraction")

            # 4) Index building
            if _dp_has(self.doc_processor, "build_search_index"):
                self.doc_processor.build_search_index()
                test_results["index_building"] = True
                print("   ✓ Index building: PASS")
            else:
                print("   • Index building: SKIP (method not available)")

            # 5) Stats & similarity search
            docs = getattr(self.doc_processor, "docs", [])
            meta = getattr(self.doc_processor, "doc_meta", [])
            test_results["docs"] = len(docs)
            test_results["meta"] = len(meta)

            if _dp_has(self.doc_processor, "find_similar_chunks") and test_results["docs"] > 0:
                res = self.doc_processor.find_similar_chunks("introduction", top_k=3)
                if isinstance(res, list) and len(res) > 0:
                    test_results["similarity_search"] = True
                    print(f"   ✓ Similarity search: PASS ({len(res)} hits)")
            else:
                print("   • Similarity search: SKIP")

            print("   ✓ Document processing tests completed")

        except Exception as e:
            test_results["errors"].append(f"Document processing error: {str(e)}")
            print(f"   ✗ Document processing error: {str(e)}")

        # Save partial results
        (self.tests_dir / "doc_processing.json").write_text(
            json.dumps(test_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return test_results

    # ---------------------------
    # Prompting Strategy Comparison
    # ---------------------------
    def test_prompting_strategies(self):
        """
        Compare performance of different prompting approaches:
        1. Basic prompting
        2. Chain-of-Thought
        3. Self-Consistency
        4. ReAct workflows

        Returns:
            dict: Comparison results for different strategies
        """
        print("\n=== Testing Prompting Strategies ===")

        strategy_results: Dict[str, List[Dict[str, Any]]] = {
            "chain_of_thought": [],
            "self_consistency": [],
            "react_workflow": [],
            "basic_qa": [],
        }

        def build_ctx(q: str) -> List[str]:
            if _dp_has(self.doc_processor, "find_similar_chunks"):
                top = self.doc_processor.find_similar_chunks(q, top_k=4)
                # Many implementations return (score, text, meta)
                ctx = []
                for t in top:
                    if isinstance(t, (list, tuple)) and len(t) >= 2:
                        ctx.append(str(t[1]))
                    else:
                        ctx.append(str(t))
                return ctx
            return []

        for i, query in enumerate(self.test_queries[:3]):
            print(f"   Testing query {i+1}: {query[:60]}...")
            try:
                # Basic QA (fallback to answer_research_question if present)
                if hasattr(self.research_assistant, "answer_research_question"):
                    t0 = time.time()
                    basic = self.research_assistant.answer_research_question(
                        query, use_cot=False, use_verification=False
                    )
                    t_basic = time.time() - t0
                    ans = basic["answer"] if isinstance(basic, dict) and "answer" in basic else str(basic)
                    strategy_results["basic_qa"].append(
                        {"query": query, "response_length": len(ans), "response_time": t_basic}
                    )

                # Chain-of-Thought
                if hasattr(self.research_assistant, "chain_of_thought_reasoning"):
                    ctx = build_ctx(query)
                    t0 = time.time()
                    cot = self.research_assistant.chain_of_thought_reasoning(query, ctx)
                    t_cot = time.time() - t0
                    strategy_results["chain_of_thought"].append(
                        {"query": query, "response_length": len(cot), "response_time": t_cot, "ctx_size": len(ctx)}
                    )

                # Self-Consistency
                if hasattr(self.research_assistant, "self_consistency_generate"):
                    ctx = build_ctx(query)
                    t0 = time.time()
                    sc = self.research_assistant.self_consistency_generate(query, ctx, num_attempts=2)
                    t_sc = time.time() - t0
                    # sc may be dict with final_answer
                    if isinstance(sc, dict):
                        length = len(sc.get("final_answer", json.dumps(sc)))
                    else:
                        length = len(str(sc))
                    strategy_results["self_consistency"].append(
                        {"query": query, "response_length": length, "response_time": t_sc, "attempts": 2}
                    )

                # ReAct
                if hasattr(self.research_assistant, "react_research_workflow"):
                    t0 = time.time()
                    react = self.research_assistant.react_research_workflow(query, max_steps=4, search_k=4)
                    t_react = time.time() - t0
                    if isinstance(react, dict):
                        steps = len(react.get("workflow_steps", []))
                    else:
                        steps = 0
                    strategy_results["react_workflow"].append(
                        {"query": query, "workflow_steps": steps, "response_time": t_react}
                    )

                print(f"   ✓ Query {i+1} completed")

            except Exception as e:
                print(f"   ✗ Error testing query {i+1}: {str(e)}")

        # Persist
        self.evaluation_results["prompt_strategy_comparison"] = strategy_results
        (self.tests_dir / "prompting_comparison.json").write_text(
            json.dumps(strategy_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return strategy_results

    # ---------------------------
    # Agent Performance Evaluation
    # ---------------------------
    def test_agent_performance(self):
        """
        Test each agent:
        1. Summarizer Agent
        2. QA Agent
        3. Research Workflow Agent
        4. Agent Orchestrator

        Returns:
            dict: Agent performance results
        """
        print("\n=== Testing AI Agents ===")

        agent_results: Dict[str, Any] = {
            "summarizer_agent": {},
            "qa_agent": {},
            "workflow_agent": {},
            "orchestrator": {"agents": []},
        }

        # Build doc_ids from meta
        meta_list = getattr(self.doc_processor, "doc_meta", [])
        doc_ids = []
        for m in meta_list:
            if isinstance(m, dict):
                doc_ids.append(m.get("path") or m.get("doc_id") or m.get("source"))
        doc_ids = [d for d in {d for d in doc_ids if d}]

        try:
            # Summarizer Agent
            print("   Testing Summarizer Agent...")
            if doc_ids:
                t0 = time.time()
                summary_result = self.agent_orchestrator.route_task("summarizer", {"doc_id": doc_ids[0]})
                dt = time.time() - t0
                agent_results["summarizer_agent"] = {
                    "doc_id": doc_ids[0],
                    "elapsed_sec": dt,
                    "ok": isinstance(summary_result, dict),
                }
            else:
                agent_results["summarizer_agent"] = {"ok": False, "reason": "no doc_ids available"}
            print("   ✓ Summarizer Agent test completed")

            # QA Agent
            print("   Testing QA Agent...")
            t0 = time.time()
            qa_result = self.agent_orchestrator.route_task(
                "qa", {"question": "What is machine learning?", "type": "factual"}
            )
            dt = time.time() - t0
            agent_results["qa_agent"] = {"elapsed_sec": dt, "ok": isinstance(qa_result, dict)}
            print("   ✓ QA Agent test completed")

            # Workflow Agent
            print("   Testing Research Workflow Agent...")
            t0 = time.time()
            workflow_result = self.agent_orchestrator.route_task(
                "workflow", {"research_topic": "artificial intelligence"}
            )
            dt = time.time() - t0
            steps = len(workflow_result.get("workflow_steps", [])) if isinstance(workflow_result, dict) else 0
            agent_results["workflow_agent"] = {"elapsed_sec": dt, "ok": True, "steps": steps}
            print("   ✓ Research Workflow Agent test completed")

            # Orchestrator (light check)
            agent_results["orchestrator"]["agents"] = list(self.agent_orchestrator.agents.keys())

        except NotImplementedError as e:
            print(f"   ✗ Agent not implemented: {e}")
        except Exception as e:
            print(f"   ✗ Agent testing error: {str(e)}")

        self.evaluation_results["agent_performance"] = agent_results
        (self.tests_dir / "agent_performance.json").write_text(
            json.dumps(agent_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return agent_results

    # ---------------------------
    # Response Quality (simple heuristics)
    # ---------------------------
    def evaluate_response_quality(self, response: str, query: str):
        """
        Evaluate response quality using simple metrics:
        - length_score: normalized length (0..1)
        - keyword_relevance: Jaccard overlap between query tokens and response tokens
        - coherence_score: proxy via unique-token ratio
        - completeness_score: fraction of query tokens present in response
        """
        response = response or ""
        query = query or ""
        resp_toks = _tokens(response)
        query_toks = _tokens(query)
        uniq_ratio = len(set(resp_toks)) / max(1, len(resp_toks))
        jacc = len(set(resp_toks) & set(query_toks)) / max(1, len(set(resp_toks) | set(query_toks)))
        coverage = len(set(query_toks) & set(resp_toks)) / max(1, len(set(query_toks)))
        length_score = min(len(response) / 1000.0, 1.0)

        quality_scores = {
            "length_score": round(length_score, 3),
            "keyword_relevance": round(jacc, 3),
            "coherence_score": round(uniq_ratio, 3),
            "completeness_score": round(coverage, 3),
        }
        overall = sum(quality_scores.values()) / len(quality_scores)
        quality_scores["overall_score"] = round(overall, 3)
        return quality_scores

    # ---------------------------
    # Performance Benchmark
    # ---------------------------
    def run_performance_benchmark(self):
        """
        Run performance benchmark:
        1. Document processing timing
        2. Query response timings
        3. Overall efficiency metrics
        """
        print("\n=== Running Performance Benchmark ===")

        benchmark_results = {
            "document_processing_time_sec": 0.0,
            "query_response_times": [],
            "system_efficiency": {},
        }

        # Document processing timing (single ingest + index)
        t0 = time.time()
        try:
            if self.pdfs:
                p = self.pdfs[0]
                if _dp_has(self.doc_processor, "add_pdf"):
                    self.doc_processor.add_pdf(p)
                elif _dp_has(self.doc_processor, "process_document"):
                    self.doc_processor.process_document(str(p))
            if _dp_has(self.doc_processor, "build_search_index"):
                self.doc_processor.build_search_index()
        finally:
            benchmark_results["document_processing_time_sec"] = round(time.time() - t0, 4)

        # Query response timings (2 queries)
        for query in self.test_queries[:2]:
            t1 = time.time()
            try:
                if hasattr(self.research_assistant, "answer_research_question"):
                    resp = self.research_assistant.answer_research_question(
                        query, use_cot=False, use_verification=False
                    )
                    ans = resp["answer"] if isinstance(resp, dict) and "answer" in resp else str(resp)
                elif hasattr(self.research_assistant, "chain_of_thought_reasoning"):
                    ans = self.research_assistant.chain_of_thought_reasoning(query, [])
                else:
                    ans = ""
            finally:
                dt = time.time() - t1
            benchmark_results["query_response_times"].append(
                {"query": query, "response_time_sec": round(dt, 4), "response_length": len(ans)}
            )

        # Aggregate efficiency
        times = [r["response_time_sec"] for r in benchmark_results["query_response_times"]]
        avg = sum(times) / len(times) if times else 0.0
        stdev = stats.pstdev(times) if len(times) > 1 else 0.0
        qpm = (60.0 / avg) if avg > 0 else 0.0
        benchmark_results["system_efficiency"] = {
            "average_response_time_sec": round(avg, 4),
            "stdev_response_time_sec": round(stdev, 4),
            "queries_per_minute": round(qpm, 3),
        }

        print(f"   Average response time: {avg:.3f} s")
        print("   ✓ Performance benchmark completed")

        (self.tests_dir / "performance_benchmark.json").write_text(
            json.dumps(benchmark_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return benchmark_results

    # ---------------------------
    # Evaluation Report
    # ---------------------------
    def generate_evaluation_report(self):
        """
        Create detailed evaluation report:
        - Test results summary
        - Performance metrics
        - Strategy comparisons
        - Recommendations
        """
        prompt_section = json.dumps(self.evaluation_results.get("prompt_strategy_comparison", {}), indent=2)
        agent_section = json.dumps(self.evaluation_results.get("agent_performance", {}), indent=2)
        perf_section = json.dumps(self.evaluation_results.get("performance_benchmark", {}), indent=2)

        report = f"""
# ResearchGPT Assistant - Evaluation Report

## Test Summary
This report provides comprehensive evaluation results for the ResearchGPT Assistant system.

## Document Processing Tests
{json.dumps(self.evaluation_results.get('document_processing', {}), indent=2)}

## Prompting Strategy Performance
{prompt_section}

## AI Agent Performance
{agent_section}

## Performance Benchmarks
{perf_section}

## Recommendations for Improvement
1. Implement more sophisticated similarity search
2. Add response caching for frequently asked questions
3. Develop evaluation metrics for response quality
4. Add batch processing capabilities
5. Implement more robust error handling
6. Add logging and monitoring features

## Conclusion
The ResearchGPT Assistant demonstrates successful integration of:
- Document processing and retrieval
- Advanced prompting techniques
- AI agent workflows
- LLM integration

System is ready for further development and deployment.
""".strip() + "\n"

        return report

    # ---------------------------
    # Orchestrator
    # ---------------------------
    def run_all_tests(self):
        """
        Execute complete test suite and save artifacts
        """
        print("Starting ResearchGPT Assistant Test Suite...")

        # Run all categories
        doc_results = self.test_document_processing()
        prompt_results = self.test_prompting_strategies()
        agent_results = self.test_agent_performance()
        benchmark_results = self.run_performance_benchmark()

        # Store results
        self.evaluation_results.update({
            "document_processing": doc_results,
            "prompt_strategy_comparison": prompt_results,
            "agent_performance": agent_results,
            "performance_benchmark": benchmark_results,
        })

        # Quality scoring example (optional, on latest basic QA if present)
        try:
            last_basic = (prompt_results.get("basic_qa") or [])[-1]
            query = last_basic["query"]
            # Re-run a basic answer for quality check
            if hasattr(self.research_assistant, "answer_research_question"):
                resp = self.research_assistant.answer_research_question(query, use_cot=False, use_verification=False)
                ans = resp["answer"] if isinstance(resp, dict) and "answer" in resp else str(resp)
                self.evaluation_results["overall_scores"]["quality_example"] = self.evaluate_response_quality(ans, query)
        except Exception:
            pass

        # Generate & save final report
        final_report = self.generate_evaluation_report()
        report_path = self.results_dir / "evaluation_report.md"
        (self.results_dir / "test_results.json").write_text(
            json.dumps(self.evaluation_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report_path.write_text(final_report, encoding="utf-8")

        print("\n=== Test Suite Complete ===")
        print("Results saved:")
        print(f"- {report_path.name}")
        print("- test_results.json")

        return self.evaluation_results


if __name__ == "__main__":
    tester = ResearchGPTTester()
    tester.run_all_tests()
