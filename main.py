print(">>> TOP OF main.py reached") #Debugging & Initialization: Adds print statements to track execution start and show the Python executable path for environment debugging.
import sys; print("python exe:", sys.executable)

from pathlib import Path
from datetime import datetime
from typing import List, Any, Dict, Tuple
import json
import os
import traceback

from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant
from research_agents import AgentOrchestrator


# ------------------------------
# Utilities
# ------------------------------

def _cfg_attr(cfg: Any, primary: str, fallback: str, default=None): #Access configuration attributes with fallback
    """Access cfg.primary if present else cfg.fallback else default."""
    return getattr(cfg, primary, getattr(cfg, fallback, default))

def _dp_has(dp: Any, method_name: str) -> bool: #Check if document processor has a specific method
    return hasattr(dp, method_name) and callable(getattr(dp, method_name))

def _ensure_dir(p: Path) -> Path: #Ensure a directory exists
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_text(path: Path, content: str): #Save text content to a file, ensuring the directory exists
    _ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")

def _save_result(filename: str, data: Any, cfg: Config, is_text: bool = False): #Save result to file (JSON or text), creating results dir if needed.
    """
    Save result to file (JSON or text), creating results dir if needed.
    Supports both cfg.RESULTS_DIR and cfg.results_dir fallbacks.
    """
    results_root = _cfg_attr(cfg, "RESULTS_DIR", "results_dir",
                             default=str(Path("results").resolve()))
    results_dir = Path(results_root)
    _ensure_dir(results_dir)
    fp = results_dir / filename
    try:
        if is_text:
            fp.write_text(str(data), encoding="utf-8")
        else:
            with fp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"   Saved: {fp}")
    except Exception as e:
        print(f"   Error saving {filename}: {e}")

def _generate_demo_report(cfg: Config, dp: DocumentProcessor) -> str: # Generate a demonstration report
    """Markdown report that adapts to available Config fields."""
    model_name = _cfg_attr(cfg, "MODEL_NAME", "model_name", "unknown")
    temperature = _cfg_attr(cfg, "TEMPERATURE", "temperature", "unknown")
    max_tokens = _cfg_attr(cfg, "MAX_TOKENS", "max_tokens", "unknown")

    # Try to get document stats if provided by DP
    if _dp_has(dp, "get_document_stats"):
        doc_stats = dp.get_document_stats()
    else:
        # Fallback: derive some basic stats if dp exposes docs/doc_meta
        num_docs = len(getattr(dp, "docs", []))
        num_meta = len(getattr(dp, "doc_meta", []))
        doc_stats = f"docs indexed: {num_docs}, metadata entries: {num_meta}"

    report = f"""# ResearchGPT Assistant - Demonstration Report

## System Overview
- Configuration: {model_name}
- Temperature: {temperature}
- Max Tokens: {max_tokens}

## Documents Processed
{doc_stats}

## Capabilities Demonstrated

### 1. Document Processing
- PDF text extraction and cleaning
- Text chunking with overlap
- TF-IDF based similarity search
- Document indexing and retrieval

### 2. Advanced Prompting Techniques
- **Chain-of-Thought**: Step-by-step reasoning for complex questions
- **Self-Consistency**: Multiple reasoning paths for robust answers
- **ReAct**: Thought-Action-Observation style research workflow
- **Verification**: Answer quality checking and improvement

### 3. AI Agents
- **Summarizer Agent**: Document/literature summarization
- **QA Agent**: Factual and analytical Q&A
- **Research Workflow Agent**: Multi-step research orchestration
- **Agent Orchestrator**: Multi-agent task routing

## Performance Insights
- Document processing: suitable for academic PDFs
- Search relevance: TF-IDF provides a solid baseline
- Response quality: advanced prompting improves reasoning robustness
- Agent coordination: multi-step workflows executed end-to-end

## Technical Implementation
- Modular Python, configuration-driven
- Error handling and logging
- Easily extensible components

## Next Steps
1. More sophisticated chunking strategies
2. Response caching
3. Quality evaluation metrics
4. Additional specialized agents
5. Support more document formats
6. Batch processing

## Conclusion
This demo integrates:
- Retrieval (TF-IDF), advanced prompting (CoT, Self-Consistency, ReAct)
- LLM-assisted analysis and verification
- Agentic workflows for practical research assistance
"""
    return report


# ------------------------------
# Data bootstrap
# ------------------------------

def auto_download_arxiv_if_empty( #Automatically download papers from arXiv if no PDFs are found
    save_dir: Path,
    query: str = "large language models evaluation",
    max_results: int = 3,
) -> List[Path]:
    """
    Ensure `save_dir` contains PDFs. If empty, fetch recent papers from arXiv.
    Returns a sorted list of PDF paths present after the check.
    """
    _ensure_dir(save_dir)
    existing_pdfs = sorted(save_dir.glob("*.pdf"))
    if existing_pdfs:
        return existing_pdfs

    try:
        import arxiv
    except ImportError as e:
        raise RuntimeError(
            "The 'arxiv' package is not installed. Run 'pip install arxiv' "
            "or add 'arxiv>=2.1.0' to requirements.txt and reinstall."
        ) from e

    print(
        f"[arxiv] No PDFs in {save_dir}. Downloading {max_results} paper(s) for query: '{query}'…"
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client()
    downloaded: List[Path] = []

    for result in client.results(search):
        short_id = result.get_short_id()
        safe_title = "".join(
            c for c in result.title if c.isalnum() or c in (" ", "_", "-")
        ).strip().replace(" ", "_")
        filename = f"{short_id}_{safe_title[:60]}.pdf" if safe_title else f"{short_id}.pdf"
        pdf_path = save_dir / filename
        try:
            print(f"  → {result.title}")
            result.download_pdf(filename=pdf_path)
            downloaded.append(pdf_path)
        except Exception as e:
            print(f"    [Warn] Failed to download '{result.title}': {e}")

    final_pdfs = sorted(save_dir.glob("*.pdf"))
    if not final_pdfs:
        print("[arxiv] No PDFs were downloaded. You can manually place PDFs in the folder and rerun.")
    else:
        print(f"[arxiv] Downloaded {len(downloaded)} file(s) to {save_dir}")
    return final_pdfs


# ------------------------------
# Main demo
# ------------------------------

def main():
    print("=== ResearchGPT Assistant Demo ===")
    print("[main] starting")
    cfg = Config()
    print("[main] config OK")

    dp = DocumentProcessor(cfg)
    print("[main] document processor OK")

    assistant = ResearchGPTAssistant(cfg, dp)
    print("[main] research assistant OK")

    orch = AgentOrchestrator(assistant)
    print("[main] agent orchestrator OK")

    # Resolve papers dir across both config styles
    sample_dir = _cfg_attr(cfg, "SAMPLE_PAPERS_DIR", "data_dir", None)
    if sample_dir is None:
        sample_dir = Path("data") / "sample_papers"
    else:
        # If SAMPLE_PAPERS_DIR is a string, treat as path; if data_dir, append folder
        sample_dir = Path(sample_dir) if isinstance(sample_dir, str) else (Path(sample_dir) / "sample_papers")
    sample_dir = Path(sample_dir)

    print("\n1. Initializing & ensuring sample documents...")
    pdfs = auto_download_arxiv_if_empty(
        save_dir=sample_dir,
        query="large language models evaluation",
        max_results=3,
    )

    # Show absolute paths & sizes
    for p in pdfs:
        try:
            size_kb = max(1, p.stat().st_size // 1024)
            print(f"PDF ready: {p.resolve()}  ({size_kb} KB)")
        except Exception as e:
            print(f"[Warn] Could not stat {p}: {e}")

    if not pdfs:
        print(f"[main] No PDFs available in {sample_dir}. Add PDFs and rerun.")
        return

    print("\n2. Processing sample documents...")
    # Support either dp.add_pdf(...) or dp.process_document(...)
    doc_ids: List[str] = []
    for p in pdfs:
        print(f"   Processing: {p.name}")
        if _dp_has(dp, "add_pdf"):
            dp.add_pdf(p)
            doc_ids.append(str(p))
        elif _dp_has(dp, "process_document"):
            # Some implementations return a doc_id from processing
            doc_id = dp.process_document(str(p))
            doc_ids.append(str(doc_id))
        else:
            raise RuntimeError(
                "DocumentProcessor must expose either add_pdf(Path) or process_document(path_str)"
            )

    print("\n3. Building search index...")
    if _dp_has(dp, "build_search_index"):
        dp.build_search_index()
    else:
        print("[index] Skipped: build_search_index() not found on DocumentProcessor")
    print(f"[index] docs indexed: {len(getattr(dp, 'docs', []))}, meta entries: {len(getattr(dp, 'doc_meta', []))}")

    # Basic stats if available
    if _dp_has(dp, "get_document_stats"):
        stats = dp.get_document_stats()
        print(f"   Document stats: {stats}")

    print("\n4. Demonstrating basic research capabilities...")
    test_query = "machine learning algorithms"
    print(f"   Testing similarity search with query: '{test_query}'")
    similar_chunks = []
    if _dp_has(dp, "find_similar_chunks"):
        similar_chunks = dp.find_similar_chunks(test_query, top_k=3)
        print(f"   Found {len(similar_chunks)} relevant chunks")
    else:
        print("   Skipped similarity search: find_similar_chunks not available")

    print("\n5. Demonstrating Chain-of-Thought reasoning...")
    cot_query = "What are the main advantages and limitations of deep learning?"
    print(f"   CoT Query: {cot_query}")

    # Prefer the explicit CoT method if present; else fall back to a generic QA that supports flags
    if hasattr(assistant, "chain_of_thought_reasoning"):
        ctx = [t[1] for t in (dp.find_similar_chunks(cot_query, top_k=4) if _dp_has(dp, "find_similar_chunks") else [])]
        cot_answer = assistant.chain_of_thought_reasoning(cot_query, ctx)
        cot_payload = {"query": cot_query, "answer": cot_answer, "context_size": len(ctx)}
    else:
        # Fall back to a generic method as per your TODO script
        cot_resp = assistant.answer_research_question(
            cot_query, use_cot=True, use_verification=False
        )
        cot_payload = cot_resp
    _save_result("cot_response.json", cot_payload, cfg, is_text=False)

    print("\n6. Demonstrating Self-Consistency prompting...")
    sc_query = "How do neural networks learn?"
    print(f"   Self-Consistency Query: {sc_query}")
    sc_ctx = []
    if _dp_has(dp, "find_similar_chunks"):
        sc_ctx = dp.find_similar_chunks(sc_query, top_k=5)
    if hasattr(assistant, "self_consistency_generate"):
        sc_response = assistant.self_consistency_generate(sc_query, [t[1] for t in sc_ctx], num_attempts=3)
        _save_result("self_consistency_response.json", sc_response, cfg, is_text=False)
    else:
        _save_result("self_consistency_response.json",
                     {"note": "self_consistency_generate not available"}, cfg)

    print("\n7. Demonstrating ReAct research workflow...")
    react_query = "What are the current trends in natural language processing?"
    print(f"   ReAct Query: {react_query}")
    if hasattr(assistant, "react_research_workflow"):
        react_result = assistant.react_research_workflow(react_query, max_steps=4, search_k=4)
        # Normalize to dict so we can show number of steps if provided
        react_payload = react_result if isinstance(react_result, dict) else {"answer": react_result}
        _save_result("react_workflow.json", react_payload, cfg, is_text=False)
        steps = len(react_payload.get("workflow_steps", []))
        print(f"   ReAct workflow completed with {steps} steps")
    else:
        print("   Skipped: react_research_workflow not available")

    print("\n8. Demonstrating AI Agents...")
    # Summarizer Agent
    if doc_ids:
        print("   Testing Summarizer Agent…")
        # Use underlying path as doc_id when we populated via add_pdf
        first_doc_id = Path(doc_ids[0]).stem
        summary_task = {"doc_id": doc_ids[0]}
        summary_result = orch.route_task("summarizer", summary_task)
        _save_result("document_summary.json", summary_result, cfg, is_text=False)

    # QA Agent
    print("   Testing QA Agent…")
    qa_task = {"question": "What methodology was used in the research?", "type": "analytical"}
    qa_result = orch.route_task("qa", qa_task)
    _save_result("qa_response.json", qa_result, cfg, is_text=False)

    # Research Workflow Agent
    print("   Testing Research Workflow Agent…")
    workflow_task = {"research_topic": "artificial intelligence applications"}
    workflow_result = orch.route_task("workflow", workflow_task)
    _save_result("research_workflow.json", workflow_result, cfg, is_text=False)

    print("\n9. Demonstrating answer verification...")
    test_answer = "Neural networks are computational models inspired by biological neural networks."
    test_query_for_verification = "What are neural networks?"
    if hasattr(assistant, "verify_and_edit_answer"):
        # Build a small context if possible
        vctx = [t[1] for t in (dp.find_similar_chunks(test_query_for_verification, top_k=4) if _dp_has(dp, "find_similar_chunks") else [])]
        verification_result = assistant.verify_and_edit_answer(test_answer, test_query_for_verification, vctx or "Sample context")
        _save_result("verification_result.json", verification_result, cfg, is_text=False)
        print("   Answer verification completed")
    else:
        print("   Skipped: verify_and_edit_answer not available")

    print("\n10. Generating final demonstration report...")
    final_report = _generate_demo_report(cfg, dp)
    _save_result("demo_report.md", final_report, cfg, is_text=True)

    # Also save step-6 style plain outputs to your original results/analyses dir if present
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analyses_dir = Path(_cfg_attr(cfg, "results_analyses_dir", "RESULTS_ANALYSES_DIR",
                                  default=Path("results") / "analyses"))
    _ensure_dir(analyses_dir)
    # Persist CoT/ReAct answers if we have them
    if "cot_payload" in locals():
        save_text(analyses_dir / f"step6_chain_of_thought_{timestamp}.txt",
                  cot_payload["answer"] if isinstance(cot_payload, dict) and "answer" in cot_payload else str(cot_payload))
    if "react_payload" in locals():
        save_text(analyses_dir / f"step6_react_{timestamp}.txt",
                  json.dumps(react_payload, indent=2, ensure_ascii=False))

    # Step 7 style: summarize each doc to results/summaries/
    print("\n=== Step 7: Agents (document summaries) ===")
    summaries_dir = Path(_cfg_attr(cfg, "results_summaries_dir", "RESULTS_SUMMARIES_DIR",
                                   default=Path("results") / "summaries"))
    _ensure_dir(summaries_dir)
    meta_list = getattr(dp, "doc_meta", [])
    doc_paths = sorted({(m["path"] if isinstance(m, dict) and "path" in m else None) for m in meta_list})
    doc_paths = [p for p in doc_paths if p]

    if not doc_paths and doc_ids:
        # fallback: use ingested paths we tracked
        doc_paths = doc_ids

    if not doc_paths:
        print("[step7] No documents found in index for summarization.")
    else:
        for doc_path in doc_paths:
            print(f"[step7] summarizing: {doc_path}")
            result = orch.route_task("summarizer", {"doc_id": doc_path})
            if "summary" not in result:
                print(f"[step7] Failed to summarize {doc_path}: {result}")
                continue
            out_path = summaries_dir / f"summary_{Path(doc_path).stem}.txt"
            out_path.write_text(result["summary"], encoding="utf-8")
            print(f"[step7] Saved summary → {out_path}")

    print("\n=== Demo Complete ===")
    results_root = _cfg_attr(cfg, "RESULTS_DIR", "results_dir",
                             default=str(Path("results").resolve()))
    print(f"Results saved in: {results_root}")
    print("\nCheck the following files for detailed results:")
    print("- cot_response.json (Chain-of-Thought reasoning)")
    print("- self_consistency_response.json (Self-Consistency prompting)")
    print("- react_workflow.json (ReAct workflow)")
    print("- document_summary.json (Document summarization)")
    print("- research_workflow.json (Complete research workflow)")
    print("- qa_response.json (QA agent)")
    print("- verification_result.json (Verification)")
    print("- demo_report.md (Final demonstration report)")


if __name__ == "__main__":
    print("=== LAUNCHING main.py ===")
    print("File path:", Path(__file__).resolve())
    print("CWD:", Path.cwd().resolve())
    try:
        main()
        print("=== main() finished successfully ===")
    except SystemExit as se:
        print(f"[SystemExit] code={se.code}")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception:
        print("=== Unhandled exception in main() ===")
        traceback.print_exc()
