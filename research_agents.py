"""
AI Research Agents for Specialized Tasks
"""

from typing import List, Dict, Any

from research_assistant import ResearchGPTAssistant


# ---------- Base Agent ----------
class BaseAgent:
    def __init__(self, research_assistant: ResearchGPTAssistant):
        """
        Base class for all research agents
        """
        self.assistant = research_assistant
        self.doc_processor = research_assistant.dp
        self.agent_name = "BaseAgent"
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Base method for executing agent tasks
        """
        raise NotImplementedError("Each agent must implement execute_task method")


# ---------- Summarizer Agent ----------
class SummarizerAgent(BaseAgent):
    def __init__(self, research_assistant: ResearchGPTAssistant):
        super().__init__(research_assistant)
        self.agent_name = "SummarizerAgent"
        self.summary_length = 200  # default target length in words

    def summarize_document(self, doc_id: str) -> Dict[str, Any]:
        # Collect all chunks that belong to this PDF (match by path)
        indices = [i for i, meta in enumerate(self.doc_processor.doc_meta) if meta["path"] == doc_id]
        if not indices:
            return {"error": f"Document not found in index: {doc_id}"}

        # Join a limited number of chunks for efficiency
        chunks = [self.doc_processor.docs[i] for i in indices[:10]]
        text = "\n".join(chunks)

        summary_prompt = f"""
        Summarize the following academic paper.
        Extract explicitly:
        - Research question or hypothesis
        - Methodology
        - Key findings
        - Conclusion
        - Limitations

        Text:
        {text}
        """

        summary = self.assistant._chat([{"role": "user", "content": summary_prompt}], temperature=0.3)

        return {
            "doc_id": doc_id,
            "summary": summary,
            "word_count": len(summary.split()),
            "key_topics": []
        }

    
    def create_literature_overview(self, doc_ids: List[str]) -> Dict[str, Any]:
        individual_summaries = [self.summarize_document(doc_id) for doc_id in doc_ids]

        overview_prompt = f"""
        Create a literature overview from the following summaries.
        Identify:
        - Common research themes
        - Different methodological approaches
        - Consistent findings vs contradictions
        - Research gaps
        - Suggested future directions

        Summaries:
        {individual_summaries}
        """

        overview = self.assistant._chat(
            [{"role": "user", "content": overview_prompt}],
            temperature=0.3
        )
        
        return {
            "overview": overview,
            "papers_analyzed": len(doc_ids),
            "individual_summaries": individual_summaries
        }
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "doc_id" in task_input:
            return self.summarize_document(task_input["doc_id"])
        elif "doc_ids" in task_input:
            return self.create_literature_overview(task_input["doc_ids"])
        return {"error": "Invalid task input for SummarizerAgent"}


# ---------- QA Agent ----------
class QAAgent(BaseAgent):
    def __init__(self, research_assistant: ResearchGPTAssistant):
        super().__init__(research_assistant)
        self.agent_name = "QAAgent"
    
    def answer_factual_question(self, question: str) -> Dict[str, Any]:
        relevant_chunks = self.doc_processor.find_similar_chunks(question, top_k=3)
        ctx = [c[1] for c in relevant_chunks]

        qa_prompt = f"""
        Answer the following factual research question concisely.
        Cite relevant snippets as [S1], [S2], etc.

        Question: {question}

        Context:
        {ctx}
        """
        answer = self.assistant._chat([{"role": "user", "content": qa_prompt}])

        return {
            "question": question,
            "answer": answer,
            "sources": [c[2] for c in relevant_chunks],
            "confidence": "high" if answer else "low"
        }
    
    def answer_analytical_question(self, question: str) -> Dict[str, Any]:
        ctx = [c[1] for c in self.doc_processor.find_similar_chunks(question, top_k=5)]
        response = self.assistant.chain_of_thought_reasoning(question, ctx)

        return {
            "question": question,
            "analysis": response,
            "reasoning_type": "chain_of_thought"
        }
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        question = task_input.get("question", "")
        qtype = task_input.get("type", "factual")
        if qtype == "analytical":
            return self.answer_analytical_question(question)
        return self.answer_factual_question(question)


# ---------- Research Workflow Agent ----------
class ResearchWorkflowAgent(BaseAgent):
    def __init__(self, research_assistant: ResearchGPTAssistant):
        super().__init__(research_assistant)
        self.agent_name = "ResearchWorkflowAgent"
        self.summarizer = SummarizerAgent(research_assistant)
        self.qa_agent = QAAgent(research_assistant)

def _extract_doc_id(doc_chunk):
    """
    Helper to extract doc_id from a doc chunk tuple.
    Assumes doc_chunk is a tuple where the third element is metadata dict with 'path' key.
    """
    if isinstance(doc_chunk, tuple) and len(doc_chunk) > 2 and isinstance(doc_chunk[2], dict):
        return doc_chunk[2].get("path")
    return None
    
    def conduct_research_session(self, research_topic: str) -> Dict[str, Any]:
        results = {
            "research_topic": research_topic,
            "generated_questions": [],
            "document_analysis": {},
            "answers": [],
            "research_gaps": "",
            "future_directions": ""
        }

        # Generate questions
        q_prompt = f"""
        Generate 3–5 specific, answerable research questions
        about the following topic:
        {research_topic}
        """
        gen_qs = self.assistant._chat([{"role": "user", "content": q_prompt}])
        results["generated_questions"] = gen_qs

        # Analyze relevant docs
        relevant_docs = self.doc_processor.find_similar_chunks(research_topic, top_k=8)
        doc_ids = [d for d in { _extract_doc_id(doc) for doc in relevant_docs } if d]
        if doc_ids:
            results["document_analysis"] = self.summarizer.create_literature_overview(doc_ids)

        # Answer generated questions
        if isinstance(gen_qs, str):
            qs = [line.strip("-• ") for line in gen_qs.split("\n") if line.strip()]
            for q in qs:
                ans = self.qa_agent.answer_factual_question(q)
                results["answers"].append(ans)

        # Research gaps
        gap_prompt = f"""
        Based on current findings about "{research_topic}", identify:
        - Remaining research gaps
        - Promising future directions
        """
        gaps = self.assistant._chat([{"role": "user", "content": gap_prompt}])
        results["research_gaps"] = gaps

        return results
    
    # In research_agents.py

class ResearchWorkflowAgent(BaseAgent):
    # ... your existing __init__ and other methods (e.g., conduct_research_session) ...

    def execute_task(self, task_input: dict):
        """
        Execute a workflow research task.
        Expects a dict like: {"research_topic": "..."}.
        Falls back to keys 'topic' or 'query' for flexibility.
        """
        if not isinstance(task_input, dict):
            return {"error": "task_input must be a dict"}

        topic = (
            task_input.get("research_topic")
            or task_input.get("topic")
            or task_input.get("query")
        )
        if not topic:
            return {"error": "Missing 'research_topic' in task_input"}

        try:
            # Use your existing workflow method (already present per your earlier stack trace)
            session_result = self.conduct_research_session(topic)
            # Normalize to a dict if your method returns a string
            if not isinstance(session_result, dict):
                session_result = {"result": session_result, "research_topic": topic}
            return session_result
        except Exception as e:
            return {"error": f"Workflow execution failed: {e}", "research_topic": topic}



# ---------- Orchestrator ----------
class AgentOrchestrator:
    def __init__(self, research_assistant: ResearchGPTAssistant):
        self.assistant = research_assistant
        self.agents = {
            "summarizer": SummarizerAgent(research_assistant),
            "qa": QAAgent(research_assistant),
            "workflow": ResearchWorkflowAgent(research_assistant),
        }
    
    def route_task(self, task_type: str, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if task_type in self.agents:
            return self.agents[task_type].execute_task(task_input)
        return {"error": f"Unknown task type: {task_type}"}
    
    def execute_complex_workflow(self, workflow_description: str) -> Dict[str, Any]:
        # Simple stub: could parse description into steps
        return {
            "workflow_description": workflow_description,
            "steps_executed": ["Not implemented"],
            "final_result": {}
        }
