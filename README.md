# ResearchGPT Assistant 🧠

**ResearchGPT Assistant** is an intelligent research tool that leverages advanced AI techniques to help researchers process academic documents, generate insights, and automate research workflows. This project integrates machine learning fundamentals, natural language processing (NLP), advanced prompting strategies, and AI agents into a practical research assistance application.

---

## ✨ Features

### Core Capabilities

* **Document Processing:** Extract and process text from PDF research papers (via **PyPDF2**).
* **Intelligent Search:** **TF-IDF** based similarity search for relevant document retrieval.
* **Advanced Prompting:** Implementations of Chain-of-Thought, Self-Consistency, and ReAct strategies.
* **AI Agents:** Specialized agents for summarization, QA, and research workflow orchestration.
* **Research Automation:** Complete research session management with multi-step workflows.

### Advanced Prompting Techniques

| Technique | Description |
| :--- | :--- |
| **Chain-of-Thought (CoT)** | Step-by-step logical reasoning for complex questions. |
| **Self-Consistency** | Multiple reasoning paths with consensus-based answers for robustness. |
| **ReAct Workflows** | Structured research processes using Thought-Action-Observation cycles. |
| **Verification & Editing** | Mechanisms for answer quality checking and improvement. |

### AI Agents

* **Summarizer Agent:** Document and literature overview generation.
* **QA Agent:** Factual and analytical question answering.
* **Research Workflow Agent:** Complete research session orchestration.
* **Agent Orchestrator:** Multi-agent task coordination and routing.

---

## 🛠️ Technical Architecture

### Technology Stack

* **Core Language:** Python 3.8+
* **LLM Integration:** Mistral API
* **Machine Learning:** `scikit-learn` (for TF-IDF), `pandas`, `numpy`
* **File Handling:** `PyPDF2` (for PDF text extraction)

### Project Structure (VS Code View)

```bash
research_gpt_assistant/
├── README.md              <-- You are here
├── requirements.txt       # Project dependencies
├── config.py              # Configuration and API settings
├── document_processor.py  # PDF processing and text extraction logic
├── research_assistant.py  # Main ResearchGPT class
├── research_agents.py     # AI agents implementation
├── main.py                # Main execution script
├── test_system.py         # Testing and evaluation script
├── data/
│   ├── sample_papers/     # Place your PDF research papers here
│   └── processed/         # Extracted text files (auto-generated)
└── prompts/
    └── prompt_templates.txt # Prompt templates