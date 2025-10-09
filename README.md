ResearchGPT Assistant 🧠
ResearchGPT Assistant is an intelligent research tool that leverages advanced AI techniques to help researchers process academic documents, generate insights, and automate research workflows. This project integrates machine learning fundamentals, natural language processing (NLP), advanced prompting strategies, and AI agents into a practical research assistance application.

✨ Features
Core Capabilities
Document Processing: Extract and process text from PDF research papers (via PyPDF2).

Intelligent Search: TF-IDF based similarity search for relevant document retrieval.

Advanced Prompting: Implementations of Chain-of-Thought, Self-Consistency, and ReAct strategies.

AI Agents: Specialized agents for summarization, QA, and research workflow orchestration.

Research Automation: Complete research session management with multi-step workflows.

Advanced Prompting Techniques
Technique	Description
Chain-of-Thought (CoT)	Step-by-step logical reasoning for complex questions.
Self-Consistency	Multiple reasoning paths with consensus-based answers for robustness.
ReAct Workflows	Structured research processes using Thought-Action-Observation cycles.
Verification & Editing	Mechanisms for answer quality checking and improvement.

Export to Sheets
AI Agents
Summarizer Agent: Document and literature overview generation.

QA Agent: Factual and analytical question answering.

Research Workflow Agent: Complete research session orchestration.

Agent Orchestrator: Multi-agent task coordination and routing.

🛠️ Technical Architecture
Technology Stack
Core Language: Python 3.8+

LLM Integration: Mistral API

Machine Learning: scikit-learn (for TF-IDF), pandas, numpy

File Handling: PyPDF2 (for PDF text extraction)

Project Structure (VS Code View)
research_gpt_assistant/
├── README.md                      <-- You are here
├── requirements.txt               # Project dependencies
├── config.py                      # Configuration and API settings
├── document_processor.py          # PDF processing and text extraction logic
├── research_assistant.py          # Main ResearchGPT class
├── research_agents.py             # AI agents implementation
├── main.py                        # Main execution script
├── test_system.py                 # Testing and evaluation script
├── data/
│   ├── sample_papers/             # Place your PDF research papers here
│   └── processed/                 # Extracted text files (auto-generated)
└── prompts/
    └── prompt_templates.txt       # Prompt templates

⚙️ Installation and Setup
Prerequisites
Python 3.8 or higher.

Mistral API key.

Git.

Visual Studio Code (VS Code).

Setup Instructions
Clone the repository:

Bash

git clone https://github.com/yourusername/research-gpt-assistant.git
cd research-gpt-assistant
Open in VS Code:

Bash

code .
Create and Activate Virtual Environment:
Open the VS Code Integrated Terminal (Ctrl+‘), and run the following commands:

Bash

cd "C:\Users\Windows\python\lesson\ai\capstone folder for ai\project_step_by_step"

# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# (On Mac/Linux use: source venv/bin/activate)

Install dependencies:

Bash

pip install -r requirements.txt
pip install --upgrade pip
pip install mistralai PyPDF2 pandas numpy scikit-learn python-dotenv nltk

Configure API Settings:
In your project folder, create a file named .env:

MISTRAL_API_KEY=your_actual_mistral_key_here


This key is loaded automatically in config.py.

Python

Verify installation works:
python -c "import mistralai, PyPDF2, pandas, numpy, sklearn, nltk; print('✅ All packages OK')"

Add Sample Papers:
The PDFs don’t come from Mistral itself. Mistral only provides the language model API that you query with your text + context.
Then, when you ask a question, the system finds the most relevant chunks and sends them to Mistral’s API, so the model can generate an answer.
papers are added with the function def auto_download_arxiv_if_empty line 133-186 in main.py 

▶️ Usage
Basic Run
Run the main demonstration to process documents, build the search index, and test core capabilities:

Bash

python main.py
Run the system tests to evaluate component performance and generate reports:

Bash

python test_system.py
Advanced Usage (Interactive Research)
You can interact with the system directly by running custom Python code, often best done in the VS Code Integrated Terminal or a temporary scratchpad file.

Initialize the system and process documents:

Python

from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant

config = Config()
doc_processor = DocumentProcessor(config)
assistant = ResearchGPTAssistant(config, doc_processor)

# Process all papers in the sample directory (or a specific paper)
# doc_processor.process_all_documents() 
# OR: 
doc_processor.process_document("data/sample_papers/my_paper.pdf")
doc_processor.build_search_index()
Ask a Research Question (with Advanced Reasoning):

Python

response = assistant.answer_research_question(
    "What are the main limitations of current machine learning approaches?",
    use_cot=True,
    use_verification=True
)
print(response['answer'])
Use AI Agents for Workflow Automation:

Python

from research_agents import AgentOrchestrator
orchestrator = AgentOrchestrator(assistant)

# Summarize a document by its ID (returned from process_document)
summary = orchestrator.route_task('summarizer', {'doc_id': 'paper_id_from_processing'})

# Conduct a complete research session
research_session = orchestrator.route_task('workflow', {
    'research_topic': 'natural language processing trends'
})
print(research_session['final_answer'])

🔒 Configuration
All key parameters are managed in config.py:

Output locations:
Analyses: results/analyses/

Summaries: results/summaries/

Each run creates timestamped result files for:

Chain-of-Thought

Self-Consistency

ReAct

Verified Answer

QA Agent

Workflow Agent

Summaries (per PDF)

Parameter	Default	Description
MISTRAL_API_KEY	(required)	Your Mistral AI API Key.
MODEL_NAME	"mistral-medium"	The Mistral model to use.
TEMPERATURE	0.1	Response randomness (0.0=deterministic to 1.0=creative).
CHUNK_SIZE	1000	Text chunk size for document processing.
OVERLAP	100	Overlap between text chunks.

Examples: 
see analyze_results.ipynb
cd "C:\Users\Windows\python\lesson\ai\capstone folder for ai\project_step_by_step"
venv\Scripts\activate
pip install matplotlib jupyter
jupyter notebook
Click “New” → “Python 3 (ipykernel)”



Export to Sheets
📝 License
This project is created for educational purposes as part of an AI/ML capstone project.

Acknowledgments:
This project demonstrates the practical application of Machine Learning, NLP, advanced LLM prompting, and AI agent architecture.