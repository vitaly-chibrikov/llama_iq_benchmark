# IQ benchmark for Instruction Tuned LLaMA-family

This project aims to design a systematic benchmark for assessing the reasoning and instruction-following capabilities of offline, instruction-tuned models in the LLaMA family (e.g., Mistral-7B-Instruct, LLaMA-2-Chat, LLaMA-3-Instruct). 

## TODO

#### Design and Compile a Question Set
Collect hundreds (ideally thousands) of diverse prompts, systematically categorized by:
- **Topic/domain**: e.g., science, history, mathematics, programming
- **Difficulty level**: basic, intermediate, advanced
- **Linguistic clarity**: well-formed vs. ambiguous questions

#### Develop an Automated Evaluation Tool
Build a Python application that sends question-answer pairs to an LLM (e.g., ChatGPT via API) and requests ratings along the following dimensions:
- **Clarity** of the response
- **Precision** and factual correctness
- **Completeness** and informativeness
- **Readability** and overall coherence

#### Run Controlled Benchmarking Experiments
Generate answers to the question set using multiple offline instruction-tuned LLaMA-family models (e.g., Mistral, LLaMA-2, LLaMA-3), and measure:
- **Generation time** per question
- **Answer quality** as scored by the evaluator model

#### Verify Evaluation Consistency
Ensure the scoring model yields stable and interpretable results by:
- **Re-scoring** the same answers multiple times
- **Comparing** results across prompt variants
- **Analyzing** score variance and inter-model ranking consistency

#### Document and Analyze Results
Summarize findings, compare models by both performance and quality, and identify:
- Domain-specific strengths and weaknesses
- Trade-offs between model size, quantization, speed, and output quality

---

## Current state

In current state it is a Python application that loads a list of questions from a text file, queries a local Large Language Model (LLM) such as **Mistral** or **LLaMA** using `llama-cpp-python`, and saves the answers along with response times into a structured CSV report.

### Features

- Runs **offline** using quantized `.gguf` LLM files
- Accepts multiple questions (paragraphs) in a plain text file
- Measures and logs generation time per response
- Saves results as a clean CSV file: `Index, Question, Answer, Time[ms]`
- Configurable via `parameters.ini`
- Supports Apple Silicon GPU acceleration via Metal (if configured)

---

### Requirements

- Python 3.8+
- A compatible `.gguf` model file (e.g., Mistral 7B Instruct, LLaMA 2 Chat)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Required Python packages:

```bash
pip install pandas configparser llama-cpp-python
