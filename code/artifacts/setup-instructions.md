# RAG Hackathon Project Setup Instructions

## Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eft-hackathon/hackathon2-ai-explorers.git
   cd hackathon2-ai-explorers
   ```

2. **Install Python (recommended: 3.10+):**
   Ensure you have Python installed. You can check with:
   ```bash
   python --version
   ```

3. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # Or: source venv/bin/activate  # On Mac/Linux
   ```

4. **Install dependencies:**
   ```bash
   pip install -r code/src/requirements.txt
   ```

5. **Set up environment variables:**
   - Create a `.env` file in `code/src/` and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-key-here
     ```

---

## Running the Solution

### Option 1: Full Pipeline (all questions)
```bash
python code/src/main.py
```
- Output: `code/artifacts/results.json`
- Logs: `code/artifacts/pipeline.log`

### Option 2: User-Driven Questions (interactive)
```bash
python code/src/cli.py
```
- Paste one or more questions (multi-line, end with blank line)
- Output: `code/artifacts/results.json`

### Option 3: Batch Mode (all questions from file)
```bash
python code/src/cli.py --batch
```
- Processes all questions from `code/src/data/questions.csv`
- Output: `code/artifacts/results.json`

---

## Output Files
- Results: `code/artifacts/results.json` (always overwritten)
- Logs: `code/artifacts/pipeline.log`

## Testing
Run all tests to verify code integrity:
```bash
python -m pytest code/tests/
```

## Troubleshooting
- Ensure your API keys are valid and you have internet access for embedding/LLM calls.
- If you encounter errors, check dependencies and Python version.
- If logging fails, ensure the `code/artifacts` directory exists.

---
For architecture and design details, see:
- [Architecture](code/artifacts/arch/architecture.md)
- [Design Overview](code/artifacts/design-overview.json)
