# Contract Compliance Agent — Getting Started Guide

Welcome! This guide will walk you through setting up and running the Contract Compliance Agent on your local machine. By the end, you'll be able to analyze contracts and generate compliance reports entirely on your own infrastructure.

---

## Prerequisites

Before you begin, make sure your system meets these requirements.

**Operating System**: Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)

**Hardware**: Minimum 8GB RAM. A GPU with 8GB+ VRAM is recommended for faster inference but not required.

**Software**: Python 3.10 or higher must be installed. You can verify this by opening a terminal and running `python --version`. If Python is not installed, download it from [python.org](https://www.python.org/downloads/) and ensure you check "Add Python to PATH" during installation.

---

## Step 1: Install Ollama

Ollama is the local inference engine that runs the AI models. The agent requires Ollama to be installed and running.

### Windows

Download the installer from [ollama.ai/download](https://ollama.ai/download) and run `OllamaSetup.exe`. Follow the installation wizard and click "Install". Once complete, Ollama will start automatically and you'll see a llama icon in your system tray (bottom-right corner).

### macOS / Linux

Open a terminal and run:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

After installation, start Ollama by running `ollama serve` in a terminal window (keep this window open).

### Verify Ollama Installation

Open a new terminal and run:

```bash
ollama --version
```

**Expected output**: A version number like `ollama version 0.1.x`

---

## Step 2: Download a Language Model

The agent needs a language model to perform analysis. We recommend starting with a smaller model that works well on most hardware.

Open a terminal and run:

```bash
ollama pull qwen2.5:3b
```

**Expected output**: A progress bar showing the download. This will download approximately 2GB. When complete, you'll see `success`.

Alternative models you can use:

| Model | Command | Size | Notes |
|-------|---------|------|-------|
| Qwen 2.5 3B | `ollama pull qwen2.5:3b` | ~2 GB | Good balance of speed and quality |
| Llama 3.1 8B | `ollama pull llama3.1:8b` | ~4.7 GB | Better quality, requires more RAM |
| Phi-3 Mini | `ollama pull phi3:mini` | ~2.3 GB | Fastest, good for testing |

### Verify Model Installation

```bash
ollama list
```

**Expected output**: A table showing your installed models, including `qwen2.5:3b` (or whichever you downloaded).

---

## Step 3: Clone the Repository

Now you need to download the agent's source code to your machine.

Open a terminal and navigate to where you want to store the project, then clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/scientific-prototypes-private.git
cd scientific-prototypes-private/compliance-agent
```

Replace `YOUR_USERNAME` with the actual GitHub username of the repository owner.

**Expected result**: A new folder called `scientific-prototypes-private` containing the project files.

---

## Step 4: Create a Virtual Environment

A virtual environment isolates the project's dependencies from your system Python installation. This prevents conflicts with other Python projects.

Make sure you're in the `compliance-agent` directory, then run:

```bash
python -m venv .venv
```

**Expected result**: A new folder called `.venv` appears in the project directory.

---

## Step 5: Activate the Virtual Environment

Before installing dependencies or running the agent, you must activate the virtual environment.

### Windows (Command Prompt)

```cmd
.venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

If you get a security error in PowerShell, run this first: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS / Linux

```bash
source .venv/bin/activate
```

**Expected result**: Your terminal prompt now shows `(.venv)` at the beginning, indicating the virtual environment is active.

---

## Step 6: Install Dependencies

With the virtual environment active, install the required Python packages:

```bash
pip install -r requirements.txt
```

**Expected output**: Several packages will be downloaded and installed. The process ends with `Successfully installed ...` followed by a list of package names.

---

## Step 7: Verify the Setup

Let's make sure everything is working correctly before analyzing contracts.

### Check Ollama Connection

```bash
python main.py status
```

**Expected output**:

```
Checking Ollama Status...
✓ Ollama is running

Available Models:
  • qwen2.5:3b
```

If you see "Ollama is not running", make sure the Ollama application is running (check your system tray on Windows, or run `ollama serve` on macOS/Linux).

### Check Agent Version

```bash
python main.py version
```

**Expected output**:

```
Contract Compliance Agent v0.1.0

A local-first AI agent for contract compliance analysis.
```

### View Available Compliance Rules

```bash
python main.py list-rules
```

**Expected output**: A formatted list of all clause categories the agent checks for, including their risk levels and whether they're required. You should see clauses like "Termination Rights", "Limitation of Liability", "Confidentiality", etc.

---

## Step 8: Analyze Your First Contract

The repository includes sample contracts for testing. Let's analyze one.

### Basic Analysis

```bash
python main.py analyze -c samples/nda_sample.txt -m qwen2.5:3b
```

**Expected output**: After 1 minutes to 8 minutes of processing, you'll see a formatted compliance report showing the overall score, risk level, and clause-by-clause analysis. The NDA sample is relatively complete, so you should see a score around 0.65-0.80 with Low or Medium risk.

### Verbose Analysis

For more detailed output during processing, add the `-v` flag:

```bash
python main.py analyze -c samples/nda_sample.txt -m qwen2.5:3b -v
```

**Expected output**: Same as above, but with additional progress information during the analysis.

### Analyze a Different Contract

Try the service agreement sample, which is intentionally incomplete:

```bash
python main.py analyze -c samples/service_agreement.txt -m qwen2.5:3b -v
```

**Expected output**: A lower compliance score (likely 0.40-0.60) with more "Medium Risk" and "High Risk" items identified.

---

## Step 9: Save Reports to Files

Instead of printing to the terminal, you can save reports to files.

### Save as Markdown

```bash
python main.py analyze -c samples/nda_sample.txt -m qwen2.5:3b -o report.md
```

**Expected output**: The message `Report saved to: report.md` and a new file called `report.md` in your current directory. Open it in any text editor or Markdown viewer to see the formatted report.

### Save as JSON

```bash
python main.py analyze -c samples/nda_sample.txt -m qwen2.5:3b -o report.json -f json
```

**Expected output**: The message `Report saved to: report.json` and a new file containing the analysis in JSON format. This is useful for programmatic processing of results.

---

## Step 10: Analyze Your Own Contracts

To analyze your own contracts, simply provide the path to your file:

```bash
python main.py analyze -c path/to/your/contract.pdf -m qwen2.5:3b -v
```

The agent supports `.txt`, `.pdf`, and `.docx` file formats.

### Specify Contract Type

If you know what type of contract you're analyzing, you can provide a hint to improve accuracy:

```bash
python main.py analyze -c contract.pdf -m qwen2.5:3b -t nda -v
```

Available contract types: `general`, `nda`, `service_agreement`, `employment`, `software_license`

---

## Command Reference

Here is a quick reference of all available commands.

### Check System Status

```bash
python main.py status
```

Shows whether Ollama is running and lists available models.

### Show Version

```bash
python main.py version
```

Displays the agent version number.

### List Compliance Rules

```bash
python main.py list-rules
```

Shows all clause categories that will be analyzed.

### Analyze a Contract

```bash
python main.py analyze -c <file> [options]
```

**Required**: `-c` or `--contract` followed by the path to your contract file.

**Optional flags**:

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--model` | Ollama model to use | `llama3.1:8b` |
| `-o`, `--output` | Save report to file | Print to terminal |
| `-f`, `--format` | Output format (markdown, json, text) | `markdown` |
| `-t`, `--type` | Contract type hint | `general` |
| `-v`, `--verbose` | Show detailed progress | Off |
| `-r`, `--rules` | Custom rules YAML file | Default rules |

### Get Help

```bash
python main.py --help
python main.py analyze --help
```

Shows available commands and options.

---

## Running Tests

To verify that all components are working correctly, you can run the automated test suite.

### Run All Tests

```bash
pytest tests/ -v
```

**Expected output**: A list of test names with `PASSED` status for each. All tests should pass.

### Run Unit Tests Only

These tests don't require Ollama to be running:

```bash
pytest tests/test_scorer.py -v
```

**Expected output**: All scoring-related tests pass.

### Run Integration Tests

These tests require Ollama to be running with a model available:

```bash
pytest tests/test_integration.py -v
```

**Expected output**: End-to-end tests pass, confirming the full pipeline works.

---

## Troubleshooting

Here are solutions to common issues you might encounter.

**"Ollama is not running"**: Make sure Ollama is installed and running. On Windows, check for the llama icon in your system tray. On macOS/Linux, run `ollama serve` in a terminal.

**"Model not found"**: You need to download a model first. Run `ollama pull qwen2.5:3b` and wait for it to complete.

**"(.venv) not showing"**: The virtual environment isn't activated. Run the activation command for your operating system (see Step 5).

**"pip: command not found"**: Python might not be installed correctly. Reinstall Python and make sure to check "Add Python to PATH".

**Analysis is very slow**: This is normal when running on CPU only. For faster processing, use a smaller model like `phi3:mini` or install on a machine with an NVIDIA GPU.

**"CUDA out of memory"**: The model is too large for your GPU. Try a smaller model: use `-m phi3:mini` or `-m qwen2.5:3b` instead of larger models.

---

## Next Steps

Now that you have the agent running, here are some things you can explore.

**Customize the rules**: Edit `config/default_rules.yaml` to add or modify the clauses being checked.

**Adjust clause weights**: Modify `config/clause_weights.yaml` to change how different clauses impact the overall score.

**Batch processing**: Write a Python script to analyze multiple contracts at once using the Python API.

**Integration**: Use the JSON output format to integrate contract analysis into your existing workflows.

---

## Support

If you encounter issues not covered in this guide, please reach out to the repository owner or open an issue on GitHub.

---

*This agent is a prototype for educational and demonstration purposes. Always have contracts reviewed by qualified legal professionals before making business decisions.*
