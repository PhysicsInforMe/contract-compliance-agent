#!/usr/bin/env python3
"""
Contract Compliance Agent - Command Line Interface

Analyze contracts for compliance using local AI models.

Usage:
    python main.py --contract path/to/contract.pdf
    python main.py --contract contract.docx --output report.md --verbose
    python main.py --contract contract.txt --model qwen2.5:7b --format json
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent import ComplianceAgent
from models import ContractType
from tools import check_ollama_status


# =============================================================================
# CLI Setup
# =============================================================================

app = typer.Typer(
    name="compliance-agent",
    help="AI-powered contract compliance analysis using local LLMs.",
    add_completion=False,
)

console = Console()


# =============================================================================
# Main Commands
# =============================================================================

@app.command()
def analyze(
    contract: Path = typer.Option(
        ...,
        "--contract", "-c",
        help="Path to the contract file (PDF, DOCX, or TXT)",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path for the report",
    ),
    model: str = typer.Option(
        "llama3.1:8b",
        "--model", "-m",
        help="Ollama model to use for analysis",
    ),
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format: markdown, json, or text",
    ),
    rules: Optional[Path] = typer.Option(
        None,
        "--rules", "-r",
        help="Path to custom compliance rules YAML",
    ),
    contract_type: str = typer.Option(
        "general",
        "--type", "-t",
        help="Contract type: general, nda, service_agreement, employment, software_license",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
):
    """
    Analyze a contract for compliance issues.
    
    Examples:
        python main.py --contract sample.pdf
        python main.py -c contract.docx -o report.md -v
        python main.py -c agreement.txt -m qwen2.5:7b -f json
    """
    # Map contract type string to enum
    type_map = {
        "general": ContractType.GENERAL,
        "nda": ContractType.NDA,
        "service_agreement": ContractType.SERVICE_AGREEMENT,
        "employment": ContractType.EMPLOYMENT,
        "software_license": ContractType.SOFTWARE_LICENSE,
    }
    ct = type_map.get(contract_type.lower(), ContractType.GENERAL)
    
    try:
        # Initialize agent
        with ComplianceAgent(
            model=model,
            rules_path=rules,
            verbose=verbose
        ) as agent:
            
            # Show progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                
                # Check Ollama
                progress.add_task("Checking Ollama status...", total=None)
                status = agent.check_status()
                
                if not status["available"]:
                    console.print("[red]Error:[/red] Ollama is not running.")
                    console.print("Start Ollama with: [cyan]ollama serve[/cyan]")
                    raise typer.Exit(1)
                
                if not status["model_available"]:
                    console.print(f"[yellow]Warning:[/yellow] Model '{model}' not found.")
                    console.print(f"Pull it with: [cyan]ollama pull {model}[/cyan]")
                    raise typer.Exit(1)
                
                # Run analysis
                task = progress.add_task(
                    f"Analyzing {contract.name}...",
                    total=None
                )
                
                result = agent.analyze(contract, contract_type=ct)
                progress.remove_task(task)
            
            # Output results
            if output:
                agent.save_report(result, output, format=format)
                console.print(f"[green]Report saved to:[/green] {output}")
            else:
                # Print to terminal
                agent.print_report(result)
            
            # Summary
            score = result.compliance_score
            risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
            risk_color = risk_colors.get(score.risk_level.value, "white")
            
            console.print()
            console.print(f"[bold]Overall Score:[/bold] {score.overall_score:.2f}")
            console.print(f"[bold]Risk Level:[/bold] [{risk_color}]{score.risk_level.value.upper()}[/{risk_color}]")
            console.print(f"[bold]Processing Time:[/bold] {result.processing_time_seconds:.2f}s")
            
    except ConnectionError as e:
        console.print(f"[red]Connection Error:[/red] {e}")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]File Not Found:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def status():
    """
    Check Ollama status and available models.
    """
    console.print("[bold]Checking Ollama Status...[/bold]")
    
    status = check_ollama_status()
    
    if status["available"]:
        console.print("[green]✓[/green] Ollama is running")
        console.print()
        console.print("[bold]Available Models:[/bold]")
        for model in status.get("models", []):
            console.print(f"  • {model}")
        
        if not status["models"]:
            console.print("  [yellow]No models installed[/yellow]")
            console.print("  Install a model with: [cyan]ollama pull llama3.1:8b[/cyan]")
    else:
        console.print("[red]✗[/red] Ollama is not running")
        console.print()
        console.print("Start Ollama:")
        console.print("  [cyan]ollama serve[/cyan]")
        console.print()
        console.print("Or install from: https://ollama.ai/download")


@app.command()
def list_rules():
    """
    List available compliance rules.
    """
    from config import load_rules
    
    rules = load_rules()
    clauses = rules.get("clauses", [])
    
    console.print("[bold]Compliance Rules[/bold]")
    console.print()
    
    for clause in clauses:
        required = "[green]Required[/green]" if clause.get("required") else "[dim]Optional[/dim]"
        risk = clause.get("risk_if_missing", "medium")
        risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
        risk_color = risk_colors.get(risk, "white")
        
        console.print(f"[bold]{clause['name']}[/bold] ({clause['id']})")
        console.print(f"  Status: {required} | Risk if missing: [{risk_color}]{risk.upper()}[/{risk_color}]")
        console.print(f"  {clause.get('description', 'No description')}")
        console.print()


@app.command()
def version():
    """
    Show version information.
    """
    from agent import __version__
    
    console.print(f"[bold]Contract Compliance Agent[/bold] v{__version__}")
    console.print()
    console.print("A local-first AI agent for contract compliance analysis.")
    console.print()
    console.print("GitHub: https://github.com/yourusername/contract-compliance-agent")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    app()
