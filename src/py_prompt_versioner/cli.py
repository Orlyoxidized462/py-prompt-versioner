import typer
from pathlib import Path
from rich import print
from typing import Optional

app = typer.Typer(help="Prompt-Versioner: Manage your AI prompts like code.")

@app.command()
def init(
    path: Path = typer.Option(
        Path("prompts"), 
        "--path", 
        "-p", 
        help="The directory where your prompts will be stored."
    )
):
    """
    Initialize a new Prompt-Versioner environment at the specified path.
    """
    # Create the base directory and a sample sub-directory
    sample_dir = path / "sample_task"
    version_file = sample_dir / "v1.md"

    try:
        # Check if the path exists but isn't a directory
        if path.exists() and not path.is_dir():
            print(f"[red]Error:[/red] '{path}' exists but is not a directory.")
            raise typer.Exit(code=1)

        # Create directories (parents=True allows nested paths like 'data/prompts')
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a boilerplate prompt file if it doesn't exist
        if not version_file.exists():
            starter_content = (
                "---\n"
                "version: v1\n"
                "model: gpt-4o\n"
                "temperature: 0.7\n"
                "---\n\n"
                "Act as a professional editor. Summarize the following text: {{ text }}"
            )
            version_file.write_text(starter_content)
            print(f"[green]✔[/green] Created prompt structure at: [bold]{sample_dir}[/bold]")
        else:
            print(f"[yellow]![/yellow] Sample prompt already exists at: {version_file}")

        print("\n[bold]Success![/bold] You can now load your first prompt using:")
        print(f"[blue]pm = PromptManager(path='{path}')[/blue]")

    except Exception as e:
        print(f"[red]Failed to initialize:[/red] {e}")
        raise typer.Exit(code=1)

@app.command()
def version():
    """Show the current version of Prompt-Versioner."""
    print("Prompt-Versioner [bold]v0.1.0[/bold]")

if __name__ == "__main__":
    app()