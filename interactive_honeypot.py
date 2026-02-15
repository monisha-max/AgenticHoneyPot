
import asyncio
import uuid
import sys
import logging
from datetime import datetime, timezone
from app.core.orchestrator import ConversationOrchestrator
from app.api.schemas import Message, SenderType
import rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

console = Console()

# Configure logging for CLI runs (main.py logging config is not used here)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def interactive_cli():
    console.print(Panel("[bold cyan]Agentic HoneyPot - Interactive CLI Testing[/bold cyan]\n[italic]Type 'exit' or 'quit' to end the session.[/italic]", border_style="cyan"))
    
    orchestrator = ConversationOrchestrator()
    session_id = str(uuid.uuid4())[:8]
    history = []
    
    console.print(f"[bold yellow]Session ID:[/bold yellow] {session_id}")
    console.print("-" * 50)
    
    while True:
        try:
            # Scammer Input
            scammer_text = console.input("[bold red]Scammer > [/bold red]")
            
            if scammer_text.lower() in ["exit", "quit"]:
                break
                
            if not scammer_text.strip():
                continue

            message = Message(
                sender=SenderType.SCAMMER,
                text=scammer_text,
                timestamp=datetime.now(timezone.utc)
            )

            # Process Message
            with console.status("[bold green]Agent is thinking...[/bold green]"):
                response_text = await orchestrator.process_message(
                    session_id=session_id,
                    message=message,
                    conversation_history=history
                )
            
            # Print Agent Response
            console.print(f"[bold blue]Agent   > [/bold blue]{response_text}")
            
            # Add to local history for display/tracking
            history.append(message)
            history.append(Message(
                sender=SenderType.USER,
                text=response_text,
                timestamp=datetime.now(timezone.utc)
            ))

            # Fetch current state to show extracted intel and notes
            state = await orchestrator.session_manager.get_session(session_id)
            
            # Display Intelligence Dashboard
            if state.intelligence:
                intel = state.intelligence
                table = Table(title="Intelligence & State", show_header=True, header_style="bold magenta", box=rich.box.ROUNDED, expand=True)
                table.add_column("Category", style="cyan", width=20)
                table.add_column("Values", style="white")
                
                if intel.bank_accounts: table.add_row("Bank Accounts", ", ".join(intel.bank_accounts))
                if intel.upi_ids: table.add_row("UPI IDs", ", ".join(intel.upi_ids))
                if intel.phone_numbers: table.add_row("Phone Numbers", ", ".join(intel.phone_numbers))
                if intel.phishing_links: table.add_row("Phishing Links", ", ".join(intel.phishing_links))
                if intel.email_addresses: table.add_row("Email Addresses", ", ".join(intel.email_addresses))
                # if intel.suspicious_keywords: table.add_row("Keywords", ", ".join(intel.suspicious_keywords[:5]))
                
                table.add_row("Current Persona", str(state.persona.value), style="bold yellow")
                table.add_row("Phase", state.conversation_phase.value)
                table.add_row("Scam Detected", "[bold green]YES[/bold green]" if state.scam_detected else "[bold red]NO[/bold red]")
                
                console.print(table)

            # Display Recent Agent Notes (Tactics/Reasoning)
            if state.agent_notes:
                notes_text = "\n".join([f"• {note}" for note in state.agent_notes[-3:]]) # Show last 3 notes
                console.print(Panel(notes_text, title="[bold yellow]Agent Tactical Notes[/bold yellow]", border_style="yellow"))
            
            console.print("-" * 50)

            # Check if session completed
            if state.conversation_phase.value == "COMPLETE":
                console.print("[bold green]SESSION COMPLETED.[/bold green]")
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

    # Final Payload Display on Exit
    console.print("\n" + "="*50)
    console.print("[bold cyan]GENERATING FINAL EXTRACTION PAYLOAD...[/bold cyan]")
    try:
        state = await orchestrator.session_manager.get_session(session_id)
        payload = orchestrator.callback_manager._build_payload(state)
        
        import json
        payload_json = json.dumps(payload.dict(), indent=2)
        
        console.print(Panel(
            payload_json, 
            title="[bold green]GUVI Callback Payload[/bold green]", 
            subtitle="Final Extraction Result",
            border_style="green",
            padding=(1, 2)
        ))
    except Exception as e:
        console.print(f"[bold red]Could not generate payload:[/bold red] {e}")

    console.print("[bold cyan]Interactive Session Ended.[/bold cyan]")

if __name__ == "__main__":
    asyncio.run(interactive_cli())
