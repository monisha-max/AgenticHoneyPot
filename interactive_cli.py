#!/usr/bin/env python3
"""
Interactive CLI for testing the Honeypot API
"""
import requests
import uuid
import time
import os
from datetime import datetime
from typing import Dict, List

API_URL = "http://localhost:8000/api/honeypot-demo"
API_KEY = os.environ.get("API_KEY", "").strip()

# Colors for terminal
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _format_field(label: str, values: List[str]) -> str:
    if not values:
        return f"{label}: -"
    return f"{label}: {', '.join(values)}"


def print_full_intelligence(intel: Dict[str, List[str]]) -> None:
    """Print full extracted intelligence values for debugging."""
    print(f"{CYAN}Full Extracted Intelligence:{RESET}")
    lines = [
        _format_field("Names", intel.get("scammer_names", [])),
        _format_field("Phones", intel.get("phone_numbers", [])),
        _format_field("UPIs", intel.get("upi_ids", [])),
        _format_field("Emails", intel.get("email_addresses", [])),
        _format_field("Bank Accounts", intel.get("bank_accounts", [])),
        _format_field("IFSC Codes", intel.get("ifsc_codes", [])),
        _format_field("References", intel.get("fake_references", [])),
        _format_field("Links", intel.get("phishing_links", [])),
        _format_field("Keywords", intel.get("suspicious_keywords", [])),
    ]
    for line in lines:
        print(f"  - {line}")


def print_callback_intelligence(result: dict) -> None:
    """Print final callback payload intelligence (5 keys)."""
    payload = result.get("guviPayload") or {}
    extracted = payload.get("extractedIntelligence") or {}
    if not extracted:
        return
    print(f"{CYAN}Final Callback Intelligence:{RESET}")
    lines = [
        _format_field("bankAccounts", extracted.get("bankAccounts", [])),
        _format_field("upiIds", extracted.get("upiIds", [])),
        _format_field("phishingLinks", extracted.get("phishingLinks", [])),
        _format_field("phoneNumbers", extracted.get("phoneNumbers", [])),
        _format_field("suspiciousKeywords", extracted.get("suspiciousKeywords", [])),
    ]
    for line in lines:
        print(f"  - {line}")

def print_header():
    print(f"\n{BOLD}{CYAN}=" * 60)
    print("   AGENTIC HONEYPOT - Interactive Testing CLI")
    print("=" * 60 + RESET)
    print(f"{YELLOW}Type 'quit' to exit, 'new' for new session{RESET}\n")

def send_message(session_id: str, message: str, history: list) -> dict:
    """Send message to API and return response"""
    payload = {
        "message": message,
        "senderId": session_id,
        "messageId": str(uuid.uuid4()),
        "timestamp": int(datetime.now().timestamp()),
        "conversationHistory": history
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    try:
        start = time.perf_counter()
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        elapsed = time.perf_counter() - start
        result = response.json()
        result["_elapsed_secs"] = round(elapsed, 3)
        return result
    except Exception as e:
        return {"status": "error", "reply": f"Error: {e}"}

def check_scam_keywords(message: str) -> tuple:
    """Quick local check for scam indicators"""
    msg_lower = message.lower()

    scam_keywords = [
        "money", "rupees", "rs", "5000", "1000", "urgently", "immediately",
        "stuck", "stranded", "help me", "gpay", "paytm", "phonepe", "upi",
        "your friend", "emergency", "send", "transfer", "need money"
    ]

    found = [kw for kw in scam_keywords if kw in msg_lower]
    is_likely_scam = len(found) >= 2

    return is_likely_scam, found

def main():
    global API_KEY
    print_header()

    if not API_KEY:
        API_KEY = input(f"{YELLOW}Enter API key for x-api-key header: {RESET}").strip()
        if not API_KEY:
            print(f"{RED}API key is required to run interactive CLI.{RESET}")
            return

    session_id = str(uuid.uuid4())[:8]
    history = []
    turn = 0

    print(f"{BLUE}Session ID: {session_id}{RESET}\n")
    print("-" * 60)

    while True:
        # Get user input (scammer message)
        try:
            message = input(f"\n{RED}Scammer > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Goodbye!{RESET}")
            break

        if not message:
            continue
        if message.lower() == 'quit':
            print(f"{YELLOW}Goodbye!{RESET}")
            break
        if message.lower() == 'new':
            session_id = str(uuid.uuid4())[:8]
            history = []
            turn = 0
            print(f"\n{CYAN}New session started: {session_id}{RESET}")
            print("-" * 60)
            continue

        turn += 1

        # Send to API
        result = send_message(session_id, message, history)

        # Display agent response
        reply = result.get("reply", "No response")
        print(f"{GREEN}Agent   > {reply}{RESET}")

        # Update history
        history.append({"sender": "scammer", "text": message})
        history.append({"sender": "user", "text": reply})

        # Local heuristic indicators (for quick input signal only)
        local_is_scam, local_keywords = check_scam_keywords(message)
        local_status = f"{RED}YES{RESET}" if local_is_scam else f"{GREEN}NO{RESET}"

        # Backend state from API (source of truth)
        session_state = result.get("sessionState") or {}
        backend_is_scam = session_state.get("scam_detected")
        backend_status = "UNKNOWN"
        if backend_is_scam is True:
            backend_status = f"{RED}YES{RESET}"
        elif backend_is_scam is False:
            backend_status = f"{GREEN}NO{RESET}"

        intel = (session_state.get("intelligence") or {})
        extracted_summary = (
            f"names={len(intel.get('scammer_names', []))}, "
            f"phones={len(intel.get('phone_numbers', []))}, "
            f"upis={len(intel.get('upi_ids', []))}, "
            f"emails={len(intel.get('email_addresses', []))}, "
            f"banks={len(intel.get('bank_accounts', []))}"
        )

        completed = bool(result.get("completed"))
        completion_reason = result.get("completionReason")
        elapsed_secs = result.get("_elapsed_secs")

        # Show state info
        print(f"\n{YELLOW}{'─' * 50}")
        print(f"Turn: {turn} | Session: {session_id}")
        print(
            f"Local Indicators: {local_status} {CYAN}[{', '.join(local_keywords[:5])}]{RESET}"
            if local_keywords else f"Local Indicators: {local_status}"
        )
        print(f"Backend ScamDetected: {backend_status}")
        print(f"Extracted Intel: {extracted_summary}")
        if isinstance(elapsed_secs, (int, float)):
            print(f"Response Time: {elapsed_secs:.3f}s")
        if completed:
            print(f"{RED}Session Completed{RESET} | Reason: {completion_reason}")
            print_full_intelligence(intel)
            print_callback_intelligence(result)
            print(f"{YELLOW}Start a new session with 'new' to continue testing.{RESET}")
        print(f"{'─' * 50}{RESET}")

if __name__ == "__main__":
    main()
