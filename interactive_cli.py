#!/usr/bin/env python3
"""
Interactive CLI for testing the Honeypot API
"""
import requests
import uuid
from datetime import datetime

API_URL = "http://localhost:8000/api/honeypot-demo"
API_KEY = "test-api-key-123"

# Colors for terminal
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

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
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        return response.json()
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
    print_header()

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

        # Show state info
        print(f"\n{YELLOW}{'─' * 50}")
        print(f"Turn: {turn} | Session: {session_id}")
        print(
            f"Local Indicators: {local_status} {CYAN}[{', '.join(local_keywords[:5])}]{RESET}"
            if local_keywords else f"Local Indicators: {local_status}"
        )
        print(f"Backend ScamDetected: {backend_status}")
        print(f"Extracted Intel: {extracted_summary}")
        if completed:
            print(f"{RED}Session Completed{RESET} | Reason: {completion_reason}")
            print(f"{YELLOW}Start a new session with 'new' to continue testing.{RESET}")
        print(f"{'─' * 50}{RESET}")

if __name__ == "__main__":
    main()
