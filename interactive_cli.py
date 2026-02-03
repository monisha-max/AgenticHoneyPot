#!/usr/bin/env python3
"""
Interactive CLI for testing the Honeypot API
"""
import requests
import json
import uuid
from datetime import datetime

API_URL = "http://localhost:8000/api/honeypot"
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

        # Check for scam indicators
        is_scam, keywords = check_scam_keywords(message)
        scam_status = f"{RED}YES{RESET}" if is_scam else f"{GREEN}NO{RESET}"

        # Show state info
        print(f"\n{YELLOW}{'─' * 50}")
        print(f"Turn: {turn} | Session: {session_id}")
        print(f"Scam Indicators: {scam_status} {CYAN}[{', '.join(keywords[:5])}]{RESET}" if keywords else f"Scam Indicators: {scam_status}")
        print(f"{'─' * 50}{RESET}")

if __name__ == "__main__":
    main()
