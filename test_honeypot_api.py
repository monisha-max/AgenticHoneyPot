import requests
import uuid
import json
from datetime import datetime


ENDPOINT_URL = "http://localhost:8000/api/honeypot"
API_KEY = "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"


test_scenario = {
    'scenarioId': 'bank_fraud',
    'name': 'Bank Fraud Detection',
    'scamType': 'bank_fraud',
    'initialMessage': 'URGENT: Your SBI account has been compromised. Your account will be blocked in 2 hours. Share your account number and OTP immediately to verify your identity.',
    'metadata': {
        'channel': 'SMS',
        'language': 'English',
        'locale': 'IN'
    },
    'maxTurns': 10,
    'fakeData': {
        'bankAccount': '1234567890123456',
        'upiId': 'scammer.fraud@fakebank',
        'phoneNumber': '+91-9876543210'
    }
}


def test_honeypot_api():
    """Test your honeypot API endpoint"""

    # Generate unique session ID
    session_id = str(uuid.uuid4())
    conversation_history = []

    # Setup headers
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY
    }

    print(f"\n🔬 Testing Session: {session_id}")
    print("=" * 60)

    # Simulate conversation turns
    for turn in range(1, test_scenario['maxTurns'] + 1):
        print(f"\n--- Turn {turn} ---")

        # First turn: use initial message
        if turn == 1:
            scammer_message = test_scenario['initialMessage']
        else:
            # For subsequent turns, prompt for input
            scammer_message = input("Enter next scammer message (or 'quit' to stop): ")
            if scammer_message.lower() == 'quit':
                break

        # Prepare message object (matches app/api/schemas.py Message model)
        message = {
            "sender": "scammer",
            "text": scammer_message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Prepare request body (matches HoneypotRequest schema)
        request_body = {
            'sessionId': session_id,
            'message': message,
            'conversationHistory': conversation_history,
            'metadata': test_scenario['metadata']
        }

        print(f"📨 Scammer: {scammer_message}")

        try:
            # Call the API
            response = requests.post(
                ENDPOINT_URL,
                headers=headers,
                json=request_body,
                timeout=60
            )

            # Check response
            if response.status_code != 200:
                print(f"❌ ERROR: API returned status {response.status_code}")
                print(f"Response: {response.text}")
                break

            response_data = response.json()

            # Extract honeypot reply (matches HoneypotResponse schema)
            honeypot_reply = response_data.get('reply', '')

            if not honeypot_reply:
                print("❌ ERROR: No 'reply' field in response")
                print(f"Response data: {json.dumps(response_data, indent=2)}")
                break

            print(f"✅ Honeypot: {honeypot_reply}")

            # Show detection status
            scam_detected = response_data.get('scamDetected')
            if scam_detected is not None:
                status_icon = "🚨" if scam_detected else "✅"
                print(f"   {status_icon} Scam Detected: {scam_detected}")

            extracted = response_data.get('extractedIntelligence', {})
            if extracted:
                non_empty = {k: v for k, v in extracted.items() if v}
                if non_empty:
                    print(f"   🔍 Intel: {json.dumps(non_empty)}")

            agent_notes = response_data.get('agentNotes', '')
            if agent_notes:
                print(f"   📝 Notes: {agent_notes}")

            # Update conversation history for next turn
            conversation_history.append(message)
            conversation_history.append({
                'sender': 'user',
                'text': honeypot_reply,
                'timestamp': datetime.utcnow().isoformat() + "Z"
            })

        except requests.exceptions.Timeout:
            print("❌ ERROR: Request timeout (>30 seconds)")
            break
        except requests.exceptions.ConnectionError as e:
            print(f"❌ ERROR: Connection failed. Is the server running?")
            print(f"   Make sure to start it: python -m app.main")
            print(f"   Detail: {e}")
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            break

    # ========================================================================
    # FINAL OUTPUT EVALUATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("📋 EVALUATING FINAL OUTPUT STRUCTURE")
    print("=" * 60)

    # This simulates what the GUVI evaluator checks (GuviCallbackPayload format)
    final_output = {
        "sessionId": session_id,
        "scamDetected": True,
        "totalMessagesExchanged": len(conversation_history),
        "extractedIntelligence": {
            "phoneNumbers": ["+91-9876543210"],
            "bankAccounts": ["1234567890123456"],
            "upiIds": ["scammer.fraud@fakebank"],
            "phishingLinks": ["http://malicious-site.com"],
            "suspiciousKeywords": ["urgent", "blocked", "otp", "verify"]
        },
        "agentNotes": "Scammer claimed to be from SBI fraud department, used urgency tactics."
    }

    # Evaluate
    score = evaluate_final_output(final_output, test_scenario, conversation_history)

    print(f"\n📊 Your Score: {score['total']}/100")
    print(f"   - Scam Detection:          {score['scamDetection']}/20")
    print(f"   - Intelligence Extraction:  {score['intelligenceExtraction']}/40")
    print(f"   - Engagement Quality:       {score['engagementQuality']}/20")
    print(f"   - Response Structure:        {score['responseStructure']}/20")

    return score


def evaluate_final_output(final_output, scenario, conversation_history):
    """Evaluate final output using the same logic as the evaluator"""

    score = {
        'scamDetection': 0,
        'intelligenceExtraction': 0,
        'engagementQuality': 0,
        'responseStructure': 0,
        'total': 0
    }

    # 1. Scam Detection (20 points)
    if final_output.get('scamDetected', False):
        score['scamDetection'] = 20

    # 2. Intelligence Extraction (40 points)
    extracted = final_output.get('extractedIntelligence', {})
    fake_data = scenario.get('fakeData', {})

    key_mapping = {
        'bankAccount': 'bankAccounts',
        'upiId': 'upiIds',
        'phoneNumber': 'phoneNumbers',
        'phishingLink': 'phishingLinks',
        'emailAddress': 'emailAddresses'
    }

    for fake_key, fake_value in fake_data.items():
        output_key = key_mapping.get(fake_key, fake_key)
        extracted_values = extracted.get(output_key, [])

        if isinstance(extracted_values, list):
            if any(fake_value in str(v) for v in extracted_values):
                score['intelligenceExtraction'] += 10
        elif isinstance(extracted_values, str):
            if fake_value in extracted_values:
                score['intelligenceExtraction'] += 10

    score['intelligenceExtraction'] = min(score['intelligenceExtraction'], 40)

    # 3. Engagement Quality (20 points)
    messages = final_output.get('totalMessagesExchanged', 0)

    if messages > 0:
        score['engagementQuality'] += 5
    if messages >= 4:
        score['engagementQuality'] += 5
    if messages >= 8:
        score['engagementQuality'] += 5
    if messages >= 12:
        score['engagementQuality'] += 5

    # 4. Response Structure (20 points) — checks GuviCallbackPayload fields
    required_fields = ['sessionId', 'scamDetected', 'extractedIntelligence']
    optional_fields = ['agentNotes', 'totalMessagesExchanged']

    for field in required_fields:
        if field in final_output:
            score['responseStructure'] += 5

    for field in optional_fields:
        if field in final_output and final_output[field]:
            score['responseStructure'] += 2.5

    score['responseStructure'] = min(score['responseStructure'], 20)

    # Calculate total
    score['total'] = sum([
        score['scamDetection'],
        score['intelligenceExtraction'],
        score['engagementQuality'],
        score['responseStructure']
    ])

    return score


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    test_honeypot_api()
