import requests
import uuid
import json
from datetime import datetime

# Your API configuration
ENDPOINT_URL = "https://your-api-endpoint.com/honeypot"
API_KEY = "your-api-key-here"  # Optional

# Test scenario
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
    headers = {'Content-Type': 'application/json'}
    if API_KEY:
        headers['x-api-key'] = API_KEY
    
    print(f"Testing Session: {session_id}")
    print("=" * 60)
    
    # Simulate conversation turns
    for turn in range(1, test_scenario['maxTurns'] + 1):
        print(f"\n--- Turn {turn} ---")
        
        # First turn: use initial message
        if turn == 1:
            scammer_message = test_scenario['initialMessage']
        else:
            # For self-testing, you can manually craft follow-up messages
            # or use simple templates
            scammer_message = input("Enter next scammer message (or 'quit' to stop): ")
            if scammer_message.lower() == 'quit':
                break
        
        # Prepare message object
        message = {
            "sender": "scammer",
            "text": scammer_message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Prepare request
        request_body = {
            'sessionId': session_id,
            'message': message,
            'conversationHistory': conversation_history,
            'metadata': test_scenario['metadata']
        }
        
        print(f"Scammer: {scammer_message}")
        
        try:
            # Call your API
            response = requests.post(
                ENDPOINT_URL,
                headers=headers,
                json=request_body,
                timeout=30
            )
            
            # Check response
            if response.status_code != 200:
                print(f"❌ ERROR: API returned status {response.status_code}")
                print(f"Response: {response.text}")
                break
            
            response_data = response.json()
            
            # Extract honeypot reply
            honeypot_reply = response_data.get('reply') or \
                           response_data.get('message') or \
                           response_data.get('text')
            
            if not honeypot_reply:
                print("❌ ERROR: No reply/message/text field in response")
                print(f"Response data: {response_data}")
                break
            
            print(f"✅ Honeypot: {honeypot_reply}")
            
            # Update conversation history
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
            print(f"❌ ERROR: Connection failed - {e}")
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            break
    
    # Test final output structure
    print("\n" + "=" * 60)
    print("Now test your final output structure:")
    print("=" * 60)
    
    final_output = {
        "sessionId": "abc123-session-id",
   "scamDetected": true,
   "totalMessagesExchanged": 18,
   "extractedIntelligence": {
    "phoneNumbers": ["+91-9876543210"],
    "bankAccounts": ["1234567890123456"],
    "upiIds": ["scammer.fraud@fakebank"],
    "phishingLinks": ["http://malicious-site.com"],
    "emailAddresses": ["scammer@fake.com"]
    },
    "agentNotes": "Scammer claimed to be from SBI fraud department, provided fake ID..."

    }
    
    # Evaluate the final output
    score = evaluate_final_output(final_output, test_scenario, conversation_history)
    
    print(f"\n📊 Your Score: {score['total']}/100")
    print(f"   - Scam Detection: {score['scamDetection']}/20")
    print(f"   - Intelligence Extraction: {score['intelligenceExtraction']}/40")
    print(f"   - Engagement Quality: {score['engagementQuality']}/20")
    print(f"   - Response Structure: {score['responseStructure']}/20")
    
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
    metrics = final_output.get('engagementMetrics', {})
    duration = metrics.get('engagementDurationSeconds', 0)
    messages = metrics.get('totalMessagesExchanged', 0)
    
    if duration > 0:
        score['engagementQuality'] += 5
    if duration > 60:
        score['engagementQuality'] += 5
    if messages > 0:
        score['engagementQuality'] += 5
    if messages >= 5:
        score['engagementQuality'] += 5
    
    # 4. Response Structure (20 points)
    required_fields = ['status', 'scamDetected', 'extractedIntelligence']
    optional_fields = ['engagementMetrics', 'agentNotes']
    
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

# Run the test
if __name__ == "__main__":
    test_honeypot_api()
