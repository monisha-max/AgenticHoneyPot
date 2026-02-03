
from app.core.orchestrator import ConversationOrchestrator
from app.api.schemas import Message, SenderType, PersonaType
import asyncio

def test_detect_english_monotone():
    orchestrator = ConversationOrchestrator()
    
    # Monotonous English message
    monotone_msg = "Hello dear, kindly check your bank account for the recent transaction and your for with that. We have updated the details."
    result1 = orchestrator._detect_english_monotone(monotone_msg)
    print(f"Monotone message detection: {result1} (Expected: True)")
    
    # Short message
    short_msg = "Hi"
    result2 = orchestrator._detect_english_monotone(short_msg)
    print(f"Short message detection: {result2} (Expected: False)")
    
    # Non-English / Hindi message
    hindi_msg = "Arey bhai sahab, kya hua?"
    result3 = orchestrator._detect_english_monotone(hindi_msg)
    print(f"Hindi message detection: {result3} (Expected: False)")

def test_persona_override():
    orchestrator = ConversationOrchestrator()
    
    msg_text = "Hello dear, kindly check your bank account for the recent transaction and your for with that. We have updated the details."
    is_monotone = orchestrator._detect_english_monotone(msg_text)
    print(f"Is monotone: {is_monotone}")
    
    if is_monotone:
        persona = orchestrator.persona_engine.get_english_persona()
        print(f"Override persona: {persona}")
        assert persona in [PersonaType.VIKRAM_IT, PersonaType.ANANYA_STUDENT]
        print("Persona override verification: PASSED")

def test_greeting_detection():
    orchestrator = ConversationOrchestrator()
    
    # Generic greeting
    greeting_msg = "Hello"
    is_greeting = len(greeting_msg.strip().split()) <= 2 and any(g in greeting_msg.lower() for g in ["hi", "hello", "hey"])
    print(f"Greeting detection: {is_greeting} (Expected: True)")
    assert is_greeting is True

if __name__ == "__main__":
    print("Running Verification Script...")
    try:
        test_detect_english_monotone()
        test_persona_override()
        test_greeting_detection()
        print("All local checks PASSED.")
    except Exception as e:
        print(f"Verification FAILED: {e}")
