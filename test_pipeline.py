import os
import json
from intent_engine import IntentEngine

def test_intent_engine():
    print("Testing IntentEngine LLM parsing without audio...")
    engine = IntentEngine(model_size="tiny", device="cpu", compute_type="int8")
    
    test_cases = [
        "Hello little dinosaur",
        "Show me a trick",
        "Look at me",
        "Who are you",
        "Goodbye"
    ]
    
    for case in test_cases:
        print(f"\nUser text: '{case}'")
        intent = engine.get_behavior_intent(case)
        print(f"Parsed Intent: {json.dumps(intent)}")

if __name__ == "__main__":
    test_intent_engine()
