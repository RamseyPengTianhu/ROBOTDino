import os
import json
import logging
from faster_whisper import WhisperModel
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntentEngine")

class IntentEngine:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8"):
        """
        Initializes the Intent Engine, loading the faster-whisper model.
        Args:
            model_size: "tiny", "base", "small", "medium", "large-v3"
            device: "cpu" or "cuda"
            compute_type: "int8" or "float16"
        """
        logger.info(f"Loading faster-whisper model '{model_size}' on '{device}'...")
        # Download root can be customized if needed
        self.asr_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("ASR model loaded successfully.")

        # Configuration for Ollama
        self.ollama_url = "http://localhost:11434/api/generate"
        self.llm_model = "qwen2.5:1.5b"
        logger.info(f"LLM parsing configured to use local Ollama model: {self.llm_model}")
        
        # Valid behaviors for RoboDino
        self.valid_behaviors = ["greet", "curious", "excited", "present", "goodbye", "idle", "confused"]
        
        self.system_prompt = f"""
        You are RoboDino's intent parser.
        RoboDino is a cute, reactive, lifelike interactive dinosaur robot.
        
        Your job:
        - Read the transcribed user's speech.
        - Output ONLY valid JSON representing the robot's physical behavior reaction.
        - No explanation, no markdown formatting.
        
        Allowed behavior values:
        {json.dumps(self.valid_behaviors)}
        
        Return exactly in this JSON format:
        {{"behavior": "<one_of_the_allowed_values>"}}
        
        Examples:
        User: "Hello there little guy!" -> {{"behavior": "greet"}}
        User: "Wow, do a trick!" -> {{"behavior": "excited"}}
        User: "Look over here." -> {{"behavior": "curious"}}
        User: "What is this?" -> {{"behavior": "present"}}
        User: "Bye bye" -> {{"behavior": "goodbye"}}
        User: (Mumbling / unclear) -> {{"behavior": "confused"}}
        User: "Nevermind" -> {{"behavior": "idle"}}
        """

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribes the given audio file to text using faster-whisper.
        """
        logger.info(f"Transcribing {audio_path}...")
        try:
            segments, info = self.asr_model.transcribe(audio_path, beam_size=5, language="en")
            
            # segments is a generator, so we must iterate
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
                
            return transcription.strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def get_behavior_intent(self, user_text: str) -> dict:
        """
        Sends the transcribed text to the LLM to get a structured JSON intent.
        """
        if not user_text:
            return {"behavior": "idle"}
            
        logger.info(f"Parsing intent for text: '{user_text}'")
        try:
           # Construct prompt for the LLM
           full_prompt = f"{self.system_prompt}\n\nUser: {user_text}\nOutput JSON:"

           response = requests.post(
               self.ollama_url,
               json={
                   "model": self.llm_model,
                   "prompt": full_prompt,
                   "stream": False,
                   "format": "json",
                   "options": {
                       "temperature": 0.1
                   }
               },
               timeout=15
           )
           
           if response.status_code == 200:
               result_str = response.json().get("response", "{}")
               try:
                   result_json = json.loads(result_str)
               except json.JSONDecodeError:
                   logger.warning(f"Failed to parse JSON from Ollama: {result_str}")
                   result_json = {}
           else:
               logger.error(f"Ollama API returned status {response.status_code}: {response.text}")
               result_json = {}
           
           # Validation
           behavior = result_json.get("behavior", "confused")
           if behavior not in self.valid_behaviors:
               logger.warning(f"LLM returned invalid behavior '{behavior}'. Defaulting to 'confused'.")
               behavior = "confused"
               
           return {"behavior": behavior}
           
        except requests.exceptions.Timeout:
            logger.error("Ollama API timed out. Make sure Ollama container/service is running.")
            return {"behavior": "confused"}
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Make sure the service is running on port 11434.")
            return {"behavior": "confused"}
        except Exception as e:
             logger.error(f"LLM Intent parsing failed: {e}")
             return {"behavior": "confused"}


if __name__ == "__main__":
    # Simple Manual Test
    engine = IntentEngine(model_size="tiny", device="cpu")
    print(engine.get_behavior_intent("Hello little dinosaur!"))
    print(engine.get_behavior_intent("Can you show me a trick?"))

