import logging
import time
import os
from audio_listener import AudioListener
from intent_engine import IntentEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RoboDinoMain")

def main():
    logger.info("Initializing RoboDino Intent Pipeline...")
    
    # Check for API Key
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("Error: OPENAI_API_KEY environment variable is missing!")
        logger.error("Please set it using: export OPENAI_API_KEY='your-key-here'")
        # We can still run, but LLM parsing will fail
        
    try:
        # 1. Initialize Audio Listener (Microphone + VAD)
        listener = AudioListener(sample_rate=16000, frame_duration_ms=30)
        
        # 2. Initialize Intent Engine (faster-whisper + OpenAI)
        # Using tiny/base models for speed. Can upgrade to small/medium for better accuracy.
        engine = IntentEngine(model_size="base", device="cpu", compute_type="int8")
        
        logger.info("Initialization complete. Entering main loop.")
        logger.info("="*50)
        logger.info("Speak into your microphone. Say 'exit' or press Ctrl+C to stop.")
        logger.info("="*50)
        
        temp_audio_file = "dino_listen_temp.wav"
        
        while True:
            # 3. Wait for and record speech
            # This blocks until speech is detected and silence follows
            recorded_file = listener.listen_and_record(temp_audio_file)
            
            if not recorded_file:
                logger.warning("No audio recorded (likely interrupted). Exiting loop.")
                break
                
            # 4. Transcribe audio to text
            logger.info("Processing audio...")
            start_time = time.time()
            transcription = engine.transcribe_audio(recorded_file)
            
            if not transcription:
                logger.info("Could not transcribe anything clearly.")
                continue
                
            logger.info(f"User Transcribed Text: --> '{transcription}' <--")
            
            # Simple exit command purely for testing convenience
            if "exit" in transcription.lower().strip() or "quit" in transcription.lower().strip():
                logger.info("Exit command spoken. Shutting down.")
                break
                
            # 5. Parse text into JSON intent
            intent = engine.get_behavior_intent(transcription)
            parsing_time = time.time() - start_time
            
            # 6. Output the final behavior mapping
            logger.info(f"Final Behavior Decision: {intent['behavior']} (Processing Time: {parsing_time:.2f}s)")
            logger.info("="*50)
            
            # Note: This is where you would connect to the Go2/MuJoCo Planner Layer
            # Example: robot_controller.perform_action(intent['behavior'])

    except KeyboardInterrupt:
        logger.info("\nCaught KeyboardInterrupt. Shutting down RoboDino...")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}")
    finally:
        # Cleanup temp file
        if os.path.exists("dino_listen_temp.wav"):
            os.remove("dino_listen_temp.wav")
            
if __name__ == "__main__":
    main()
