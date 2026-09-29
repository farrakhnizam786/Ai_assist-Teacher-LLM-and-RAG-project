import os
from transcribe import convert_video_to_mp3, transcribe_and_optimize_chunks
from rag_pipeline import store_chunks_in_chroma, retrieve_top_chunks, generate_ai_tutor_response

if __name__ == "__main__":
    print("==================================================")
    print("       AI TEACHING ASSISTANT (LIVE TERMINAL)      ")
    print("==================================================")
    
    INPUT_DIR = "input_videos"
    OUTPUT_DIR = "output_data"
    DB_DIR = "vector_db"
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Video ka naam input lena
    video_name = input("Enter video file name inside 'input_videos/' (default: sample_video.mp4): ").strip()
    if not video_name:
        video_name = "sample_video.mp4"
        
    VIDEO_PATH = os.path.join(INPUT_DIR, video_name)
    AUDIO_PATH = os.path.join(OUTPUT_DIR, "extracted_audio.mp3")
    JSON_PATH = os.path.join(OUTPUT_DIR, "transcript_chunks.json")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] '{VIDEO_PATH}' nahi mili! Kripya 'input_videos' folder mein yeh video rakhein.")
    else:
        # Step 2: Processing & Indexing
        audio_file = convert_video_to_mp3(VIDEO_PATH, AUDIO_PATH)
        chunks = transcribe_and_optimize_chunks(audio_file, JSON_PATH)
        collection = store_chunks_in_chroma(JSON_PATH, db_path=DB_DIR)
        
        print("\n--------------------------------------------------")
        print("🎓 AI Teaching Assistant: Hello student!")
        print("📁 Your video successfully read aur index ho chuki hai.")
        print("--------------------------------------------------")
        
        # Step 3: User Intent Check (Yes/No)
        user_intent = input("\n❓ Do you have any quesiotn related to this topic/video ? (yes/no): ").strip().lower()
        
        if user_intent in ['yes', 'y', 'ha', 'haan']:
            student_query = input("✍️ Please write here: ").strip()
            if not student_query:
                student_query = "What is the main concept explained in this video?"
        else:
            # Agar user 'no' karega, toh automatic video overview/summary query run ho jayegi
            print("\n[INFO] 'No' selected by you,Video overview is in progress please wait...")
            student_query = "What is the main concept explained in this video?"
            
        # Step 4: RAG Retrieval & LLM Generation
        top_matching_chunks = retrieve_top_chunks(student_query, collection, top_k=3)
        final_answer = generate_ai_tutor_response(student_query, top_matching_chunks)
        
        print("\n==================================================")
        print("                 AI TUTOR ANSWER                  ")
        print("==================================================")
        print(final_answer)
        print("==================================================")