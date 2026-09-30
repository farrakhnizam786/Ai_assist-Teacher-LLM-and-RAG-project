# import time
# import os
# from google import genai

# # Initialize the Gemini Client
# # Make sure GEMINI_API_KEY is set in your environment variables
# client = genai.Client()

# def process_and_query_video_with_gemini(video_path, user_question):
#     """
#     Uploads a video file directly to Gemini API, waits for processing,
#     and answers questions based on both audio and visual/slide content.
#     """
#     if not os.path.exists(video_path):
#         return "Error: Video file not found!"

#     print("[INFO] Uploading video to Gemini cloud for multimodal analysis...")
    
#     # 1. Upload video file using the official Files API
#     video_file = client.files.upload(file=video_path)
    
#     # 2. Wait for Google servers to process the video (transcode/frame sample)
#     while video_file.state == "PROCESSING":
#         print("⏳ Waiting for video processing to complete...")
#         time.sleep(5)
#         video_file = client.files.get(name=video_file.name)
        
#     if video_file.state == "FAILED":
#         raise ValueError("Video processing failed on Gemini servers.")
        
#     print("✅ Video processed successfully by Gemini! Generating answer...")

#     # 3. Prompt engineering tailored for an AI Teaching Assistant
#     prompt = f"""
#     [ROLE]
#     You are an expert, patient, and knowledgeable AI Teaching Assistant. 
#     Analyze the provided video carefully (both what is spoken in the audio and what is visually shown on the screen, slides, or board).

#     [INSTRUCTIONS]
#     - Answer the student's question accurately using details from the video.
#     - If a question was shown or asked on screen, address it.
#     - Provide reference timestamps (MM:SS) wherever applicable.
#     - If the answer is completely absent from the video, state: "I couldn't find this topic in the uploaded video."

#     [STUDENT QUESTION]
#     {user_question}
#     """

#     # 4. Generate content using Gemini 2.5 Flash
#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=[video_file, prompt]
#     )

#     return response.text