import os
import json
import whisper
from moviepy import VideoFileClip

def convert_video_to_mp3(video_path, audio_path):
    if os.path.exists(audio_path):
        print("[INFO] Audio file pehle se maujood hai.")
        return audio_path
        
    print(f"[INFO] Video '{video_path}' ko MP3 mein convert kiya ja raha hai...")
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, codec='libmp3lame')
    video_clip.close()
    print(f"[SUCCESS] Audio successfully save ho gaya: '{audio_path}'")
    return audio_path

def transcribe_and_optimize_chunks(audio_path, json_path):
    if os.path.exists(json_path):
        print("[INFO] Transcript JSON pehle se maujood hai, file load ki ja rahi hai...")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print("[INFO] Whisper model (base version) load kiya ja raha hai...")
    model = whisper.load_model("base")
    
    print(f"[INFO] Audio transcribe ho raha hai: {audio_path}...")
    result = model.transcribe(audio_path)
    
    raw_segments = result["segments"]
    optimized_chunks = []
    
    current_chunk_text = ""
    start_time = raw_segments[0]["start"] if raw_segments else 0.0
    end_time = start_time
    
    # Chunks ko optimize karna taaki context behtar rahe (Bonus feature)
    for segment in raw_segments:
        current_chunk_text += " " + segment["text"].strip()
        end_time = segment["end"]
        
        if len(current_chunk_text) >= 400:
            optimized_chunks.append({
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": current_chunk_text.strip()
            })
            current_chunk_text = ""
            start_time = end_time
            
    if current_chunk_text.strip():
        optimized_chunks.append({
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "text": current_chunk_text.strip()
        })
        
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(optimized_chunks, f, indent=4, ensure_ascii=False)
        
    print(f"[SUCCESS] Transcription aur chunking complete! Saved to '{json_path}'")
    return optimized_chunks