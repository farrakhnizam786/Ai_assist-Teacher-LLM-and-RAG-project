import os
import time
import yt_dlp
import streamlit as st
from transcribe import convert_video_to_mp3, transcribe_and_optimize_chunks
from rag_pipeline import store_chunks_in_chroma, retrieve_top_chunks, generate_ai_tutor_response

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Directories setup
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_data"
DB_DIR = "vector_db"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.session_state.collection = None

if "current_video" not in st.session_state:
    st.session_state.current_video = None

# Helper function to download video from URL (Updated & Robust)
# Helper function to download video from URL (Fixed for YouTube changes)
def download_video_from_url(url):
    base_path = os.path.join(INPUT_DIR, "downloaded_youtube_video")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': base_path + '.%(ext)s',
        'quiet': True,
        'overwrites': True,
        'extractor_args': {'youtube': {'player_client': ['default']}}
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename

# 2. Sidebar Layout - Upload, YouTube Link & Manager
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.title("Control Panel")
    st.markdown("Upload a video or paste a YouTube link to start.")
    
    st.divider()
    
    # Option A: File Uploader
    uploaded_file = st.file_uploader("📂 Upload Local Video (.mp4)", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        new_video_path = os.path.join(INPUT_DIR, uploaded_file.name)
        if not os.path.exists(new_video_path):
            with open(new_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")

    st.markdown("<p style='text-align: center; color: gray;'>- OR -</p>", unsafe_allow_html=True)
    
    # Option B: YouTube Link Input
    youtube_url = st.text_input("🔗 Paste YouTube Video Link:")
    download_btn = st.button("📥 Download from Link", use_container_width=True)
    
    if download_btn and youtube_url:
        with st.spinner("Downloading video from link... Please wait."):
            try:
                downloaded_path = download_video_from_url(youtube_url)
                st.success("Download complete! Select it from storage below.")
            except Exception as e:
                st.error(f"Download failed: {e}")

    st.divider()
    st.subheader("📼 Stored Videos Manager")
    
    existing_videos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv'))]
    
    selected_video = None
    if existing_videos:
        selected_video = st.selectbox("Select video for Q&A:", existing_videos)
        
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("🚀 Process", type="primary", use_container_width=True)
        with col2:
            delete_btn = st.button("🗑️ Delete", type="secondary", use_container_width=True)
            
        if delete_btn and selected_video:
            video_to_delete = os.path.join(INPUT_DIR, selected_video)
            try:
                os.remove(video_to_delete)
                st.success(f"Deleted {selected_video}!")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting file: {e}")
    else:
        st.info("No videos found. Upload or download one.")
        process_btn = False

# 3. Main Header Section
st.markdown('<p class="main-title">🎓 AI Teaching Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Your intelligent video lecture companion powered by RAG & Groq LLM</p>', unsafe_allow_html=True)
st.divider()

if st.session_state.current_video:
    st.info(f"🟢 **Currently Active Video for Q&A:** `{st.session_state.current_video}`")

# 4. Processing Pipeline with Live Progress Bar
if 'process_btn' in locals() and process_btn and selected_video:
    video_path = os.path.join(INPUT_DIR, selected_video)
    st.session_state.current_video = selected_video
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🎙️ Step 1/3: Extracting Audio via MoviePy...")
        progress_bar.progress(10)
        
        audio_path = os.path.join(OUTPUT_DIR, "extracted_audio.mp3")
        audio_file = convert_video_to_mp3(video_path, audio_path)
        progress_bar.progress(30)
        
        status_text.text("📝 Step 2/3: Transcribing & Optimizing Chunks via Whisper...")
        progress_bar.progress(50)
        
        json_path = os.path.join(OUTPUT_DIR, "transcript_chunks.json")
        chunks = transcribe_and_optimize_chunks(audio_file, json_path)
        progress_bar.progress(80)
        
        status_text.text("🗄️ Step 3/3: Indexing Embeddings into ChromaDB Vector Store...")
        progress_bar.progress(90)
        
        collection = store_chunks_in_chroma(json_path, db_path=DB_DIR)
        st.session_state.collection = collection
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        st.success(f"🎉 '{selected_video}' successfully processed! You can now start asking questions below.")
        
        st.session_state.messages = []
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during processing: {str(e)}")

# 5. Interactive Chat Interface
if st.session_state.collection is not None and st.session_state.current_video is not None:
    st.subheader(f"💬 Chatting about: `{st.session_state.current_video}`")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your question about the lecture..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing lecture transcript and retrieving context..."):
                try:
                    top_chunks = retrieve_top_chunks(prompt, st.session_state.collection, top_k=5)
                    answer = generate_ai_tutor_response(prompt, top_chunks)
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = f"⚠️ An error occurred: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
else:
    st.info("👈 Please upload a video or paste a YouTube link in the sidebar, select it, and click **'Process'** to begin.")