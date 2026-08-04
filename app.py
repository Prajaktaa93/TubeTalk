import streamlit as st
import os
import shutil
import yt_dlp
import time

# --- RAG IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


# DIRECT IMPORTS
from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
import google.generativeai as genai

st.set_page_config(page_title="TubeTalk 🎥", page_icon="🤖", layout="wide")

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # Current video's messages
if "past_chats" not in st.session_state:
    st.session_state.past_chats = []         # List of saved chat sessions
if "current_video_url" not in st.session_state:
    st.session_state.current_video_url = ""  # URL of currently loaded video
if "content" not in st.session_state:
    st.session_state.content = None          # Transcript text or Gemini audio file
if "content_type" not in st.session_state:
    st.session_state.content_type = None     # "text" or "audio"
if "video_title" not in st.session_state:
    st.session_state.video_title = ""        # Title of current video
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None      # FAISS vector index for RAG

# ============================================================
# HELPERS
# ============================================================

@st.cache_resource(show_spinner=False)
def build_vectorstore(text: str):
    """Chunks transcript text and builds a FAISS vector index for RAG retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def get_working_model_name(api_key):
    """Returns the best available Gemini model name."""
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if 'models/gemini-1.5-flash' in models:
            return 'models/gemini-1.5-flash'
        for m in models:
            if 'flash' in m:
                return m
        return 'models/gemini-pro'
    except:
        return 'gemini-1.5-flash'


def get_video_title(video_url):
    """Fetches the video title using yt-dlp (no download)."""
    try:
        ydl_opts = {'quiet': True, 'skip_download': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return info.get('title', video_url[:60])
    except:
        return video_url[:60]


def download_audio(video_url):
    """Downloads audio from YouTube when transcript fails."""
    # Cross-platform ffmpeg path: works on Mac, Linux (Streamlit Cloud), Windows
    ffmpeg_path = shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': False,
        'no_warnings': False,
        'extractor_args': {
            'youtube': {
                'player_client': ['android'],  # Bypasses SABR streaming & JS runtime requirement
            }
        },
        'ffmpeg_location': ffmpeg_path,
    }
    try:
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            mp3_path = filename.rsplit('.', 1)[0] + '.mp3'

        if os.path.exists("temp_audio.mp3") and os.path.getsize("temp_audio.mp3") > 0:
            return "temp_audio.mp3", None
        elif os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
            return mp3_path, None
        else:
            return None, "Audio file was empty after download. Try a different video."

    except Exception as e:
        return None, str(e)


def get_video_content(video_url, api_key):
    """Extracts video content via transcript (fast) or audio (fallback)."""
    # 1. Try Transcript First
    try:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        else:
            video_id = video_url.split("/")[-1].split("?")[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript_list])
        return text, "text", None
    except:
        # 2. Fallback to Audio
        st.info("⚠️ No transcript found. Switching to Audio Mode (listening to video)...")
        audio_path, error = download_audio(video_url)

        if error:
            return None, None, f"Failed to download audio: {error}"

        genai.configure(api_key=api_key)
        myfile = genai.upload_file(audio_path)

        while myfile.state.name == "PROCESSING":
            time.sleep(2)
            myfile = genai.get_file(myfile.name)

        return myfile, "audio", None


def get_ai_response(query, api_key):
    """Gets multi-turn AI response using full conversation history as context."""
    model_name = get_working_model_name(api_key)
    model = genai.GenerativeModel(model_name)

    # Build conversation context from last 6 Q&A pairs (12 messages)
    history_text = ""
    recent_history = st.session_state.chat_history[-12:]
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    if st.session_state.content_type == "text":
        # RAG: retrieve only the most relevant chunks for this query
        relevant_docs = st.session_state.vectorstore.similarity_search(query, k=4)
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""You are a helpful assistant answering questions about a YouTube video.
Base your answer ONLY on the relevant transcript sections provided below.
Be concise, accurate, and refer specifically to the video content.

--- RELEVANT TRANSCRIPT SECTIONS ---
{context}

--- CONVERSATION HISTORY ---
{history_text}
--- END HISTORY ---

User: {query}
Assistant:"""
        response = model.generate_content(prompt)

    else:
        # Audio mode: pass file object + context prompt
        context_prompt = f"""You are a helpful assistant answering questions about this video.
Use the audio content to answer accurately.

--- CONVERSATION HISTORY ---
{history_text}
--- END HISTORY ---

Current Question: {query}"""
        response = model.generate_content([st.session_state.content, context_prompt])

    return response.text


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
    st.divider()
    st.info("ℹ️ App auto-switches to **Audio Mode** if no transcript exists.")

    # --- PAST CHATS PANEL ---
    if st.session_state.past_chats:
        st.markdown("---")
        st.markdown("## 💬 Past Chats")
        # Show most recent first
        for past in reversed(st.session_state.past_chats):
            title = past.get("title", past["url"])
            short_title = (title[:35] + "...") if len(title) > 35 else title
            with st.expander(f"📼 {short_title}"):
                st.caption(f"🔗 `{past['url'][:55]}`")
                st.markdown("---")
                for msg in past["messages"]:
                    if msg["role"] == "user":
                        st.markdown(f"🧑 **You:** {msg['content']}")
                    else:
                        st.markdown(f"🤖 **TubeTalk:** {msg['content']}")
                    st.markdown("")


# ============================================================
# MAIN PAGE
# ============================================================
st.title("🤖 TubeTalk: Chat with Videos")
st.caption("Paste any YouTube URL and start a conversation about the video.")

st.markdown("")

# --- URL INPUT ROW ---
col1, col2 = st.columns([5, 1])
with col1:
    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...  or  https://youtu.be/...",
        label_visibility="collapsed"
    )
with col2:
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True, type="primary")

# --- ANALYZE LOGIC ---
if analyze_clicked:
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar.")
    elif not video_url:
        st.warning("⚠️ Please paste a YouTube URL.")
    elif video_url.strip() == st.session_state.current_video_url.strip():
        st.info("ℹ️ This video is already loaded. Ask your questions below!")
    else:
        # Save the current chat before switching to new video
        if st.session_state.chat_history and st.session_state.current_video_url:
            st.session_state.past_chats.append({
                "url": st.session_state.current_video_url,
                "title": st.session_state.video_title,
                "messages": st.session_state.chat_history.copy()
            })

        # Reset state for new video
        st.session_state.chat_history = []
        st.session_state.content = None
        st.session_state.content_type = None
        st.session_state.vectorstore = None
        st.session_state.current_video_url = video_url.strip()
        st.session_state.video_title = ""

        with st.spinner("🧠 Analyzing video content..."):
            title = get_video_title(video_url)
            st.session_state.video_title = title

            content, content_type, error = get_video_content(video_url, api_key)

            if error:
                st.error(f"❌ {error}")
                st.session_state.current_video_url = ""  # Allow retry
            else:
                st.session_state.content = content
                st.session_state.content_type = content_type

                # Build RAG vector index for transcript mode
                if content_type == "text":
                    with st.spinner("🔍 Building knowledge index (RAG)..."):
                        st.session_state.vectorstore = build_vectorstore(content)

                mode = "📝 RAG Mode" if content_type == "text" else "🎧 Audio Mode"
                st.success(f"✅ **{title}** — ready in {mode}. Start chatting below!")

# --- CURRENT VIDEO BANNER ---
if st.session_state.content and st.session_state.video_title:
    mode_icon = "📝" if st.session_state.content_type == "text" else "🎧"
    st.markdown(
        f"<div style='background:#1e1e2e;padding:10px 16px;border-radius:8px;border-left:3px solid #7c3aed;'>"
        f"{mode_icon} <b>Now chatting about:</b> {st.session_state.video_title}"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("")

# --- CHAT HISTORY DISPLAY ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- CHAT INPUT (pinned to bottom) ---
if st.session_state.content and api_key:
    query = st.chat_input("Ask something about the video...")
    if query:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Get and show AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = get_ai_response(query, api_key)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"❌ Error: {e}")

elif not st.session_state.content:
    st.markdown(
        "<div style='text-align:center;color:#888;margin-top:60px;'>"
        "👆 Paste a YouTube URL above and click <b>Analyze</b> to start chatting!"
        "</div>",
        unsafe_allow_html=True
    )