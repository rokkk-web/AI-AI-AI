"""Streamlit multi-user, multi-session RAG chatbot backed by Supabase."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = "gpt-4o-mini"
PASSWORD_ITERATIONS = 260_000


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    """Configure warning/error logging for the app."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"multi_user_rag_{datetime.now().strftime('%Y%m%d')}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

    for name in ("httpx", "httpcore", "openai", "langchain", "supabase"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multi_user_rag")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Prompts & text helpers
# ---------------------------------------------------------------------------
ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 기획예산처 RAG 챗봇입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
- 구분선(---, ===, ___), 취소선(~~텍스트~~), URL 출처 나열은 사용하지 마세요.
- 답변 마지막에는 사용자가 이어서 물어볼 만한 질문 3개를 추가하세요.
"""

TITLE_PROMPT = """다음 첫 번째 사용자 질문과 첫 번째 답변을 바탕으로 세션 제목을 한국어로 짧게 만드세요.

규칙:
- 20자 이내
- 따옴표, 번호, 설명 없이 제목만 출력
- 너무 일반적인 제목은 피하고 핵심 주제를 반영
"""


def remove_separators(text: str) -> str:
    """Remove unwanted markdown separators and excessive blank lines."""
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def utc_now_iso() -> str:
    """Return an ISO timestamp suitable for Supabase timestamptz columns."""
    return datetime.now(timezone.utc).isoformat()


def get_secret(name: str) -> str:
    """Read a key from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:  # noqa: BLE001
        value = ""
    return str(value or os.getenv(name, "")).strip()


def missing_keys() -> list[str]:
    """Return missing configuration keys needed by this app."""
    required = ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY")
    return [name for name in required if not get_secret(name)]


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 30) -> str:
    """Format recent chat messages for model context."""
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for msg in tail:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        role = "사용자" if msg.get("role") == "user" else "어시스턴트"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _first_user_assistant_pair(
    messages: list[dict[str, str]],
) -> tuple[str | None, str | None]:
    """Find the first user/assistant exchange for title generation."""
    first_question: str | None = None
    for msg in messages:
        if msg.get("role") == "user" and not first_question:
            first_question = msg.get("content", "")
        elif msg.get("role") == "assistant" and first_question:
            return first_question, msg.get("content", "")
    return first_question, None


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_supabase_client(url: str, anon_key: str) -> Client:
    """Create a cached Supabase client."""
    return create_client(url, anon_key)


def get_supabase() -> Client | None:
    """Return Supabase client or None when configuration is incomplete."""
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return _get_supabase_client(url, key)


def get_llm(temperature: float = 0.3) -> ChatOpenAI | None:
    """Return GPT-4o-mini or None when OPENAI_API_KEY is missing."""
    key = get_secret("OPENAI_API_KEY")
    if not key:
        return None
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature, api_key=key)


def get_embeddings() -> OpenAIEmbeddings | None:
    """Return OpenAI embeddings or None when OPENAI_API_KEY is missing."""
    key = get_secret("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-SHA256 and an embedded random salt."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against the stored PBKDF2 hash."""
    try:
        algorithm, iterations_text, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
    except Exception:  # noqa: BLE001
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


# ---------------------------------------------------------------------------
# Supabase user/session/message persistence
# ---------------------------------------------------------------------------
def create_user(supabase: Client, *, login_id: str, password: str) -> dict[str, Any]:
    """Create an app-managed user row."""
    existing = (
        supabase.table("user")
        .select("id")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        raise ValueError("이미 사용 중인 아이디입니다.")

    payload = {
        "login_id": login_id,
        "password_hash": hash_password(password),
        "created_at": utc_now_iso(),
    }
    resp = supabase.table("user").insert(payload).execute()
    rows = list(resp.data or [])
    if not rows:
        raise RuntimeError("회원가입에 실패했습니다.")
    return rows[0]


def authenticate_user(
    supabase: Client,
    *,
    login_id: str,
    password: str,
) -> dict[str, Any] | None:
    """Authenticate against the app-managed user table."""
    resp = (
        supabase.table("user")
        .select("id,login_id,password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = list(resp.data or [])
    if not rows:
        return None

    user = rows[0]
    if not verify_password(password, str(user.get("password_hash") or "")):
        return None
    return {"id": user["id"], "login_id": user["login_id"]}


def list_sessions(supabase: Client, *, user_id: str) -> list[dict[str, Any]]:
    """Load saved sessions for the current user in newest-first order."""
    resp = (
        supabase.table("chat_sessions")
        .select("id,title,created_at,updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(resp.data or [])


def load_messages(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
) -> list[dict[str, str]]:
    """Load messages for one user-owned session."""
    resp = (
        supabase.table("chat_messages")
        .select("role,content,created_at,id")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .order("created_at")
        .order("id")
        .execute()
    )
    return [
        {"role": str(row.get("role") or "assistant"), "content": str(row.get("content") or "")}
        for row in resp.data or []
    ]


def load_session(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    """Load a user-owned session row by id."""
    resp = (
        supabase.table("chat_sessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    rows = list(resp.data or [])
    return rows[0] if rows else None


def current_file_names(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
) -> list[str]:
    """Return unique file names stored for this user's session."""
    resp = (
        supabase.table("vector_documents")
        .select("file_name")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .execute()
    )
    return sorted({row["file_name"] for row in resp.data or [] if row.get("file_name")})


def generate_session_title(messages: list[dict[str, str]]) -> str:
    """Generate a compact LLM-based title from the first exchange."""
    question, answer = _first_user_assistant_pair(messages)
    if not question:
        return f"새 세션 {datetime.now().strftime('%m%d %H:%M')}"

    llm = get_llm(temperature=0.1)
    if llm is None or not answer:
        return question[:20].strip() or f"새 세션 {datetime.now().strftime('%m%d %H:%M')}"

    try:
        result = llm.invoke(
            [
                SystemMessage(content=TITLE_PROMPT),
                HumanMessage(content=f"[질문]\n{question}\n\n[답변]\n{answer[:3000]}"),
            ]
        )
        title = remove_separators(str(getattr(result, "content", "") or "")).strip()
        title = re.sub(r"^[\"'`]|[\"'`]$", "", title).strip()
        return title[:40] or question[:20].strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Title generation failed: %s", exc)
        return question[:20].strip() or f"새 세션 {datetime.now().strftime('%m%d %H:%M')}"


def ensure_session_row(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]] | None = None,
) -> None:
    """Create or update the session row before child rows reference it."""
    payload = {
        "id": session_id,
        "user_id": user_id,
        "title": generate_session_title(messages or []),
        "updated_at": utc_now_iso(),
    }
    supabase.table("chat_sessions").upsert(payload, on_conflict="id").execute()


def save_messages(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]],
) -> None:
    """Replace the messages stored for a user-owned session."""
    ensure_session_row(
        supabase,
        user_id=user_id,
        session_id=session_id,
        messages=messages,
    )
    supabase.table("chat_messages").delete().eq("user_id", user_id).eq(
        "session_id",
        session_id,
    ).execute()

    if messages:
        rows = [
            {
                "user_id": user_id,
                "session_id": session_id,
                "role": msg.get("role", "assistant"),
                "content": msg.get("content", ""),
                "created_at": utc_now_iso(),
            }
            for msg in messages
        ]
        for i in range(0, len(rows), 50):
            supabase.table("chat_messages").insert(rows[i : i + 50]).execute()

    title = generate_session_title(messages)
    supabase.table("chat_sessions").update(
        {"title": title, "updated_at": utc_now_iso()}
    ).eq("user_id", user_id).eq("id", session_id).execute()


def duplicate_vectors(
    supabase: Client,
    *,
    user_id: str,
    source_session_id: str,
    target_session_id: str,
) -> None:
    """Copy vector rows when the user explicitly saves as a new session."""
    resp = (
        supabase.table("vector_documents")
        .select("content,metadata,embedding,file_name")
        .eq("user_id", user_id)
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = list(resp.data or [])
    if not rows:
        return

    for i in range(0, len(rows), 10):
        batch = []
        for row in rows[i : i + 10]:
            metadata = dict(row.get("metadata") or {})
            metadata["user_id"] = user_id
            metadata["session_id"] = target_session_id
            batch.append(
                {
                    "user_id": user_id,
                    "session_id": target_session_id,
                    "content": row["content"],
                    "metadata": metadata,
                    "embedding": row["embedding"],
                    "file_name": row["file_name"],
                }
            )
        supabase.table("vector_documents").insert(batch).execute()


def save_session_as_new(
    supabase: Client,
    *,
    user_id: str,
    source_session_id: str,
    messages: list[dict[str, str]],
) -> str:
    """Insert a new saved session while keeping the existing one intact."""
    new_id = str(uuid.uuid4())
    ensure_session_row(
        supabase,
        user_id=user_id,
        session_id=new_id,
        messages=messages,
    )
    save_messages(
        supabase,
        user_id=user_id,
        session_id=new_id,
        messages=messages,
    )
    duplicate_vectors(
        supabase,
        user_id=user_id,
        source_session_id=source_session_id,
        target_session_id=new_id,
    )
    return new_id


def delete_session(supabase: Client, *, user_id: str, session_id: str) -> None:
    """Delete a user-owned session and its data."""
    supabase.table("vector_documents").delete().eq("user_id", user_id).eq(
        "session_id",
        session_id,
    ).execute()
    supabase.table("chat_messages").delete().eq("user_id", user_id).eq(
        "session_id",
        session_id,
    ).execute()
    supabase.table("chat_sessions").delete().eq("user_id", user_id).eq(
        "id",
        session_id,
    ).execute()


# ---------------------------------------------------------------------------
# Vector ingestion & retrieval
# ---------------------------------------------------------------------------
def process_pdf_uploads(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
    uploaded_files: list[Any],
) -> list[str]:
    """Load PDF files, split them, embed chunks, and store them in Supabase."""
    embeddings = get_embeddings()
    if embeddings is None:
        raise RuntimeError("PDF 임베딩을 위해 OPENAI_API_KEY를 설정해 주세요.")

    if not uploaded_files:
        return []

    ensure_session_row(
        supabase,
        user_id=user_id,
        session_id=session_id,
        messages=st.session_state.chat_history,
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    all_docs: list[Any] = []
    processed_names: list[str] = []

    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata["file_name"] = uploaded.name
                doc.metadata["session_id"] = session_id
                doc.metadata["user_id"] = user_id
            all_docs.extend(docs)
            processed_names.append(uploaded.name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    supabase.table("vector_documents").delete().eq("user_id", user_id).eq(
        "session_id",
        session_id,
    ).execute()

    if not all_docs:
        return processed_names

    splits = splitter.split_documents(all_docs)
    batch_size = 10
    for i in range(0, len(splits), batch_size):
        batch_docs = splits[i : i + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        vectors = embeddings.embed_documents(texts)
        rows = []
        for doc, vector in zip(batch_docs, vectors, strict=True):
            file_name = str(doc.metadata.get("file_name") or "unknown.pdf")
            rows.append(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": vector,
                    "file_name": file_name,
                }
            )
        supabase.table("vector_documents").insert(rows).execute()

    return processed_names


def retrieve_documents(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
    question: str,
    match_count: int = 8,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks with the Supabase RPC function."""
    embeddings = get_embeddings()
    if embeddings is None:
        return []

    query_embedding = embeddings.embed_query(question)
    try:
        resp = supabase.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter_user_id": user_id,
                "filter_session_id": session_id,
            },
        ).execute()
        return list(resp.data or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC retrieval failed, using fallback filtering: %s", exc)
        resp = (
            supabase.table("vector_documents")
            .select("content,metadata,file_name")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .limit(match_count)
            .execute()
        )
        return list(resp.data or [])


def build_rag_messages(
    *,
    question: str,
    retrieved_docs: list[dict[str, Any]],
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    """Build model messages for a RAG response."""
    if retrieved_docs:
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            metadata = doc.get("metadata") or {}
            name = doc.get("file_name") or metadata.get("file_name") or "문서"
            context_parts.append(f"[문서 {idx}: {name}]\n{doc.get('content', '')}")
        context = "\n\n".join(context_parts)
    else:
        context = "(검색된 문서가 없습니다.)"

    system = f"""{ANSWER_STYLE_SYSTEM}

[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=system), HumanMessage(content=question)]


# ---------------------------------------------------------------------------
# Streamlit state & UI
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "user": None,
        "session_id": str(uuid.uuid4()),
        "chat_history": [],
        "processed_names": [],
        "selected_session_id": None,
        "last_loaded_session_id": None,
        "show_vectordb": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_screen() -> None:
    """Clear the current screen and start a fresh unsaved working session."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.processed_names = []
    st.session_state.selected_session_id = None
    st.session_state.last_loaded_session_id = None
    st.session_state.show_vectordb = False


def logout() -> None:
    """Clear user and working-session state."""
    st.session_state.user = None
    reset_screen()


def current_user_id() -> str | None:
    """Return the logged-in user's id."""
    user = st.session_state.get("user")
    if not user:
        return None
    return str(user.get("id") or "")


def apply_custom_style() -> None:
    """Apply the color style used by the reference Streamlit UI."""
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render logo and centered title."""
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">기획예산처</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def render_auth(supabase: Client | None) -> bool:
    """Render login/signup UI and return True when authenticated."""
    if st.session_state.user:
        return True

    st.info("로그인 후 사용자별 세션과 문서를 사용할 수 있습니다.")
    if supabase is None:
        st.warning("Supabase 설정이 없어 로그인/회원가입을 사용할 수 없습니다.")
        return False

    login_tab, signup_tab = st.tabs(["로그인", "회원가입"])
    with login_tab:
        with st.form("login_form"):
            login_id = st.text_input("아이디", key="login_id")
            password = st.text_input("비밀번호", type="password", key="login_password")
            submitted = st.form_submit_button("로그인")
        if submitted:
            if not login_id.strip() or not password:
                st.warning("아이디와 비밀번호를 입력해 주세요.")
            else:
                try:
                    user = authenticate_user(
                        supabase,
                        login_id=login_id.strip(),
                        password=password,
                    )
                    if user is None:
                        st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
                    else:
                        st.session_state.user = user
                        reset_screen()
                        st.success("로그인되었습니다.")
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Login failed: %s", exc)
                    st.error(f"로그인 중 오류가 발생했습니다: {exc}")

    with signup_tab:
        with st.form("signup_form"):
            new_login_id = st.text_input("새 아이디")
            new_password = st.text_input("새 비밀번호", type="password")
            confirm_password = st.text_input("비밀번호 확인", type="password")
            submitted = st.form_submit_button("회원가입")
        if submitted:
            if not new_login_id.strip() or not new_password:
                st.warning("아이디와 비밀번호를 입력해 주세요.")
            elif new_password != confirm_password:
                st.warning("비밀번호 확인이 일치하지 않습니다.")
            elif len(new_password) < 8:
                st.warning("비밀번호는 8자 이상으로 입력해 주세요.")
            else:
                try:
                    user = create_user(
                        supabase,
                        login_id=new_login_id.strip(),
                        password=new_password,
                    )
                    st.session_state.user = {
                        "id": user["id"],
                        "login_id": user["login_id"],
                    }
                    reset_screen()
                    st.success("회원가입이 완료되었습니다.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Signup failed: %s", exc)
                    st.error(f"회원가입 중 오류가 발생했습니다: {exc}")

    return False


def render_saved_session(
    supabase: Client,
    *,
    user_id: str,
    session_id: str,
    show_success: bool = False,
) -> None:
    """Load a saved session into the current screen."""
    row = load_session(supabase, user_id=user_id, session_id=session_id)
    if not row:
        st.sidebar.warning("선택한 세션을 찾을 수 없습니다.")
        return

    st.session_state.session_id = row["id"]
    st.session_state.chat_history = load_messages(
        supabase,
        user_id=user_id,
        session_id=row["id"],
    )
    st.session_state.processed_names = current_file_names(
        supabase,
        user_id=user_id,
        session_id=row["id"],
    )
    st.session_state.selected_session_id = row["id"]
    st.session_state.last_loaded_session_id = row["id"]
    if show_success:
        st.sidebar.success(f"세션을 로드했습니다: {row.get('title', '제목 없음')}")


def render_sidebar(supabase: Client | None) -> bool:
    """Render sidebar controls and return whether the app can answer."""
    user_id = current_user_id()
    missing = missing_keys()

    with st.sidebar:
        st.radio("LLM 모델 선택", (MODEL_NAME,), index=0)

        if st.session_state.user:
            st.markdown(f"**사용자:** `{st.session_state.user['login_id']}`")
            if st.button("로그아웃", use_container_width=True):
                logout()
                st.rerun()

        if missing:
            st.warning(
                "다음 키를 Streamlit secrets 또는 `.env`에 설정해 주세요: "
                + ", ".join(missing)
            )

        st.markdown("### 세션 관리")
        sessions: list[dict[str, Any]] = []

        if supabase is not None and user_id:
            try:
                sessions = list_sessions(supabase, user_id=user_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session list failed: %s", exc)
                st.error(f"세션 목록을 불러오지 못했습니다: {exc}")

        options = ["새 세션"] + [f"{s['title']} ({s['id'][:8]})" for s in sessions]
        option_ids = [None] + [s["id"] for s in sessions]
        current_index = 0
        if st.session_state.selected_session_id in option_ids:
            current_index = option_ids.index(st.session_state.selected_session_id)

        selected_label = st.selectbox("세션 선택", options, index=current_index)
        selected_id = option_ids[options.index(selected_label)]
        st.session_state.selected_session_id = selected_id

        if (
            supabase is not None
            and user_id
            and selected_id
            and selected_id != st.session_state.last_loaded_session_id
        ):
            render_saved_session(supabase, user_id=user_id, session_id=selected_id)
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("세션저장", use_container_width=True):
                if supabase is None or not user_id:
                    st.error("로그인과 Supabase 설정이 필요합니다.")
                elif not st.session_state.chat_history:
                    st.warning("저장할 대화가 없습니다.")
                else:
                    new_id = save_session_as_new(
                        supabase,
                        user_id=user_id,
                        source_session_id=st.session_state.session_id,
                        messages=st.session_state.chat_history,
                    )
                    st.session_state.session_id = new_id
                    st.session_state.selected_session_id = new_id
                    st.session_state.last_loaded_session_id = new_id
                    st.success("새 세션으로 저장했습니다.")
                    st.rerun()
        with col2:
            if st.button("세션로드", use_container_width=True):
                if supabase is None or not user_id:
                    st.error("로그인과 Supabase 설정이 필요합니다.")
                elif not selected_id:
                    st.warning("로드할 세션을 선택해 주세요.")
                else:
                    render_saved_session(
                        supabase,
                        user_id=user_id,
                        session_id=selected_id,
                        show_success=True,
                    )
                    st.rerun()

        col3, col4 = st.columns(2)
        with col3:
            if st.button("세션삭제", use_container_width=True):
                if supabase is None or not user_id:
                    st.error("로그인과 Supabase 설정이 필요합니다.")
                elif not selected_id:
                    st.warning("삭제할 세션을 선택해 주세요.")
                else:
                    delete_session(supabase, user_id=user_id, session_id=selected_id)
                    reset_screen()
                    st.success("선택한 세션을 삭제했습니다.")
                    st.rerun()
        with col4:
            if st.button("화면초기화", use_container_width=True):
                reset_screen()
                st.rerun()

        if st.button("vectordb", use_container_width=True):
            st.session_state.show_vectordb = not st.session_state.show_vectordb

        st.markdown("### PDF 문서")
        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기", use_container_width=True):
            if supabase is None or not user_id:
                st.error("로그인과 Supabase 설정이 필요합니다.")
            elif not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    with st.spinner("PDF를 임베딩하고 Supabase에 저장 중입니다..."):
                        names = process_pdf_uploads(
                            supabase,
                            user_id=user_id,
                            session_id=st.session_state.session_id,
                            uploaded_files=list(uploads),
                        )
                        st.session_state.processed_names = names
                        save_messages(
                            supabase,
                            user_id=user_id,
                            session_id=st.session_state.session_id,
                            messages=st.session_state.chat_history,
                        )
                    st.success("PDF 처리가 완료되었습니다.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF processing failed: %s", exc)
                    st.error(f"PDF 처리 중 오류가 발생했습니다: {exc}")

        if st.session_state.show_vectordb:
            st.markdown("**현재 vectordb 파일명**")
            if supabase is not None and user_id:
                try:
                    names = current_file_names(
                        supabase,
                        user_id=user_id,
                        session_id=st.session_state.session_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    names = []
                    st.error(f"vectordb 조회 실패: {exc}")
                if names:
                    for name in names:
                        st.text(f"- {name}")
                else:
                    st.caption("현재 세션에 저장된 파일이 없습니다.")

        st.text(
            "설정\n"
            f"모델: {MODEL_NAME}\n"
            f"세션 ID: {st.session_state.session_id[:8]}\n"
            f"대화 메시지 수: {len(st.session_state.chat_history)}\n"
            f"처리된 PDF 파일 수: {len(st.session_state.processed_names)}"
        )

    return not missing and supabase is not None and bool(user_id)


def render_chat_history() -> None:
    """Display current chat messages."""
    for msg in st.session_state.chat_history:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(remove_separators(msg.get("content", "")))


def answer_question(supabase: Client, *, user_id: str, question: str) -> str:
    """Stream a RAG answer into the active assistant chat message."""
    llm = get_llm()
    if llm is None:
        return "# 안내\n\nOPENAI_API_KEY를 Streamlit secrets 또는 `.env`에 설정해 주세요."

    memory_text = _format_memory_block(st.session_state.chat_history[:-1])
    docs = retrieve_documents(
        supabase,
        user_id=user_id,
        session_id=st.session_state.session_id,
        question=question,
    )
    messages = build_rag_messages(
        question=question,
        retrieved_docs=docs,
        memory_text=memory_text,
    )

    placeholder = st.empty()
    acc = ""
    for chunk in llm.stream(messages):
        piece = getattr(chunk, "content", "") or ""
        if piece:
            acc += piece
            placeholder.markdown(remove_separators(acc) + "▌")

    answer = remove_separators(acc) or "(응답이 비어 있습니다.)"
    placeholder.markdown(answer)
    return answer


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="기획예산처 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    init_session_state()
    apply_custom_style()
    render_header()

    supabase = get_supabase()
    ready = render_sidebar(supabase)

    if not render_auth(supabase):
        return

    render_chat_history()

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    user_id = current_user_id()
    with st.chat_message("assistant"):
        if not ready or supabase is None or not user_id:
            missing = missing_keys()
            detail = ", ".join(missing) if missing else "로그인 또는 Supabase 설정"
            answer = (
                "# 안내\n\n"
                "앱을 사용하려면 다음 설정을 확인해 주세요: "
                + detail
            )
            st.markdown(answer)
        else:
            try:
                answer = answer_question(supabase, user_id=user_id, question=user_input)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Answer generation failed: %s", exc)
                answer = f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
                st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if ready and supabase is not None and user_id:
        try:
            save_messages(
                supabase,
                user_id=user_id,
                session_id=st.session_state.session_id,
                messages=st.session_state.chat_history,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-save failed: %s", exc)
            st.warning(f"자동 저장 중 오류가 발생했습니다: {exc}")


if __name__ == "__main__":
    main()
