# 1) 필요한 도구를 불러온다.
# os: 컴퓨터 환경 변수(예: API 키)를 읽기 위한 도구
# warnings: 화면에 뜨는 불필요한 경고를 잠시 숨기기 위한 도구
# Path: 파일 경로를 다루는 도구
# streamlit: 웹 화면을 만드는 도구
# load_dotenv: .env 파일에 저장된 값(예: API 키)을 불러오는 도구
# Chroma: 문장을 벡터로 저장하고 빠르게 찾는 DB 도구
# ChatAnthropic: Anthropic 모델을 호출하는 도구
# HuggingFaceEmbeddings: 문장을 숫자 벡터로 바꾸는 도구
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 특정 라이브러리 경고가 너무 많이 뜨는 것을 막는다.
# 이것은 앱 기능과는 별개로, 경고만 숨기는 용도다.
warnings.filterwarnings(
    "ignore",
    message=r".*max_pixels.*PaddleOCRVLImageProcessorKwargs.*",
    category=UserWarning,
)

# .env 파일의 값들을 현재 프로그램 안으로 불러온다.
# 예: ANTHROPIC_API_KEY 같은 비밀 값을 쉽게 읽을 수 있게 해 준다.
load_dotenv(override=True)

from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings

# 문서 인덱싱 함수도 같은 앱에서 사용하기 위해 불러온다.
from ingest import create_vector_db

# .env 파일을 다시 읽어오면 기존 값이 덮어써질 수 있으니 한 번 더 로드한다.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# 2) DB와 문서 파일이 저장될 위치를 정한다.
# chroma_db: 문장 벡터를 저장하는 폴더
# documents: 사용자가 업로드한 문서를 보관하는 폴더
PERSIST_DIRECTORY = "chroma_db"
DOCUMENTS_DIR = BASE_DIR / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True, parents=True)
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".csv", ".docx", ".xlsx", ".xls",
    ".odt", ".ods", ".odp", ".hwp"
}
def find_latest_persist_directory(base_dir: str = PERSIST_DIRECTORY):
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True, parents=True)

    candidates = []
    if base_path.exists():
        candidates.append(base_path)

    for candidate in sorted(
        base_path.parent.glob(f"{base_path.name}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if candidate.is_dir():
            candidates.append(candidate)

    if not candidates:
        return str(base_path)

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)


def cleanup_old_chroma_folders(base_dir: str = PERSIST_DIRECTORY):
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True, parents=True)

    removed = []
    for candidate in sorted(base_path.parent.glob(f"{base_path.name}_*"), key=lambda p: p.name):
        if not candidate.is_dir():
            continue
        try:
            shutil.rmtree(candidate)
            removed.append(candidate.name)
        except Exception as exc:
            removed.append(f"{candidate.name} (실패: {exc})")

    return removed


if "persist_directory" not in st.session_state:
    st.session_state["persist_directory"] = find_latest_persist_directory()
if "db_build_log" not in st.session_state:
    st.session_state["db_build_log"] = []

# 3) 웹 페이지 기본 설정
# page_title: 브라우저 탭에 보이는 제목
# page_icon: 탭 왼쪽의 아이콘
st.set_page_config(
    page_title="RAG 문서 챗봇",
    page_icon="📚"
)

# 4) 화면 위쪽에 제목과 짧은 설명을 보여준다.
# 이 부분은 사용자가 앱에 들어왔을 때 가장 먼저 보는 화면이다.
st.title("📚 RAG 기반 문서 챗봇")
st.caption("등록된 문서 내용에 근거하여 답변합니다.")

st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] .stButton > button,
    div[data-testid="stSidebar"] button[kind="secondary"],
    div[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #d93025 !important;
        color: #ffffff !important;
        border: 1px solid #d93025 !important;
        border-radius: 6px !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        line-height: 1.1 !important;
        padding: 0.18rem 0.5rem !important;
        min-width: 48px !important;
        width: auto !important;
        max-width: 70px !important;
        height: 28px !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        vertical-align: middle !important;
        text-align: center !important;
        margin: 0 !important;
    }
    div[data-testid="stSidebar"] .stButton > button:hover,
    div[data-testid="stSidebar"] button[kind="secondary"]:hover,
    div[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #b3261e !important;
        color: #ffffff !important;
        border-color: #b3261e !important;
    }
    div[data-testid="stSidebar"] .stButton > button:focus,
    div[data-testid="stSidebar"] button[kind="secondary"]:focus,
    div[data-testid="stSidebar"] button[kind="primary"]:focus {
        box-shadow: 0 0 0 0.15rem rgba(217, 48, 37, 0.28) !important;
        outline: none !important;
    }
    div[data-testid="stSidebar"] .stButton > button span,
    div[data-testid="stSidebar"] button[kind="secondary"] span,
    div[data-testid="stSidebar"] button[kind="primary"] span {
        font-size: 11px !important;
        line-height: 1.1 !important;
        display: inline-block !important;
        vertical-align: middle !important;
    }
    div[data-testid="stSidebar"] .st-cq {
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.38) !important;
        border-radius: 10px !important;
        background: rgba(15, 23, 42, 0.18) !important;
        margin-top: 0.4rem !important;
    }
    div[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        padding: 0.25rem 0.5rem 0.4rem 0.5rem !important;
    }
    div[data-testid="stSidebar"] [data-testid="stCodeBlock"] {
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        padding: 0.5rem 0.6rem !important;
        font-size: 10.5px !important;
        line-height: 1.45 !important;
        white-space: pre-wrap !important;
    }
    div[data-testid="stSidebar"] pre {
        background: transparent !important;
        color: inherit !important;
        margin: 0 !important;
        white-space: pre-wrap !important;
    }
    div[data-testid="stSidebar"] code {
        font-family: "Consolas", "SFMono-Regular", monospace !important;
        background: transparent !important;
    }
    div[data-testid="stSidebar"] summary {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }
    div[role="tablist"] {
        gap: 0.6rem !important;
        background: rgba(15, 23, 42, 0.44) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 16px !important;
        padding: 0.45rem !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
    }
    div[role="tab"] {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 1.3rem !important;
        border-radius: 12px !important;
        border: 1px solid transparent !important;
        background: rgba(148, 163, 184, 0.08) !important;
        color: #e2e8f0 !important;
        transition: all 0.2s ease !important;
    }
    div[role="tab"][aria-selected="true"] {
        color: #ffffff !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.28) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }
    div[role="tab"]:nth-child(1)[aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    }
    div[role="tab"]:nth-child(2)[aria-selected="true"] {
        background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    }
    div[role="tab"]:nth-child(3)[aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    }
    div[role="tab"]:hover {
        border-color: rgba(255,255,255,0.08) !important;
        background: rgba(148, 163, 184, 0.12) !important;
    }
    .tab-panel {
        background: linear-gradient(180deg, rgba(15,23,42,0.62), rgba(15,23,42,0.42));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1.25rem 1.4rem 1.4rem 1.4rem;
        margin-top: 0.9rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.22);
    }
    .panel-chat { border-top: 3px solid #3b82f6; }
    .panel-tests { border-top: 3px solid #22c55e; }
    .panel-results { border-top: 3px solid #a78bfa; }
    .section-title {
        margin: 0 0 0.8rem 0;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
    }
    .section-subtitle {
        color: #cbd5e1;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 문서 목록을 보여주기 위해 documents 폴더 안의 파일을 읽어오는 함수
# 파일 이름 기준으로 정렬해서 깔끔하게 보여준다.
def list_document_files():
    return sorted(
        [
            p for p in DOCUMENTS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def load_test_cases():
    test_cases_path = BASE_DIR / "data" / "test_cases.json"
    if not test_cases_path.exists():
        return []
    try:
        with open(test_cases_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def get_latest_report_path():
    reports_dir = BASE_DIR / "reports"
    if not reports_dir.exists():
        return None
    report_files = sorted(reports_dir.glob("evaluation_report_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return report_files[0] if report_files else None


def run_judge_agent():
    judge_script = BASE_DIR / "judge_agent.py"
    if not judge_script.exists():
        return False, "judge_agent.py 파일을 찾을 수 없습니다."

    try:
        result = subprocess.run(
            [sys.executable, str(judge_script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        combined = "\n".join(part for part in [stdout, stderr] if part)
        if result.returncode == 0:
            return True, combined or "평가 완료"
        return False, combined or f"Judge 평가 실패 (코드: {result.returncode})"
    except subprocess.TimeoutExpired:
        return False, "Judge 평가 시간이 초과되었습니다."
    except Exception as exc:
        return False, f"Judge 평가 실행 중 오류: {exc}"


# 업로드된 파일을 실제로 documents 폴더에 저장하는 함수
# uploaded_files는 사용자가 선택한 파일 목록이다.
def save_uploaded_documents(uploaded_files):
    saved = []
    for uploaded_file in uploaded_files:
        destination = DOCUMENTS_DIR / uploaded_file.name
        destination.write_bytes(uploaded_file.getvalue())
        saved.append(uploaded_file.name)
    return saved


# 문서 하나를 삭제하는 함수
# 삭제할 파일 경로를 찾은 뒤 파일을 실제로 지운다.
def delete_document_file(file_name: str):
    target = DOCUMENTS_DIR / file_name
    if target.exists():
        target.unlink()


# 사이드바에 문서 관리 영역을 만든다.   
# 사용자는 여기서 파일을 올리거나, 목록을 보고, 삭제할 수 있다.
st.sidebar.header("📁 문서 관리")

if st.sidebar.button("Judge 평가", type="primary", use_container_width=True):
    with st.sidebar:
        with st.spinner("Judge Agent를 실행 중입니다..."):
            ok, message = run_judge_agent()
    if ok:
        st.sidebar.success("Judge 평가 완료")
        st.session_state["judge_output"] = message
    else:
        st.sidebar.error("Judge 평가 실패")
        st.session_state["judge_output"] = message

if "judge_output" in st.session_state:
    with st.sidebar.expander("Judge 실행 로그", expanded=False):
        st.code(st.session_state["judge_output"], language="text")

with st.sidebar.expander("DB 상태", expanded=True):
    st.caption(f"활성 DB 경로: {st.session_state.get('persist_directory', PERSIST_DIRECTORY)}")
    if st.button("기존 Chroma 폴더 정리", key="cleanup_chroma_folders", use_container_width=True):
        removed = cleanup_old_chroma_folders()
        st.session_state["vector_db"] = None
        st.session_state["persist_directory"] = find_latest_persist_directory()
        if removed:
            st.sidebar.success(f"정리 완료: {', '.join(removed)}")
        else:
            st.sidebar.info("정리할 이전 Chroma 폴더가 없습니다.")
        st.rerun()

    if st.session_state.get("db_build_log"):
        st.write("최근 DB 로그")
        st.code("\n".join(st.session_state["db_build_log"][-20:]), language="text")

if "upload_counter" not in st.session_state:
    st.session_state["upload_counter"] = 0
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None


# 업로드할 파일이 선택되면 실제로 저장할지 확인하는 창을 띄운다.
# 사용자가 실수로 업로드를 누르는 일을 줄이기 위한 안전장치다.
def upload_confirm_dialog():
    pending_files = st.session_state.get("pending_uploads") or []
    if not pending_files:
        return

    st.sidebar.warning("다음 파일을 documents 폴더에 추가/덮어쓰시겠습니까?")
    st.sidebar.code("\n".join(file.name for file in pending_files))

    col1, col2 = st.sidebar.columns(2)
    with col1:
        # 적용 버튼: 파일을 실제로 저장하고 화면을 다시 불러온다.
        if st.button("적용", use_container_width=True, key="apply_upload_confirm"):
            saved_files = save_uploaded_documents(pending_files)
            st.session_state["pending_uploads"] = []
            st.session_state["last_processed_upload_key"] = tuple(
                sorted((file.name, getattr(file, "size", 0)) for file in pending_files)
            )
            st.session_state["upload_counter"] += 1
            st.sidebar.success(f"저장 완료: {', '.join(saved_files)}")
            st.session_state["force_refresh"] = True
            st.rerun()
    with col2:
        # 취소 버튼: 저장하지 않고 그냥 뒤로 돌아간다.
        if st.button("취소", use_container_width=True, key="cancel_upload_confirm"):
            st.session_state["pending_uploads"] = []
            st.session_state["last_processed_upload_key"] = tuple(
                sorted((file.name, getattr(file, "size", 0)) for file in pending_files)
            )
            st.rerun()


# 파일 업로드 위젯을 만든다.
# 여러 파일을 동시에 선택할 수 있고, txt/md/pdf/csv/docx/xlsx/hwp 같은 파일을 받는다.
upload_widget_key = f"doc_uploader_{st.session_state['upload_counter']}"
uploaded_files = st.sidebar.file_uploader(
    "문서를 추가하거나 덮어쓰기",
    type=["txt", "md", "pdf", "csv", "docx", "xlsx", "xls", "odt", "ods", "odp", "hwp"],
    accept_multiple_files=True,
    key=upload_widget_key,
)
if uploaded_files:
    # 새로 선택한 파일들인지 비교해서, 이전에 처리한 파일이 아니면 잠시 보관한다.
    current_upload_key = tuple(sorted((file.name, getattr(file, "size", 0)) for file in uploaded_files))
    if st.session_state.get("last_processed_upload_key") != current_upload_key:
        st.session_state["pending_uploads"] = uploaded_files
        st.session_state["last_processed_upload_key"] = current_upload_key

# 파일을 저장한 뒤에는 확인 창이 다시 떠서 같은 파일이 계속 반복되지 않도록 처리한다.
upload_confirm_dialog()

# 적용 후 화면을 다시 불러오도록 강제 새로고침한다.
if st.session_state.get("force_refresh"):
    st.session_state["force_refresh"] = False
    st.rerun()


# 문서 목록을 접을 수 있는 영역으로 보여준다.
with st.sidebar.expander("문서 목록", expanded=True):
    files = list_document_files()
    if not files:
        st.caption("documents 폴더가 비어 있습니다.")
    else:
        for file_path in files:
            col_name, col_delete = st.columns([4.5, 1.1], gap="small")
            with col_name:
                # 파일 이름을 보이기 좋게 가운데 정렬 스타일로 표시한다.
                st.markdown(
                    f"<div style='height:28px; display:flex; align-items:center; font-size:15px; line-height:1.2; vertical-align:middle;'>{file_path.name}</div>",
                    unsafe_allow_html=True,
                )
            with col_delete:
                # 파일을 삭제하는 빨간 버튼
                if st.button("삭제", key=f"delete_{file_path.name}", use_container_width=True):
                    delete_document_file(file_path.name)
                    st.rerun()


# 문서가 추가/삭제된 뒤 DB를 다시 만들기 위해 버튼을 둔다.
# 이 버튼을 누르면 새 문서들이 챗봇 검색 대상이 된다.
if st.sidebar.button("DB에 새로 적용", type="primary", use_container_width=True):
    try:
        import gc

        rebuild_log = []

        def append_log(message: str):
            rebuild_log.append(message)
            st.session_state["db_build_log"] = rebuild_log.copy()

        st.session_state["vector_db"] = None
        gc.collect()
        st.cache_resource.clear()

        with st.spinner("문서를 다시 인덱싱하는 중입니다..."):
            document_count, chunk_count, new_db_path = create_vector_db(
                persist_directory=st.session_state.get("persist_directory", PERSIST_DIRECTORY),
                log_callback=append_log,
            )

        st.session_state["persist_directory"] = new_db_path
        st.session_state["vector_db"] = None
        st.session_state["db_build_log"] = rebuild_log.copy()
        gc.collect()
        st.cache_resource.clear()

        st.sidebar.success(f"DB 반영 완료: 문서 {document_count}개, 청크 {chunk_count}개")
        with st.sidebar.expander("DB 반영 상세 로그", expanded=True):
            st.code("\n".join(rebuild_log), language="text")
        st.rerun()
    except Exception as e:
        st.session_state["db_build_log"] = [f"[오류] {e}"]
        st.sidebar.error(f"DB 반영 실패: {e}")
        with st.sidebar.expander("오류 로그", expanded=True):
            st.code(str(e), language="text")


# 6) 벡터 DB를 만들거나 불러오는 함수
# RAG 재인덱싱 중 기존 Chroma 파일이 잠겨서 삭제되지 않는 문제를 막기 위해,
# 이 객체는 Streamlit 캐시에 오래 남기지 않는다.
# 매번 새로 연결해서 안전하게 재생성한다.
def get_vector_db():
    persist_directory = st.session_state.get("persist_directory", PERSIST_DIRECTORY)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag_documents"
    )


# Anthropic 모델 이름을 안정적으로 정하는 함수
# 예전 잘못된 모델명은 피하고, 현재 사용 가능한 모델을 선택한다.
def resolve_model_name():
    invalid_values = {"claude-3-5-sonnet-20241022"}
    candidates = [
        os.getenv("ANTHROPIC_MODEL"),
        os.getenv("EVALUATOR_MODEL"),
        "claude-sonnet-5",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        value = candidate.strip()
        if not value or value in invalid_values:
            continue
        os.environ["ANTHROPIC_MODEL"] = value
        return value

    os.environ["ANTHROPIC_MODEL"] = "claude-sonnet-5"
    return "claude-sonnet-5"


# 7) Anthropic LLM을 준비하는 함수
# 이 함수는 실제로 AI 모델을 시작하는 역할을 한다.
# API 키가 있어야 하고, 모델 이름도 정해져 있어야 답변을 받을 수 있다.
@st.cache_resource
def get_llm():
    # 환경 변수 이름이 조금 다를 수 있어서 여러 후보를 순서대로 확인한다.
    candidates = ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYf", "ANTHROPIC_API"]
    raw_key = None
    found_name = None
    for name in candidates:
        v = os.getenv(name)
        if v:
            raw_key = v
            found_name = name
            break

    # API 키가 없으면 사용자에게 어떤 값을 넣어야 하는지 알려준다.
    if not raw_key:
        raise ValueError(
            "ANTHROPIC_API_KEY가 비어 있습니다. .env 파일에 유효한 Anthropic API 키를 넣고 환경변수 이름이 'ANTHROPIC_API_KEY'인지 확인하세요."
        )

    # 앞뒤 쌍따옴표 또는 작은따옴표가 붙어 있으면 정리한다.
    api_key = raw_key.strip().strip('"').strip("'")
    model_name = resolve_model_name()

    # 키 값 자체는 너무 길어서 일부만 보여준다. (보안상)
    print("앱이 시작되었습니다. LLM을 준비합니다... (env)", found_name, "model=", model_name)

    # 실제 AI 모델 객체를 만든다.
    return ChatAnthropic(
        api_key=api_key,
        model=model_name,
        max_tokens=1024
    )


# 8) 사용자의 질문을 받아서 답변을 만드는 핵심 함수
# question: 사용자가 물어본 질문 문자열
# 반환값: (답변 문자열, 참고 문서 목록)
def answer_question(question: str):
    # 8-1. 문서 DB와 AI 모델을 준비한다.
    vector_db = st.session_state.get("vector_db")
    if vector_db is None:
        vector_db = get_vector_db()
        st.session_state["vector_db"] = vector_db
    llm = get_llm()

    # 8-2. 질문과 비슷한 문서를 벡터 DB에서 3개 찾는다.
    # 가장 관련 있는 문서 3개를 뽑아서 답변 근거로 사용한다.
    retrieved_docs = vector_db.similarity_search(
        question,
        k=3
    )

    # 8-3. 찾은 문서가 없으면 바로 알려준다.
    if not retrieved_docs:
        return "관련 문서를 찾지 못했습니다.", []

    # 8-4. 문서 조각들을 하나의 긴 설명으로 합친다.
    # 각 문서 앞에 [출처: 파일명]을 붙여서 모델이 어디서 나온 내용인지 알 수 있게 한다.
    context = "\n\n".join(
        [
            f"[출처: {doc.metadata.get('source', '알 수 없음')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    # 8-5. 모델에게 보낼 프롬프트를 만든다.
    # 여기서는 문서 안에 있는 내용만 기반으로 답하게 강하게 지시한다.
    prompt = f"""
당신은 교육과정 안내 문서 챗봇입니다.

반드시 아래 제공된 문서 내용만 근거로 답변하십시오.
문서에 없는 내용은 추측하지 말고,
"제공된 문서에서는 확인할 수 없습니다."라고 답하십시오.

[문서 내용]
{context}

[사용자 질문]
{question}

[답변 작성 원칙]
1. 한국어로 답변한다.
2. 핵심 답변을 먼저 제시한다.
3. 문서 근거가 있으면 자연스럽게 설명한다.
4. 답변 끝에 참고한 파일명을 표시한다.
"""

    # 8-6. 모델에 질문을 보내고 답변을 받는다.
    # 만약 오류가 나면 그냥 에러 내용을 보여준다.
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        return str(e), []

    # 8-7. 어떤 파일을 참고했는지 중복 없이 정리한다.
    sources = list(
        {
            doc.metadata.get("source", "알 수 없음")
            for doc in retrieved_docs
        }
    )

    # 8-8. 최종 답변과 참고 문서 목록을 돌려준다.
    return response.content, sources


# 9) 대화 기록을 저장할 공간을 만든다.
# Streamlit은 새로고침 때 상태가 사라질 수 있어서
# session_state를 사용해 이전 대화 내용이 유지되도록 한다.
if "messages" not in st.session_state:
    st.session_state.messages = []

chat_tab, test_cases_tab, results_tab = st.tabs(["대화", "테스트 케이스", "평가 결과"])

with chat_tab:
    st.markdown('<div class="tab-panel panel-chat">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗨️ 대화</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">문서 기반 질문에 답하고, 대화 내용을 이어갑니다.</div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("문서에 대해 질문해 보세요.")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("관련 문서를 찾고 있습니다..."):
                answer, sources = answer_question(question)
            st.markdown(answer)
            if sources:
                st.caption(f"참고 문서: {', '.join(sources)}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown('</div>', unsafe_allow_html=True)

with test_cases_tab:
    st.markdown('<div class="tab-panel panel-tests">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧪 테스트 케이스</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">등록된 평가 시나리오를 한눈에 확인하고 검증합니다.</div>', unsafe_allow_html=True)

    test_cases = load_test_cases()
    if not test_cases:
        st.warning("data/test_cases.json 파일을 찾을 수 없거나 비어 있습니다.")
    else:
        rows = []
        for tc in test_cases:
            rows.append({
                "TC_ID": tc.get("tc_id", ""),
                "유형": tc.get("type", ""),
                "질문": tc.get("question", ""),
                "기대 키워드": ", ".join(tc.get("expected_keywords", [])),
                "기대 출처": tc.get("expected_source", "")
            })
        st.dataframe(rows, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with results_tab:
    st.markdown('<div class="tab-panel panel-results">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 평가 결과</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">최근 리포트와 평가 요약을 빠르게 확인할 수 있습니다.</div>', unsafe_allow_html=True)

    latest_report = get_latest_report_path()
    if latest_report is None:
        st.info("아직 생성된 평가 결과가 없습니다. 먼저 Judge 평가 버튼을 눌러 실행해 주세요.")
    else:
        st.caption(f"최근 보고서: {latest_report.name}")
        markdown_text = latest_report.read_text(encoding="utf-8")
        st.markdown(markdown_text)
    st.markdown('</div>', unsafe_allow_html=True)
