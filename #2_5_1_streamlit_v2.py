

# =========================================================
# Streamlit 기반 스마트폰 과의존 실태조사 RAG 챗봇 v5
# 
# 개선사항:
# 1. Validate 기반 회복 루프 (PASS/FAIL 분기)
# 2. Clarify 대화형 중단 + 상태 저장
# 3. Query Rewrite / Rerank 노드
# 4. 안전 가드 노드 (Context Sanitize, Safety Check)
# =========================================================
import streamlit as st
import json
import re
import os
import pandas as pd
from typing import Dict, Any, List, Optional, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="스마트폰 과의존 실태조사 챗봇 v5",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 커스텀 CSS
# =========================================================
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    .guide-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .guide-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .guide-item {
        background: rgba(255,255,255,0.15);
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .guide-item:last-child {
        margin-bottom: 0;
    }
    
    .status-box {
        background-color: #e3f2fd;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .retry-badge {
        background-color: #fff3e0;
        color: #e65100;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .validation-pass {
        color: #2e7d32;
        font-weight: 600;
    }
    
    .validation-fail {
        color: #c62828;
        font-weight: 600;
    }
    
    h1 {
        color: #1a237e;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 상수 설정
# =========================================================
YEAR_TO_FILENAME = {
    2020: "2020년_스마트폰_과의존_실태조_사보고서.pdf",
    2021: "2021년_스마트_과의존_실태조사_보고서.pdf",
    2022: "2022년_스마트폰_과의존_실태조사_보고서.pdf",
    2023: "2023년_스마트폰_과의존실태조사_최종보고서.pdf",
    2024: "2024_스마트폰_과의존_실태조사_본_보고서.pdf",
}
ALLOWED_FILES = list(YEAR_TO_FILENAME.values())

BOT_IDENTITY = """2020~2024년 스마트폰 과의존 실태조사 보고서 분석 시스템입니다.

**제공 가능한 정보:**
- 연도별 스마트폰 과의존 위험군 비율 및 추이
- 대상별(유아동, 청소년, 성인, 60대) 과의존 현황
- 학령별(초/중/고/대학생) 세부 분석
- 과의존 관련 요인 분석 (SNS, 숏폼, 게임 이용 등)
- 조사 방법론 및 표본 설계 정보
"""

# Hugging Face 설정
HF_REPO_ID = "Rosaldowithbaek/smartphone-addiction-chroma-db"
LOCAL_DB_PATH = "./chroma_db_store"

# 검색 파라미터 (기본값 / 재시도용)
DEFAULT_K_PER_QUERY = 10
DEFAULT_TOP_PARENTS = 15
DEFAULT_TOP_PARENTS_PER_FILE = 5

RETRY_K_PER_QUERY = 15
RETRY_TOP_PARENTS = 20
RETRY_TOP_PARENTS_PER_FILE = 7

MAX_CHUNKS_PER_PARENT = 5
MAX_CHARS_PER_DOC = 10000
SUMMARY_TYPES = ["page_summary", "table_summary"]

MAX_RETRY_COUNT = 2

# 키워드 분류
TARGET_KEYWORDS = {
    # 기존 키 유지하면서, 보고서 표현/연령 표기/변형을 최대한 흡수함
    "대상": [
        # 전체/모집단 표현
        "전체", "전국", "전국민", "전국 가구", "모집단", "조사대상", "조사 모집단",
        "스마트폰 이용자", "스마트폰(인터넷) 이용자", "이용자",

        # 보고서에서 쓰는 대상 구분(연령대 큰 덩어리)
        "유아동", "영유아", "유아", "아동", "어린이", "만 3~9세", "만3~9세", "만 3∼9세",
        "청소년", "10대", "십대", "10 대", "만 10~19세", "만10~19세", "만 10∼19세",
        "성인", "만 20~59세", "만20~59세", "만 20∼59세",
        "60대", "고령층", "고령자", "만 60~69세", "만60~69세", "만 60∼69세",
    ],
    "위험군": [    # 보고서 분류 체계(고위험/잠재적/일반 + 과의존위험군=고위험+잠재적)
    "스마트폰 과의존", "과의존", "과다이용",
    "과의존 수준", "과의존 수준별",
    "과의존위험군", "과의존 위험군", "스마트폰 과의존위험군", "스마트폰 과의존 위험군",
    "고위험군", "고 위험군",
    "잠재적위험군", "잠재적 위험군", "잠재 위험군",
    "일반사용자군", "일반 사용자군", "일반군",
    ],

    "학령": [
        # 기존 + 변형/동의 표현
        "유치원생", "유치원", "미취학", "미취학 아동",
        "초등학생", "초등", "초등생", "초등학교", "초등 저학년", "초등 고학년",
        "중학생", "중등", "중학교", "중등학생",
        "고등학생", "고등", "고등학교", "고등생",
        "대학생", "대학", "대학교", "대학 재학생",
    ],

    "성별": [
        "남성", "여성", "남자", "여자",
        "남", "여", "남녀", "성별",
    ],
    "지역": ["대도시", "중소도시", "읍면지역", "읍/면"],
    "위험군": ["과의존위험군", "일반사용자군", "고위험군", "잠재적위험군"],
}

TOPIC_KEYWORDS = {
    "콘텐츠": [
        # (핵심) 보고서 ‘콘텐츠 이용정도’ 26개 분류 기반
        "SNS", "이메일", "메신저", "새로운 친구만남", "새로운 친구 만남",
        "생활관리", "건강관리", "화상회의", "원격근무", "화상회의/원격근무",
        "쇼핑", "쇼핑(상품/서비스)", "상품/서비스 판매", "금융거래", "투자 및 자산관리",
        "게임", "영화/TV/동영상", "영화", "TV", "동영상",
        "음악", "라디오", "팟캐스트", "라디오/팟캐스트",
        "웹툰", "웹소설", "독서", "웹툰/웹소설/독서",
        "사진", "촬영", "편집", "사진(촬영 편집) 및 그림", "그림",
        "여행",
        "성인용 콘텐츠", "사행성 게임",
        "뉴스보기", "뉴스 보기",
        "학업/업무용 검색", "학업", "업무", "업무용 검색",
        "관심사(취미)검색", "취미", "관심사 검색",
        "지도", "네비게이션", "지도 및 네비게이션",
        "교육", "원격수업", "E-러닝", "인터넷강의", "교육(원격수업/E-러닝/인터넷강의)",
        "생성형 AI서비스", "생성형AI", "정보검색", "문서보조", "번역",

        # 온라인 동영상 서비스(OVS)·숏폼 관련(보고서 표/그림 표현 반영)
        "온라인 동영상 서비스", "온라인동영상서비스", "동영상 서비스",
        "숏폼", "쇼츠", "릴스", "숏폼 플랫폼",

        # 보고서에서 제시된 ‘주 이용 숏폼 플랫폼(1순위)’ 항목(고유명)
        "유튜브 쇼츠", "인스타그램 릴스", "틱톡", "카카오톡", "네이버 클립",

    ],

    "지표": [
        # 핵심 결과 지표
        "과의존률", "과의존 위험군 비율", "과의존위험군 비율",
        "고위험군 비율", "잠재적위험군 비율", "일반사용자군 비율",
        "비율", "률", "%", "%p", "단위:%", "단위: %",

        # 척도/점수
        "점수", "총점", "평균", "4점 만점", "4점만점",
        "기준점수", "기준 점수", "역문항", "역척도",

        # 비교/추이 표현
        "연도별", "전년대비", "최근 1년", "최근1년", "추이", "증가", "감소", "변화",
        "대상별", "연령별", "연령대별", "성별", "학령별", "도시규모별",

        # 과다이용·조절
        "과다이용", "과다이용 인식",
        "이용시간", "이용 시간", "이용시간 조절", "이용시간 조절 어려움",
        "본인 의지대로 조절", "조절 어려움 정도",
    ],

    "요인": [
        # 과의존 3요인(보고서 공통 프레임)
        "조절실패", "현저성", "문제적 결과",

        # 심층문항/경험 영역(목차·표 제목 기반)
        "사용조절", "사용 조절",
        "생활 우선성", "스마트폰의 생활 우선성",
        "폐해 경험", "폐해", "부정적 결과",
        "신체 건강", "정신 건강", "대인관계", "대인관계 맥락", "생산성 저하",

        # 숏폼 관련 영향요인
        "숏폼 시청 조절", "숏폼 시청 조절의 어려움", "알고리즘", "추천 알고리즘", "알고리즘으로 인한 숏폼 시청 영향",

        # 생활·역량·만족도(보고서 구성)
        "여가활동", "주 여가활동", "희망하는 여가활동",
        "디지털 사용 역량", "정보 검색 역량", "정보 신뢰 판단", "사회문제 참여", "콘텐츠 제작/편집",
        "개인정보보호", "프라이버시", "학업·직업 관련 활동",
        "삶의 만족도", "전반적 만족도", "인간관계 만족도", "일/학업 만족도", "여가활동 만족도",

        # 가정·배경 요인(기존 + 확장)
        "가구원", "가구원 수", "가구", "가구주",
        "소득", "가구소득", "가구 월소득",
        "맞벌이", "한부모", "양육자", "주 양육자",
    ],

    "조사": [
        # 조사 운영/설계(보고서 표현 그대로 + 변형)
        "조사개요", "조사 개요",
        "조사방법", "자료수집", "자료 수집", "자료처리", "자료 처리",
        "가구방문 면접조사", "가구 방문", "면접조사", "면접 조사",
        "구조화된 설문지", "조사표", "가구주용 설문지", "가구원 설문지",
        "조사기간", "조사 기준시점", "2024년 9월~11월",

        # 표본설계/추정
        "표본", "표본설계", "표본 설계", "표본배분", "표본 배분", "표본추출", "표본 추출",
        "조사구", "가구명부", "가구 명부", "인구주택총조사", 
        "층화", "층별", "주택유형", "아파트", "보통조사구",
        "가중치", "가중치 산정", "모수추정", "모수 추정", "추정식",
        "신뢰수준", "표본오차", "표집오차", "반올림", "복수응답",
    ],

    # ---- 추가 토픽(필요 시) ----
    "예방·상담": [
        "예방교육", "예방 교육", "상담", "프로그램",
        "인지율", "이용경험", "경험률", "도움정도", "참여 의향",
        "스마트폰 과의존 예방 기관", "스마트폰 과의존 예방 프로그램",
        "스마트쉼센터",
    ],

    "해결방안": [
        "과의존 심각성 인식",
        "과의존 해소 방안", "대처방안", "대처 방안",
        "문제해결 주체", "문제 해결 주체",
        "개인의 해소방안", "개인의 장애요인",
        "기업의 해소방안", "정부의 해소방안", "교육시설의 해소방안",
        "디지털 디톡스", "디지털 디톡스 경험",
    ],
}

# =========================================================
# 세션 상태 초기화
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "clarification_context" not in st.session_state:
    st.session_state.clarification_context = None

# =========================================================
# LangGraph State 정의
# =========================================================
ValidationResult = Literal["PASS", "FAIL_NO_EVIDENCE", "FAIL_UNCLEAR", "FAIL_FORMAT"]

class GraphState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    session_id: str
    intent_raw: Optional[str]
    intent: Optional[str]
    is_chat_reference: Optional[bool]
    followup_type: Optional[str]
    plan: Optional[Dict[str, Any]]
    resolved_question: Optional[str]
    previous_context: Optional[str]
    rewritten_queries: Optional[List[str]]
    retrieval: Optional[Dict[str, Any]]
    context: Optional[str]
    reranked_docs: Optional[List[Document]]
    compressed_context: Optional[str]
    sanitized_context: Optional[str]
    draft_answer: Optional[str]
    safety_passed: Optional[bool]
    safety_issues: Optional[List[str]]
    validation_result: Optional[ValidationResult]
    validation_reason: Optional[str]
    validator_output: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    retry_count: Optional[int]
    retry_type: Optional[str]
    pending_clarification: Optional[str]
    clarification_context: Optional[Dict[str, Any]]
    used_default_years: Optional[bool]  # v5.1: 기본 연도 사용 플래그
    debug_info: Optional[Dict[str, Any]]

# =========================================================
# Hugging Face에서 DB 다운로드
# =========================================================
@st.cache_resource
def download_chroma_db():
    if os.path.exists(LOCAL_DB_PATH) and os.listdir(LOCAL_DB_PATH):
        return LOCAL_DB_PATH, None
    
    try:
        from huggingface_hub import snapshot_download
        downloaded_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DB_PATH,
            local_dir_use_symlinks=False
        )
        return downloaded_path, None
    except Exception as e:
        return None, str(e)

# =========================================================
# 초기화 함수
# =========================================================
@st.cache_resource
def init_resources():
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        pass
    
    os.environ['OPENAI_API_KEY'] = api_key
    
    if not os.path.exists(LOCAL_DB_PATH):
        return None, None, f"Chroma DB를 찾을 수 없습니다: {LOCAL_DB_PATH}"
    
    try:
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        vectorstore = Chroma(
            persist_directory=LOCAL_DB_PATH,
            embedding_function=embedding,
            collection_name="pdf_pages_with_summary_v2"
        )
        
        llms = {
            "router": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50),
            "casual": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=500),
            "main": ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=4000),
            "planner": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000),
            "rewrite": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500),
        }
        
        return vectorstore, llms, None
    except Exception as e:
        return None, None, str(e)

# =========================================================
# 헬퍼 함수들
# =========================================================
def is_chat_reference_question(user_input: str) -> bool:
    name_intro_patterns = [
        r"(내|제)\s*이름은?\s*[가-힣a-zA-Z]+",
        r"(저는|나는)\s*[가-힣a-zA-Z]+",
    ]
    for p in name_intro_patterns:
        if re.search(p, user_input):
            return False
    
    patterns = [
        r"(내|제)\s*이름\s*(뭐|뭔|알|기억)",
        r"(내|제)\s*이름\s*[?]",
        r"뭐라고\s*(했|물어|말)",
        r"아까", r"방금", r"이전에",
    ]
    for p in patterns:
        if re.search(p, user_input):
            return True
    return False

def parse_year_range(text: str) -> List[int]:
    years = set()
    range_patterns = [
        r"(20[2][0-4])\s*년?\s*(?:에서|부터|~|-|–)\s*(20[2][0-4])\s*년?\s*(?:까지)?",
        r"(20[2][0-4])\s*(?:~|-|–)\s*(20[2][0-4])",
    ]
    for pattern in range_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            start, end = int(m[0]), int(m[1])
            for y in range(start, end + 1):
                if y in YEAR_TO_FILENAME:
                    years.add(y)
    
    single_years = re.findall(r"\b(20[2][0-4])\s*년?\b", text)
    for y in single_years:
        yi = int(y)
        if yi in YEAR_TO_FILENAME:
            years.add(yi)
    
    return sorted(list(years))

def classify_followup_type(user_input: str, prev_context: Dict[str, Any]) -> str:
    if not prev_context.get("last_topic"):
        return "none"
    
    has_new_topic_keyword = False
    for keywords in TOPIC_KEYWORDS.values():
        for kw in keywords:
            if kw in user_input and kw not in str(prev_context.get("last_topic_core", "")):
                has_new_topic_keyword = True
                break
    
    if len(user_input) >= 30 and has_new_topic_keyword:
        return "none"
    
    target_patterns = [
        r"^(청소년|유아동|성인|60대|대학생|중학생|고등학생|초등학생|남성|여성)[은의]?\s*[?]?$",
        r"^(청소년|유아동|성인|60대)[은의]?\s*(어때|어떻게|어떤가|결과|기준|경우)",
    ]
    for p in target_patterns:
        if re.search(p, user_input):
            return "target_change"
    
    if len(user_input) <= 20:
        for keywords in TARGET_KEYWORDS.values():
            for kw in keywords:
                if kw in user_input:
                    return "target_change"
    
    year_patterns = [
        r"^(20[2][0-4])년?\s*[은의]?\s*[?]?$",
        r"^(20[2][0-4])년?\s*(어때|어떻게|결과|기준)",
    ]
    for p in year_patterns:
        if re.search(p, user_input):
            return "year_change"
    
    if len(user_input) <= 15:
        years = parse_year_range(user_input)
        if years:
            return "year_change"
    
    detail_patterns = [
        r"(더|좀)\s*(자세히|구체적|상세)",
        r"(왜|원인|이유).*[?]",
    ]
    for p in detail_patterns:
        if re.search(p, user_input):
            return "detail_request"
    
    if len(user_input) <= 15 and re.search(r"[?]$", user_input):
        return "detail_request"
    
    return "none"

def extract_previous_context(chat_history: List[BaseMessage]) -> Dict[str, Any]:
    context = {
        "user_name": None,
        "last_topic": None,
        "last_topic_core": None,
        "last_target": None,
        "last_years": [],
    }
    
    if not chat_history:
        return context
    
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            name_match = re.search(r"(?:내\s*이름은?|저는?|나는?)\s*([가-힣a-zA-Z]+)", msg.content)
            if name_match:
                context["user_name"] = name_match.group(1)
    
    human_msgs = [m for m in chat_history if isinstance(m, HumanMessage)][-2:]
    
    for msg in reversed(human_msgs):
        content = msg.content
        
        if not context["last_topic"]:
            context["last_topic"] = content[:300]
        
        years = parse_year_range(content)
        if years and not context["last_years"]:
            context["last_years"] = years
        
        if not context["last_target"]:
            for keywords in TARGET_KEYWORDS.values():
                for kw in keywords:
                    if kw in content:
                        context["last_target"] = kw
                        break
                if context["last_target"]:
                    break
        
        if not context["last_topic_core"]:
            topic_parts = []
            for keywords in TOPIC_KEYWORDS.values():
                for kw in keywords:
                    if kw in content:
                        topic_parts.append(kw)
            if topic_parts:
                context["last_topic_core"] = " ".join(topic_parts[:3])
    
    return context

def _keyword_boost_score(doc: Document, query: str) -> float:
    text = (doc.page_content or "").lower()
    query_terms = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
    boost = 0.0
    for term in query_terms:
        if len(term) >= 2 and term in text:
            boost += 0.02
    return min(boost, 0.15)

# =========================================================
# 테이블 파싱 및 렌더링
# =========================================================
def parse_markdown_table(text: str) -> List[Dict[str, Any]]:
    tables = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('|') and line.endswith('|'):
            table_lines = []
            start_idx = i
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('|') and line.endswith('|'):
                    table_lines.append(line)
                    i += 1
                elif line.startswith('|---') or line.startswith('| ---'):
                    i += 1
                    continue
                else:
                    break
            
            if len(table_lines) >= 2:
                header_line = table_lines[0]
                headers = [h.strip() for h in header_line.split('|')[1:-1]]
                data_rows = []
                for row_line in table_lines[1:]:
                    if '---' in row_line:
                        continue
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    if len(cells) == len(headers):
                        data_rows.append(cells)
                
                if headers and data_rows:
                    tables.append({
                        'headers': headers,
                        'rows': data_rows,
                        'start_idx': start_idx,
                        'end_idx': i
                    })
        else:
            i += 1
    return tables

def render_answer_with_tables(answer: str) -> None:
    tables = parse_markdown_table(answer)
    if not tables:
        st.markdown(answer)
        return
    
    lines = answer.split('\n')
    current_pos = 0
    
    for table in tables:
        before_text = '\n'.join(lines[current_pos:table['start_idx']])
        if before_text.strip():
            st.markdown(before_text)
        
        try:
            df = pd.DataFrame(table['rows'], columns=table['headers'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except:
            st.markdown("| " + " | ".join(table['headers']) + " |")
            for row in table['rows']:
                st.markdown("| " + " | ".join(row) + " |")
        
        current_pos = table['end_idx']
    
    after_text = '\n'.join(lines[current_pos:])
    if after_text.strip():
        st.markdown(after_text)

# =========================================================
# 프롬프트 정의
# =========================================================
def get_router_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "사용자 질문을 분류하는 라우터입니다.\n"
         "분류: SMALLTALK / RAG / CHAT_REF / OFFTOPIC\n"
         "출력: 분류명만"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_smalltalk_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         f"스마트폰 과의존 실태조사 보고서 분석 시스템입니다.\n{BOT_IDENTITY}\n"
         "인사에는 간결하게 응대하고 예시 질문을 제안하세요."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_offtopic_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "스마트폰 과의존 실태조사 보고서 분석 시스템입니다.\n"
         "해당 질문은 전문 분야가 아닙니다. 정중하게 안내하세요."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_planner_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "검색 계획 수립기입니다. JSON만 출력하세요.\n"
         "허용 파일명:\n" +
         "\n".join([f"- {y}년: {fn}" for y, fn in YEAR_TO_FILENAME.items()]) +
         "\n\nJSON: {{\"resolved_question\": \"...\", \"years\": [...], "
         "\"file_name_filters\": [...], \"queries\": [...]}}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",
         "질문: {input}\n후속질문 유형: {followup_type}\n"
         "이전 주제: {topic_core}\n이전 대상: {last_target}\n이전 연도: {last_years}\n\nJSON:")
    ])

def get_rewrite_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "검색 쿼리 최적화 전문가입니다.\n"
         "불필요한 조사/어미 제거, 핵심 키워드 추출, 동의어 확장.\n"
         "JSON: {{\"optimized_queries\": [\"쿼리1\", \"쿼리2\", ...]}}"
        ),
        ("human",
         "원본 질문: {resolved_question}\n원본 쿼리: {queries}\n연도: {years}\n\nJSON:")
    ])

def get_answer_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "스마트폰 과의존 실태조사 보고서 분석 시스템입니다.\n\n"
         "원칙:\n"
         "1. CONTEXT에서 수치 인용 필수\n"
         "2. 출처(파일명 p.페이지) 필수\n"
         "3. 변화량(%p) 명시\n"
         "4. CONTEXT에 없으면 '검색 결과에 포함되지 않았습니다' 명시"
        ),
        ("human",
         "[질문]\n{input}\n\n[CONTEXT]\n{context}\n\n답변:")
    ])

def get_answer_retry_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "스마트폰 과의존 실태조사 보고서 분석 시스템입니다.\n\n"
         "⚠️ 이전 문제: {previous_issue}\n\n"
         "수정 지침:\n"
         "1. 모든 수치에 출처 형식: (파일명.pdf p.00)\n"
         "2. CONTEXT에서 직접 인용만\n"
         "3. 없는 정보는 '포함되지 않았습니다' 명시"
        ),
        ("human",
         "[질문]\n{input}\n\n[CONTEXT]\n{context}\n\n수정된 답변:")
    ])

def get_validator_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "답변 품질 검수기입니다.\n\n"
         "분류:\n"
         "- PASS: 양호\n"
         "- FAIL_NO_EVIDENCE: 근거 부족 (검색 재시도 필요)\n"
         "- FAIL_UNCLEAR: 질문 불명확 (명확화 필요)\n"
         "- FAIL_FORMAT: 형식 문제 (재작성 필요)\n\n"
         "JSON: {{\"result\": \"PASS|FAIL_...\", \"reason\": \"...\", "
         "\"clarify_question\": \"...\", \"corrected_answer\": \"...\"}}"
        ),
        ("human",
         "[질문]\n{input}\n\n[CONTEXT]\n{context}\n\n[답변]\n{answer}\n\nJSON:")
    ])

# =========================================================
# 노드 함수들 생성
# =========================================================
def create_node_functions(vectorstore, llms, status_placeholder):
    
    def update_status(message: str, retry_info: str = ""):
        retry_badge = f'<span class="retry-badge">{retry_info}</span>' if retry_info else ""
        status_placeholder.markdown(f"""
        <div class="status-box">🔄 {message} {retry_badge}</div>
        """, unsafe_allow_html=True)
    
    # ----- 노드 1: 라우터 -----
    def route_intent(state: GraphState) -> GraphState:
        update_status("질문 분석 중...")
        try:
            user_input = state["input"]
            chat_history = state.get("chat_history", [])
            
            state["retry_count"] = state.get("retry_count") or 0
            
            # Clarification 응답 처리
            if state.get("clarification_context"):
                state["intent"] = "RAG"
                state["clarification_context"] = None
                return state
            
            if is_chat_reference_question(user_input):
                state["intent"] = "CHAT_REF"
                state["followup_type"] = "none"
                return state
            
            prev_ctx = extract_previous_context(chat_history)
            followup_type = classify_followup_type(user_input, prev_ctx)
            state["followup_type"] = followup_type
            
            rag_keywords = [
                # 핵심 주제/용어
                "스마트폰", "과의존", "스마트폰 과의존", "과다이용", "과의존위험군", "고위험군", "잠재적위험군", "일반사용자군",
                "조절실패", "현저성", "문제적 결과",
                "조사", "실태조사", "조사개요", "조사방법", "조사대상", "모집단", "표본", "표본설계", "표본추출", "가중치", "모수추정",
                
                # 지표/표현
                "과의존률", "비율", "률", "%", "%p", "단위", "점수", "총점", "평균", "4점만점", "기준점수", "역문항",
                
                # 대상/분류
                "유아동", "영유아", "아동", "청소년", "성인", "60대", "고령층",
                "초등학생", "중학생", "고등학생", "대학생",
                "성별", "남성", "여성", "지역", "도시규모", "대도시", "중소도시", "읍면지역",
                
                # 이용/조절/심층문항
                "이용시간", "이용시간 조절", "조절 어려움", "본인 의지대로 조절",
                "사용조절", "생활 우선성", "폐해 경험", "신체 건강", "정신 건강", "대인관계", "생산성 저하",
                
                # 콘텐츠/플랫폼
                "콘텐츠", "이용률", "이용정도", "생활에 도움이 되는 콘텐츠", "부작용 우려 콘텐츠", "최근 1년간 이용량 증가",
                "온라인 동영상 서비스", "OVS", "숏폼", "알고리즘", "추천 알고리즘",
                "유튜브 쇼츠", "인스타그램 릴스", "틱톡", "네이버 클립", "페이스북 릴스",
                "SNS", "메신저", "게임", "영화/TV/동영상", "뉴스보기", "쇼핑", "투자 및 자산관리", "성인용 콘텐츠", "사행성 게임",
                "생성형 AI서비스",

                # 예방/상담/해결
                "예방교육", "상담", "프로그램", "인지율", "경험률", "도움정도", "참여 의향",
                "대처방안", "해소 방안", "문제해결 주체", "기업", "정부", "교육시설", "디지털 디톡스",
                # 문서 내 참조 토큰(질의에 “표/그림 번호”로 들어오는 경우 대비)
                "표", "그림", "문항", "요인별 속성", "문항별 속성",
            ]

            
            if re.search(r"\b(20[2][0-4])\s*년?\b", user_input):
                state["intent"] = "RAG"
                return state
            
            if any(kw in user_input for kw in rag_keywords):
                state["intent"] = "RAG"
                return state
            
            if followup_type != "none":
                state["intent"] = "RAG"
                return state
            
            result = (get_router_prompt() | llms["router"] | StrOutputParser()).invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            state["intent_raw"] = result.strip().upper()
            
            if state["intent_raw"] in ("SMALLTALK", "RAG", "OFFTOPIC", "CHAT_REF"):
                state["intent"] = state["intent_raw"]
            else:
                state["intent"] = "RAG"
            
            return state
        except Exception as e:
            state["intent"] = "RAG"
            state["followup_type"] = "none"
            return state
    
    # ----- 노드 2a: SMALLTALK -----
    def handle_smalltalk(state: GraphState) -> GraphState:
        update_status("응답 생성 중...")
        try:
            answer = (get_smalltalk_prompt() | llms["casual"] | StrOutputParser()).invoke({
                "input": state["input"],
                "chat_history": state.get("chat_history", [])
            })
            state["final_answer"] = answer
            return state
        except Exception as e:
            state["final_answer"] = f"오류: {e}"
            return state
    
    # ----- 노드 2b: OFFTOPIC -----
    def handle_offtopic(state: GraphState) -> GraphState:
        update_status("응답 생성 중...")
        try:
            answer = (get_offtopic_prompt() | llms["casual"] | StrOutputParser()).invoke({
                "input": state["input"],
                "chat_history": state.get("chat_history", [])
            })
            state["final_answer"] = answer
            return state
        except Exception as e:
            state["final_answer"] = f"오류: {e}"
            return state
    
    # ----- 노드 2c: CHAT_REF -----
    def handle_chat_reference(state: GraphState) -> GraphState:
        update_status("대화 기록 확인 중...")
        try:
            chat_history = state.get("chat_history", [])
            user_input = state["input"]
            prev_ctx = extract_previous_context(chat_history)
            
            if re.search(r"(내|제)\s*이름", user_input):
                if prev_ctx["user_name"]:
                    state["final_answer"] = f"{prev_ctx['user_name']}님으로 말씀하셨습니다."
                else:
                    state["final_answer"] = "아직 이름을 말씀해주시지 않았습니다."
                return state
            
            state["final_answer"] = "이전 대화 참조가 명확하지 않습니다."
            return state
        except Exception as e:
            state["final_answer"] = f"오류: {e}"
            return state
    
    # ----- 노드 3: 플래너 -----
    def plan_search(state: GraphState) -> GraphState:
        update_status("검색 계획 수립 중...")
        try:
            user_input = state["input"]
            chat_history = state.get("chat_history", [])
            followup_type = state.get("followup_type", "none")
            
            prev_ctx = extract_previous_context(chat_history)
            
            if followup_type == "none":
                topic_core, last_target, last_years = "", "", []
            else:
                topic_core = prev_ctx.get("last_topic_core", "") or ""
                last_target = prev_ctx.get("last_target", "") or ""
                last_years = prev_ctx.get("last_years", [])
            
            result = (get_planner_prompt() | llms["planner"] | StrOutputParser()).invoke({
                "input": user_input,
                "chat_history": chat_history[-4:],
                "followup_type": followup_type,
                "topic_core": topic_core,
                "last_target": last_target,
                "last_years": str(last_years),
            })
            
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group()
            
            plan = json.loads(result)
            
            years = plan.get('years', [])
            input_years = parse_year_range(user_input)
            years = sorted(list(set([y for y in (years + input_years) if y in YEAR_TO_FILENAME])))
            
            # ✅ v5.1: 연도 미지정 시 기본값 (최근 2년) 설정
            used_default_years = False
            if not years:
                years = [2023, 2024]
                used_default_years = True
            
            state["used_default_years"] = used_default_years
            
            fns = [fn for fn in plan.get("file_name_filters", []) if fn in ALLOWED_FILES]
            if years and not fns:
                fns = [YEAR_TO_FILENAME[y] for y in years]
            
            queries = [str(q).strip() for q in plan.get('queries', []) if str(q).strip()]
            resolved_q = plan.get("resolved_question", user_input) or user_input
            
            while len(queries) < 3:
                queries.append(resolved_q)
            
            state["plan"] = {
                "years": years,
                "file_name_filters": fns,
                "queries": queries[:3],
                "resolved_question": resolved_q,
                "used_default_years": used_default_years,
            }
            state["resolved_question"] = resolved_q
            return state
            
        except Exception as e:
            years = parse_year_range(state["input"])
            
            # ✅ 폴백에서도 기본 연도 적용
            used_default_years = False
            if not years:
                years = [2023, 2024]
                used_default_years = True
            
            fns = [YEAR_TO_FILENAME[y] for y in years if y in YEAR_TO_FILENAME]
            state["plan"] = {
                "years": years,
                "file_name_filters": fns,
                "queries": [state["input"]] * 3,
                "resolved_question": state["input"],
                "used_default_years": used_default_years,
            }
            state["resolved_question"] = state["input"]
            state["used_default_years"] = used_default_years
            return state
    
    # ----- 노드 4: Query Rewrite -----
    def query_rewrite(state: GraphState) -> GraphState:
        update_status("쿼리 최적화 중...")
        try:
            plan = state["plan"]
            queries = plan.get("queries", [])
            resolved_q = plan.get("resolved_question", "")
            years = plan.get("years", [])
            
            # 멀티연도 쿼리 추가
            if len(years) > 1:
                base_query_clean = re.sub(r'20[2][0-4]년?', '', resolved_q).strip()
                for y in years:
                    year_query = f"{y}년 {base_query_clean}"
                    if year_query not in queries:
                        queries.append(year_query)
            
            result = (get_rewrite_prompt() | llms["rewrite"] | StrOutputParser()).invoke({
                "resolved_question": resolved_q,
                "queries": str(queries),
                "years": str(years),
            })
            
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group()
            
            optimized = json.loads(result)
            rewritten = optimized.get("optimized_queries", queries)
            
            if not isinstance(rewritten, list) or not rewritten:
                rewritten = queries
            
            # 중복 제거
            unique_queries = list(dict.fromkeys(rewritten))
            
            state["rewritten_queries"] = unique_queries[:6]
            state["plan"]["queries"] = unique_queries[:6]
            return state
            
        except Exception as e:
            state["rewritten_queries"] = state["plan"].get("queries", [])
            return state
    
    # ----- 노드 5: 검색 -----
    def retrieve_documents(state: GraphState) -> GraphState:
        retry_count = state.get("retry_count", 0)
        retry_info = f"재시도 #{retry_count}" if retry_count > 0 else ""
        update_status("보고서 검색 중...", retry_info)
        
        try:
            plan = state["plan"]
            target_files = plan.get("file_name_filters", [])
            queries = state.get("rewritten_queries") or plan.get("queries", [])
            resolved_q = plan.get("resolved_question", "")
            
            # 재시도 시 파라미터 증가
            if retry_count > 0 and state.get("retry_type") == "retrieve":
                k_per_query = RETRY_K_PER_QUERY
                top_parents = RETRY_TOP_PARENTS
                top_parents_per_file = RETRY_TOP_PARENTS_PER_FILE
            else:
                k_per_query = DEFAULT_K_PER_QUERY
                top_parents = DEFAULT_TOP_PARENTS
                top_parents_per_file = DEFAULT_TOP_PARENTS_PER_FILE
            
            all_docs = []
            files_searched = []
            
            if target_files:
                for fn in target_files:
                    file_filter = {'$and': [
                        {'doc_type': {"$in": SUMMARY_TYPES}},
                        {'file_name': fn}
                    ]}
                    
                    file_docs = []
                    seen_keys = set()
                    
                    for q in queries:
                        if not q:
                            continue
                        try:
                            hits = vectorstore.similarity_search_with_relevance_scores(
                                q, k=k_per_query, filter=file_filter
                            )
                            for doc, score in hits:
                                key = f"{doc.metadata.get('parent_id')}|{doc.metadata.get('page')}"
                                if key not in seen_keys:
                                    doc.metadata["_score"] = float(score)
                                    doc.metadata["_source_file"] = fn
                                    file_docs.append(doc)
                                    seen_keys.add(key)
                        except:
                            pass
                    
                    for doc in file_docs:
                        boost = _keyword_boost_score(doc, resolved_q)
                        doc.metadata["_final_score"] = doc.metadata.get("_score", 0) + boost
                    
                    file_docs.sort(key=lambda d: d.metadata.get("_final_score", 0), reverse=True)
                    all_docs.extend(file_docs[:top_parents_per_file * 2])
                    
                    if file_docs:
                        files_searched.append(fn)
            else:
                base_filter = {'doc_type': {"$in": SUMMARY_TYPES}}
                seen_keys = set()
                
                for q in queries:
                    if not q:
                        continue
                    hits = vectorstore.similarity_search_with_relevance_scores(
                        q, k=k_per_query, filter=base_filter
                    )
                    for doc, score in hits:
                        key = f"{doc.metadata.get('parent_id')}|{doc.metadata.get('page')}"
                        if key not in seen_keys:
                            doc.metadata["_score"] = float(score)
                            all_docs.append(doc)
                            seen_keys.add(key)
                
                for doc in all_docs:
                    boost = _keyword_boost_score(doc, resolved_q)
                    doc.metadata["_final_score"] = doc.metadata.get("_score", 0) + boost
                
                files_searched = ["전체"]
            
            all_docs.sort(key=lambda d: d.metadata.get("_final_score", 0), reverse=True)
            
            # Parent ID 선정
            parent_ids = []
            seen_pid = set()
            
            if target_files:
                for fn in target_files:
                    for doc in all_docs:
                        if doc.metadata.get("_source_file") == fn or doc.metadata.get("file_name") == fn:
                            pid = doc.metadata.get("parent_id")
                            if pid and pid not in seen_pid:
                                parent_ids.append(pid)
                                seen_pid.add(pid)
                                break
            
            for doc in all_docs:
                if len(parent_ids) >= top_parents:
                    break
                pid = doc.metadata.get("parent_id")
                if pid and pid not in seen_pid:
                    parent_ids.append(pid)
                    seen_pid.add(pid)
            
            # Chunk 확장
            expanded_chunks = []
            for pid in parent_ids:
                try:
                    got = vectorstore._collection.get(
                        where={'parent_id': pid},
                        include=['documents', 'metadatas']
                    )
                    chunks = []
                    for txt, meta in zip(got.get("documents", []), got.get("metadatas", [])):
                        if isinstance(meta, dict) and meta.get("doc_type") == "text_chunk":
                            chunks.append((int(meta.get("chunk_index", 0)), txt or "", meta))
                    
                    chunks.sort(key=lambda x: x[0])
                    for _, txt, meta in chunks[:MAX_CHUNKS_PER_PARENT]:
                        expanded_chunks.append(Document(page_content=txt, metadata=meta))
                except:
                    pass
            
            pid_set = set(parent_ids)
            kept_summaries = [d for d in all_docs if d.metadata.get("parent_id") in pid_set]
            final_docs = kept_summaries + expanded_chunks
            
            blocks = []
            for i, d in enumerate(final_docs, start=1):
                m = d.metadata
                text = d.page_content[:MAX_CHARS_PER_DOC]
                blocks.append(f"[{i}] {m.get('file_name', 'unknown')} (p.{m.get('page', '?')})\n{text}")
            
            state["retrieval"] = {
                "docs": final_docs,
                "parent_ids": parent_ids,
                "files_searched": files_searched,
                "doc_count": len(final_docs),
            }
            state["context"] = "\n\n---\n\n".join(blocks)
            return state
            
        except Exception as e:
            state["context"] = ""
            state["retrieval"] = {"docs": [], "parent_ids": [], "files_searched": [], "doc_count": 0}
            return state
    
    # ----- 노드 6: Rerank & Compress -----
    def rerank_compress(state: GraphState) -> GraphState:
        update_status("결과 정렬 및 압축 중...")
        try:
            docs = state["retrieval"].get("docs", [])
            query = state.get("resolved_question", "")
            
            if not docs:
                state["reranked_docs"] = []
                state["compressed_context"] = ""
                return state
            
            query_keywords = set(re.findall(r'[가-힣]+', query))
            
            for doc in docs:
                content_keywords = set(re.findall(r'[가-힣]+', doc.page_content or ""))
                overlap = len(query_keywords & content_keywords)
                doc.metadata["_rerank_score"] = doc.metadata.get("_final_score", 0) + (overlap * 0.01)
            
            docs.sort(key=lambda d: d.metadata.get("_rerank_score", 0), reverse=True)
            
            # 중복 제거
            seen_content = set()
            unique_docs = []
            for doc in docs:
                content_hash = hash(doc.page_content[:500])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            compressed_docs = unique_docs[:20]
            
            blocks = []
            for i, d in enumerate(compressed_docs, start=1):
                m = d.metadata
                text = d.page_content[:MAX_CHARS_PER_DOC]
                blocks.append(f"[{i}] {m.get('file_name', 'unknown')} (p.{m.get('page', '?')})\n{text}")
            
            state["reranked_docs"] = compressed_docs
            state["compressed_context"] = "\n\n---\n\n".join(blocks)
            return state
            
        except Exception as e:
            state["reranked_docs"] = state["retrieval"].get("docs", [])
            state["compressed_context"] = state.get("context", "")
            return state
    
    # ----- 노드 7: Context Sanitize -----
    def context_sanitize(state: GraphState) -> GraphState:
        update_status("컨텍스트 검증 중...")
        try:
            context = state.get("compressed_context") or state.get("context", "")
            
            danger_patterns = [
                r"(?i)ignore\s+(previous|above|all)\s+instructions?",
                r"(?i)you\s+are\s+now\s+",
                r"(?i)act\s+as\s+",
                r"(?i)system\s*:\s*",
            ]
            
            sanitized = context
            for pattern in danger_patterns:
                sanitized = re.sub(pattern, "[FILTERED]", sanitized)
            
            state["sanitized_context"] = sanitized
            return state
            
        except Exception as e:
            state["sanitized_context"] = state.get("compressed_context") or state.get("context", "")
            return state
    
    # ----- 노드 8: 답변 생성 -----
    def generate_answer(state: GraphState) -> GraphState:
        retry_count = state.get("retry_count", 0)
        retry_info = f"재생성 #{retry_count}" if retry_count > 0 and state.get("retry_type") == "generate" else ""
        update_status("답변 생성 중...", retry_info)
        
        try:
            context = state.get("sanitized_context") or state.get("compressed_context") or state.get("context", "")
            
            if not context.strip():
                state["draft_answer"] = "검색 결과를 찾지 못했습니다. 질문을 다시 구체적으로 말씀해주시겠습니까?"
                return state
            
            if retry_count > 0 and state.get("retry_type") == "generate":
                previous_issue = state.get("validation_reason", "형식 문제")
                answer = (get_answer_retry_prompt() | llms["main"] | StrOutputParser()).invoke({
                    "input": state["resolved_question"] or state["input"],
                    "context": context,
                    "previous_issue": previous_issue,
                })
            else:
                answer = (get_answer_prompt() | llms["main"] | StrOutputParser()).invoke({
                    "input": state["resolved_question"] or state["input"],
                    "context": context
                })
            
            state["draft_answer"] = answer
            return state
            
        except Exception as e:
            state["draft_answer"] = f"답변 생성 중 오류: {e}"
            return state
    
    # ----- 노드 9: Safety Check -----
    def safety_check(state: GraphState) -> GraphState:
        update_status("안전성 검사 중...")
        try:
            answer = state.get("draft_answer", "")
            issues = []
            
            sensitive_patterns = [
                (r"(?i)(자살|자해)", "자해 관련 내용"),
                (r"(?i)(폭력|학대)", "폭력 관련 내용"),
            ]
            
            for pattern, issue_name in sensitive_patterns:
                if re.search(pattern, answer):
                    issues.append(issue_name)
            
            state["safety_passed"] = len(issues) == 0
            state["safety_issues"] = issues
            return state
            
        except Exception as e:
            state["safety_passed"] = True
            state["safety_issues"] = []
            return state
    
    # ----- 노드 10: Validate -----
    def validate_answer(state: GraphState) -> GraphState:
        update_status("답변 검증 중...")
        try:
            retry_count = state.get("retry_count", 0)
            
            if retry_count >= MAX_RETRY_COUNT:
                state["validation_result"] = "PASS"
                final_answer = state["draft_answer"]
                
                # ✅ v5.1: 기본 연도 사용 시 확인 메시지 추가
                if state.get("used_default_years"):
                    final_answer = _append_year_confirmation(final_answer, state)
                
                state["final_answer"] = final_answer
                return state
            
            context = state.get("sanitized_context") or state.get("context", "")
            
            result = (get_validator_prompt() | llms["main"] | StrOutputParser()).invoke({
                "input": state["resolved_question"] or state["input"],
                "context": context[:15000],
                "answer": state["draft_answer"]
            })
            
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group()
            
            validator_out = json.loads(result)
            state["validator_output"] = validator_out
            
            validation_result = validator_out.get("result", "PASS").upper()
            valid_results = ["PASS", "FAIL_NO_EVIDENCE", "FAIL_UNCLEAR", "FAIL_FORMAT"]
            if validation_result not in valid_results:
                validation_result = "PASS"
            
            state["validation_result"] = validation_result
            state["validation_reason"] = validator_out.get("reason", "")
            
            if validation_result == "PASS":
                corrected = validator_out.get("corrected_answer", "")
                final_answer = corrected if corrected and len(corrected) > 50 else state["draft_answer"]
                
                # ✅ v5.1: 기본 연도 사용 시 확인 메시지 추가
                if state.get("used_default_years"):
                    final_answer = _append_year_confirmation(final_answer, state)
                
                state["final_answer"] = final_answer
            elif validation_result == "FAIL_UNCLEAR":
                clarify_q = validator_out.get("clarify_question", "")
                if clarify_q:
                    state["pending_clarification"] = clarify_q
            
            return state
            
        except Exception as e:
            state["validation_result"] = "PASS"
            final_answer = state["draft_answer"]
            
            # ✅ v5.1: 기본 연도 사용 시 확인 메시지 추가
            if state.get("used_default_years"):
                final_answer = _append_year_confirmation(final_answer, state)
            
            state["final_answer"] = final_answer
            return state
    
    # ✅ v5.1: 기본 연도 확인 메시지 헬퍼 함수
    def _append_year_confirmation(answer: str, state: GraphState) -> str:
        years = state.get("plan", {}).get("years", [2023, 2024])
        year_str = ", ".join([f"{y}년" for y in years])
        
        confirmation_msg = (
            f"\n\n---\n"
            f"📌 **연도 확인 요청**: 질문에 특정 연도가 명시되지 않아 "
            f"**최근 데이터({year_str})**를 기준으로 답변드렸습니다. "
            f"다른 연도(2020~2024년)의 정보가 필요하시면 말씀해 주세요."
        )
        
        return answer + confirmation_msg
    
    # ----- 노드 11: Clarify -----
    def handle_clarify(state: GraphState) -> GraphState:
        update_status("명확화 질문 생성 중...")
        try:
            clarify_question = state.get("pending_clarification", "")
            if not clarify_question:
                clarify_question = "질문을 좀 더 구체적으로 말씀해 주시겠습니까? 예를 들어, 특정 연도나 대상(청소년, 성인 등)을 지정해 주시면 더 정확한 답변이 가능합니다."
            
            state["clarification_context"] = {
                "original_query": state["input"],
                "partial_plan": state.get("plan"),
            }
            state["final_answer"] = clarify_question
            return state
            
        except Exception as e:
            state["final_answer"] = "질문을 좀 더 구체적으로 말씀해 주시겠습니까?"
            return state
    
    # ----- 노드 12: Retrieve Retry -----
    def retrieve_retry(state: GraphState) -> GraphState:
        state["retry_count"] = (state.get("retry_count") or 0) + 1
        state["retry_type"] = "retrieve"
        
        queries = state["plan"].get("queries", [])
        resolved_q = state.get("resolved_question", "")
        
        synonyms = {
    # 핵심 지표/집단
    "과의존률": ["과의존 위험군 비율", "과의존위험군 비율", "스마트폰 과의존위험군 비율", "스마트폰 과의존"],
    "과의존위험군": ["과의존 위험군", "스마트폰 과의존위험군", "스마트폰 과의존 위험군", "고위험군+잠재적위험군"],
    "잠재적위험군": ["잠재적 위험군", "잠재 위험군"],
    "일반사용자군": ["일반 사용자군", "일반군"],

    # 대상/연령
    "유아동": ["영유아", "유아", "아동", "어린이", "만 3~9세", "만3~9세", "만 3∼9세"],
    "청소년": ["10대", "십대", "만 10~19세", "만10~19세", "만 10∼19세"],
    "성인": ["만 20~59세", "만20~59세", "20대", "30대", "40대", "50대"],
    "60대": ["고령층", "고령자", "만 60~69세", "만60~69세", "만 60∼69세"],

        }
        
        expanded_queries = list(queries)
        for original, alternatives in synonyms.items():
            if original in resolved_q:
                for alt in alternatives:
                    new_query = resolved_q.replace(original, alt)
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)
        
        state["plan"]["queries"] = expanded_queries[:8]
        state["rewritten_queries"] = expanded_queries[:8]
        return state
    
    # ----- 노드 13: Generate Retry -----
    def generate_retry(state: GraphState) -> GraphState:
        state["retry_count"] = (state.get("retry_count") or 0) + 1
        state["retry_type"] = "generate"
        return state
    
    return {
        "route_intent": route_intent,
        "smalltalk": handle_smalltalk,
        "offtopic": handle_offtopic,
        "chat_ref": handle_chat_reference,
        "plan_search": plan_search,
        "query_rewrite": query_rewrite,
        "retrieve": retrieve_documents,
        "rerank_compress": rerank_compress,
        "context_sanitize": context_sanitize,
        "generate": generate_answer,
        "safety_check": safety_check,
        "validate": validate_answer,
        "clarify": handle_clarify,
        "retrieve_retry": retrieve_retry,
        "generate_retry": generate_retry,
    }

# =========================================================
# 그래프 빌더
# =========================================================
def build_graph(node_functions):
    workflow = StateGraph(GraphState)
    
    for name, func in node_functions.items():
        workflow.add_node(name, func)
    
    def route_by_intent(state: GraphState) -> str:
        intent = state.get("intent", "RAG")
        if intent == "SMALLTALK":
            return "smalltalk"
        elif intent == "OFFTOPIC":
            return "offtopic"
        elif intent == "CHAT_REF":
            return "chat_ref"
        else:
            return "rag_pipeline"
    
    def route_after_validate(state: GraphState) -> str:
        retry_count = state.get("retry_count", 0)
        if retry_count >= MAX_RETRY_COUNT:
            return "end"
        
        result = state.get("validation_result", "PASS")
        if result == "PASS":
            return "end"
        elif result == "FAIL_NO_EVIDENCE":
            return "retrieve_retry"
        elif result == "FAIL_UNCLEAR":
            return "clarify"
        elif result == "FAIL_FORMAT":
            return "generate_retry"
        return "end"
    
    workflow.set_entry_point("route_intent")
    
    workflow.add_conditional_edges(
        "route_intent",
        route_by_intent,
        {
            "smalltalk": "smalltalk",
            "offtopic": "offtopic",
            "chat_ref": "chat_ref",
            "rag_pipeline": "plan_search"
        }
    )
    
    workflow.add_edge("smalltalk", END)
    workflow.add_edge("offtopic", END)
    workflow.add_edge("chat_ref", END)
    workflow.add_edge("clarify", END)
    
    workflow.add_edge("plan_search", "query_rewrite")
    workflow.add_edge("query_rewrite", "retrieve")
    workflow.add_edge("retrieve", "rerank_compress")
    workflow.add_edge("rerank_compress", "context_sanitize")
    workflow.add_edge("context_sanitize", "generate")
    workflow.add_edge("generate", "safety_check")
    workflow.add_edge("safety_check", "validate")
    
    workflow.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "end": END,
            "retrieve_retry": "retrieve_retry",
            "clarify": "clarify",
            "generate_retry": "generate_retry"
        }
    )
    
    workflow.add_edge("retrieve_retry", "retrieve")
    workflow.add_edge("generate_retry", "generate")
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# =========================================================
# 메인 UI
# =========================================================
def main():
    st.title("📊 스마트폰 과의존 실태조사 분석 시스템 v5")
    
    # 사이드바
    with st.sidebar:
        st.header("📋 시스템 정보")
        st.markdown(BOT_IDENTITY)
        
        st.divider()
        
        st.subheader("🔧 v5 새 기능")
        st.caption("✅ 회복 루프 (검색/생성 재시도)")
        st.caption("✅ Query Rewrite (쿼리 최적화)")
        st.caption("✅ Rerank & Compress")
        st.caption("✅ Safety Guard")
        
        st.divider()
        
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.clarification_context = None
            st.rerun()
        
        st.divider()
        
        debug_mode = st.checkbox("🔧 디버그 모드", value=False)
    
    # 사용자 가이드 박스
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">📌 사용 안내</div>
        <div class="guide-item">
            <strong>ℹ️ 용도:</strong> 스마트폰 과의존 실태조사 보고서(2020~2024) <strong>단순 정보 검색용</strong>입니다. <br>
            인사이트 제공, 일반 대화, 보고서 외 정보 검색에는 적합하지 않습니다.
        </div>
        <div class="guide-item">
            <strong>💡 검색 팁:</strong> 질문은 <strong>최대한 구체적으로</strong> 작성해 주세요.<br>
            과도한 검색결과 방지를 위한 설정으로 인해 일부 연도가 검색 결과에서 누락될 수 있습니다. 그럴 때는 해당 연도를 지정해서 다시 질문해주세요.<br>
            보고서 내 유사한 내용이 다수 있어, 검색 성능이 안나올 수 있습니다. 요구하고자하는 바를 확실히 설명해주세요<br>
            예) "과의존률" → "2024년 청소년 스마트폰 과의존 위험군 비율"
            예) "숏폼과 과의존" → "숏폼 이용률에 따른 과의존 차이" or "과의존위험군별 숏폼 이용 특성의 차이"
        </div>
        <div class="guide-item">
            <strong>⚠️ 주의:</strong> AI 답변에 <strong>오류(할루시네이션)</strong>가 있을 수 있습니다. <br>
            검색 결과를 바로 인용하지 마시고, <strong>원문을 통해 한번 더 확인한 뒤</strong> 정보를 최종적으로 사용하십시요.
            <a href="https://www.nia.or.kr" target="_blank" style="color: #fff;">NIA 홈페이지</a>에서 원문 확인 권장.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # DB 다운로드
    if not os.path.exists(LOCAL_DB_PATH) or not os.listdir(LOCAL_DB_PATH):
        st.info("🔄 Chroma DB를 다운로드하고 있습니다...")
        with st.spinner(f"Hugging Face에서 다운로드 중... ({HF_REPO_ID})"):
            db_path, error = download_chroma_db()
        
        if error:
            st.error(f"DB 다운로드 실패: {error}")
            return
        else:
            st.success("DB 다운로드 완료!")
            st.rerun()
    
    # 리소스 초기화
    vectorstore, llms, error = init_resources()
    
    if error:
        st.error(f"초기화 오류: {error}")
        if "API" in error:
            st.info("Streamlit Secrets에 OPENAI_API_KEY를 설정해주세요.")
            with st.form("api_key_form"):
                api_key = st.text_input("OpenAI API 키", type="password")
                if st.form_submit_button("설정") and api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
                    st.rerun()
        return
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_with_tables(message["content"])
            else:
                st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요... (예: 2024년 청소년 과의존률은?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            
            try:
                node_functions = create_node_functions(vectorstore, llms, status_placeholder)
                graph = build_graph(node_functions)
                
                config = {"configurable": {"thread_id": "streamlit_session"}}
                
                result = graph.invoke(
                    {
                        "input": prompt,
                        "chat_history": st.session_state.chat_history,
                        "session_id": "streamlit_session",
                        "clarification_context": st.session_state.clarification_context,
                    },
                    config=config
                )
                
                status_placeholder.empty()
                
                # Clarification context 저장
                if result.get("clarification_context"):
                    st.session_state.clarification_context = result["clarification_context"]
                else:
                    st.session_state.clarification_context = None
                
                final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")
                
                with answer_placeholder.container():
                    render_answer_with_tables(final_answer)
                
                # 디버그 정보
                if debug_mode:
                    with st.expander("🔍 디버그 정보 (v5)", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Intent:** {result.get('intent', 'N/A')}")
                            st.write(f"**Followup:** {result.get('followup_type', 'N/A')}")
                            st.write(f"**Retry Count:** {result.get('retry_count', 0)}")
                            st.write(f"**Default Years Used:** {result.get('used_default_years', False)}")
                            
                            validation_result = result.get('validation_result', 'N/A')
                            if validation_result == "PASS":
                                st.markdown(f"**Validation:** <span class='validation-pass'>{validation_result}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**Validation:** <span class='validation-fail'>{validation_result}</span>", unsafe_allow_html=True)
                        
                        with col2:
                            if result.get("rewritten_queries"):
                                st.write("**Rewritten Queries:**")
                                for q in result["rewritten_queries"][:3]:
                                    st.caption(f"• {q[:50]}...")
                        
                        if result.get("retrieval"):
                            st.write(f"**검색 파일:** {result['retrieval'].get('files_searched', [])}")
                            st.write(f"**문서 수:** {result['retrieval'].get('doc_count', 0)}")
                        
                        if result.get("plan"):
                            st.write(f"**검색 연도:** {result['plan'].get('years', [])}")
                        
                        if result.get("validation_reason"):
                            st.write(f"**Validation Reason:** {result['validation_reason'][:100]}")
                        
                        st.write(f"**Safety:** passed={result.get('safety_passed', 'N/A')}")
                
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))
                
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]
                
            except Exception as e:
                status_placeholder.empty()
                st.error(f"오류가 발생했습니다: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()


