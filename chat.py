from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        color: black;
    }
    .assistant-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
        color: black;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"])
)

openai_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def run_rag_with_fallback(user_query: str) -> str:
    results = search_client.search(user_query, top=5)
    documents = [doc["content"] for doc in results if "content" in doc]

    if not documents:
        # Fallback: 문서 없이 사용자 질문만
        prompt = f"""
        문서 검색 결과가 없습니다.
        사용자 질문: "{user_query}"
        이 질문에 대해 일반적인 지식을 바탕으로 정중하게 답변해 주세요.
        """
    else:
        # 검색된 문서를 context로 사용
        context = "\n\n".join(documents)
        prompt = f"""
        당신은 문서 기반 질문 응답 도우미입니다.

        아래는 검색된 문서입니다:
        -------------------------
        {context}
        -------------------------

        사용자의 질문은 다음과 같습니다:
        "{user_query}"

        문서 내용을 최대한 반영하여 정확하고 친절하게 답변해 주세요.
        """

    # GPT 응답 생성
    response = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# 상부 섹션 (2/10 비율)
with st.container():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 트랜드 기반 VOD 추천 시스템</h1>
    </div>
    """, unsafe_allow_html=True)

# 하부 섹션 (8/10 비율)
with st.container():
    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 컨테이너
    chat_container = st.container()
    
    with chat_container:
        # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # 채팅 메시지 표시
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 사용자:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 AI:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 입력 섹션
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "질문을 입력하세요:",
                height=100,
                placeholder="여기에 질문을 입력하세요...",
                key="user_input"
            )
        
        with col2:
            st.write("")  # 공간 맞추기
            st.write("")  # 공간 맞추기
            if st.button("전송", key="send_button"):
                if user_input.strip():
                    # 사용자 메시지 추가
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # AI 응답 생성
                    with st.spinner("AI가 응답을 생성하고 있습니다..."):
                        ai_response = run_rag_with_fallback(user_input)
                    
                    # AI 메시지 추가
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    # 페이지 새로고침
                    st.rerun()

# 사이드바에 추가 정보
with st.sidebar:
    st.markdown("### 📊 시스템 정보")
    st.markdown("""
    - **모델**: Azure OpenAI
    - **검색**: Azure AI Search
    - **문서 수**: 최대 5개 검색
    """)
    
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# if __name__ == "__main__":
#     query = input("질문을 입력하세요: ")
#     answer = run_rag_with_fallback(query)
#     print("\n📘 GPT 응답:")
#     print(answer)
