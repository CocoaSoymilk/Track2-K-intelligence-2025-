import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import base64
import io
from typing import Dict, List
import tempfile

# 패키지 import 안전성 처리
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("Plotly 패키지가 설치되지 않았습니다. requirements.txt를 확인해주세요.")
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    st.warning("OpenAI 패키지가 설치되지 않았습니다. 시뮬레이션 모드로 실행합니다.")
    OPENAI_AVAILABLE = False
    openai = None

# 페이지 설정
st.set_page_config(
    page_title="소리로 쓰는 하루",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# OpenAI API 키 설정
@st.cache_resource
def init_openai():
    """OpenAI API 초기화"""
    if not OPENAI_AVAILABLE:
        return None
        
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        elif "openai_api_key" in st.session_state and st.session_state.openai_api_key:
            return openai.OpenAI(api_key=st.session_state.openai_api_key)
        else:
            return None
    except Exception as e:
        st.error(f"OpenAI 클라이언트 초기화 오류: {e}")
        return None

# OpenAI 클라이언트 초기화
openai_client = init_openai()

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    
    .emotion-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        transform: translateY(-3px);
    }
    
    .feedback-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
    }
    
    .recording-container {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px dashed #667eea;
    }
    
    .daily-summary {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
        }
        
        .emotion-card, .metric-container, .feedback-box {
            margin: 0.5rem 0;
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'diary_entries' not in st.session_state:
    st.session_state.diary_entries = []

def transcribe_audio_with_whisper(audio_bytes):
    """Whisper API를 사용하여 음성을 텍스트로 변환"""
    if not openai_client:
        return None
    
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Whisper API 호출
        with open(tmp_file_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"
            )
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        return transcript.text
        
    except Exception as e:
        st.error(f"음성 변환 오류: {str(e)}")
        return None

def analyze_emotion_with_gpt(text: str) -> Dict:
    """GPT-4를 사용하여 감정을 분석합니다."""
    if not openai_client:
        return analyze_emotion_simulation(text)
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 "소리로 쓰는 하루" 서비스의 따뜻하고 공감적인 AI 마음 분석가입니다. 
                    사용자가 음성이나 글로 들려준 하루 이야기를 분석하여 정확한 JSON 형식으로 응답해주세요.
                    
                    응답 형식:
                    {
                        "emotions": ["기쁨", "슬픔", "분노", "불안", "평온", "중립" 중 해당하는 것들의 배열],
                        "stress_level": 스트레스 수치 (0-100의 정수),
                        "energy_level": 에너지 수치 (0-100의 정수),
                        "mood_score": 전체적인 마음 점수 (-70부터 +70 사이의 정수),
                        "summary": "따뜻하고 공감적인 톤으로 한두 문장 요약",
                        "keywords": ["핵심 키워드들"],
                        "tone": "긍정적" 또는 "중립적" 또는 "부정적"
                    }
                    
                    사용자의 마음을 깊이 이해하고, 따뜻하게 공감하는 분석을 해주세요."""
                },
                {
                    "role": "user",
                    "content": f"오늘의 이야기를 들어주세요: {text}"
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            # 코드 블록 제거
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1]
            
            result = json.loads(result_text.strip())
            
            # 필수 필드 확인 및 기본값 설정
            required_fields = {
                'emotions': ['중립'],
                'stress_level': 30,
                'energy_level': 50,
                'mood_score': 0,
                'summary': '일반적인 상태입니다.',
                'keywords': [],
                'tone': '중립적'
            }
            
            for field, default_value in required_fields.items():
                if field not in result:
                    result[field] = default_value
            
            return result
            
        except json.JSONDecodeError:
            st.warning("GPT 응답 파싱 중 오류 발생. 시뮬레이션 모드로 전환합니다.")
            return analyze_emotion_simulation(text)
        
    except Exception as e:
        st.error(f"GPT 분석 오류: {str(e)}")
        return analyze_emotion_simulation(text)

def analyze_emotion_simulation(text: str) -> Dict:
    """GPT API 없이 기본 감정 분석"""
    emotions_map = {
        '기쁨': ['좋다', '행복', '기쁘다', '즐겁다', '웃음', '성공', '뿌듯', '만족', '사랑', '고마운'],
        '슬픔': ['슬프다', '우울', '눈물', '힘들다', '실망', '아프다', '외롭다', '그립다'],
        '분노': ['화나다', '짜증', '분하다', '억울', '답답', '열받다', '미치겠다'],
        '불안': ['걱정', '불안', '스트레스', '두렵다', '긴장', '무서워', '초조'],
        '평온': ['평온', '차분', '안정', '편안', '휴식', '여유', '고요']
    }
    
    detected_emotions = []
    stress_level = 30
    energy_level = 50
    keywords = []
    
    text_lower = text.lower()
    
    # 감정 키워드 감지
    for emotion, emotion_keywords in emotions_map.items():
        for keyword in emotion_keywords:
            if keyword in text_lower:
                detected_emotions.append(emotion)
                keywords.append(keyword)
                break
    
    # 스트레스와 에너지 수준 추정
    stress_keywords = ['스트레스', '힘들다', '피곤', '지쳐', '화나다', '걱정', '바쁘다']
    energy_keywords = ['좋다', '행복', '에너지', '활기', '뿌듯', '즐겁다', '신나다']
    
    stress_count = sum(1 for word in stress_keywords if word in text_lower)
    energy_count = sum(1 for word in energy_keywords if word in text_lower)
    
    if stress_count > energy_count:
        stress_level = min(80, 40 + stress_count * 15)
        energy_level = max(20, 60 - stress_count * 15)
        tone = "부정적"
    elif energy_count > stress_count:
        stress_level = max(15, 40 - energy_count * 10)
        energy_level = min(85, 50 + energy_count * 15)
        tone = "긍정적"
    else:
        tone = "중립적"
    
    mood_score = energy_level - stress_level
    
    return {
        'emotions': detected_emotions if detected_emotions else ['중립'],
        'stress_level': stress_level,
        'energy_level': energy_level,
        'mood_score': mood_score,
        'summary': f"{tone} 상태로, 주요 감정은 {', '.join(detected_emotions[:2]) if detected_emotions else '중립'}입니다.",
        'keywords': keywords[:5],
        'tone': tone
    }

def generate_personalized_feedback(entries: List[Dict]) -> str:
    """개인화된 피드백 생성"""
    if not entries:
        return "첫 번째 음성 일기를 작성해보세요! 🎙️"
    
    recent_entries = entries[-7:]  # 최근 7일
    
    if not openai_client:
        return generate_basic_feedback(recent_entries)
    
    try:
        # 최근 데이터 요약
        summary_data = []
        for entry in recent_entries:
            summary_data.append({
                'date': entry['date'],
                'emotions': entry['analysis']['emotions'],
                'stress': entry['analysis']['stress_level'],
                'energy': entry['analysis']['energy_level'],
                'tone': entry['analysis'].get('tone', '중립적')
            })
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 "소리로 쓰는 하루" 서비스의 따뜻하고 공감적인 AI 마음 케어 코치입니다. 
                    사용자의 최근 일주일간 마음 데이터를 분석하여 다음을 제공하세요:
                    
                    1. 마음 패턴에 대한 따뜻한 관찰 (1-2문장)
                    2. 구체적이고 실용적인 마음 케어 조언 (1-2문장)
                    3. 격려와 위로의 메시지 (1문장)
                    
                    전체 3-4문장으로, 친근하고 따뜻한 톤으로 작성해주세요.
                    의학적 진단이나 치료를 언급하지 말고, 일상적인 마음 케어 조언에 집중하세요."""
                },
                {
                    "role": "user",
                    "content": f"최근 일주일간의 마음 데이터를 살펴봐 주세요:\n{json.dumps(summary_data, ensure_ascii=False, indent=2)}"
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return generate_basic_feedback(recent_entries)

def generate_basic_feedback(entries: List[Dict]) -> str:
    """기본 피드백 생성"""
    if not entries:
        return "첫 번째 음성 일기를 작성해보세요! 🎙️"
    
    avg_stress = sum(entry['analysis']['stress_level'] for entry in entries) / len(entries)
    avg_energy = sum(entry['analysis']['energy_level'] for entry in entries) / len(entries)
    
    # 감정 빈도 분석
    all_emotions = []
    for entry in entries:
        all_emotions.extend(entry['analysis']['emotions'])
    
    emotion_counts = {}
    for emotion in all_emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    most_frequent = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "중립"
    
    if avg_stress > 65:
        return f"최근 스트레스 지수가 {avg_stress:.0f}%로 높은 편이에요. 깊은 호흡이나 짧은 산책으로 마음을 달래보세요. 작은 휴식도 큰 도움이 됩니다! 🌿"
    elif avg_energy < 35:
        return f"최근 에너지가 {avg_energy:.0f}%로 낮아 보여요. 충분한 수면과 좋아하는 활동으로 에너지를 충전해보세요. 당신을 위한 시간을 가져보세요! ⚡"
    elif most_frequent == "기쁨":
        return f"최근 긍정적인 감정이 많이 보이네요! 이 좋은 에너지를 유지하며 새로운 목표에 도전해보는 건 어떨까요? ✨"
    else:
        return f"전체적으로 안정적인 상태를 보이고 있어요. 꾸준히 자신의 감정을 기록하는 습관이 정말 훌륭합니다! 계속 응원할게요! 👍"

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🎙️ 소리로 쓰는 하루</h1>
    <p>목소리로 담는 오늘, AI가 읽어주는 마음</p>
    <small style="opacity: 0.8;">하루 1분, 내 마음을 알아가는 시간</small>
</div>
""", unsafe_allow_html=True)

# API 키 설정 체크
if not openai_client:
    with st.sidebar:
        st.warning("🔑 OpenAI API 키가 필요합니다")
        with st.expander("API 키 입력하기"):
            st.markdown("**소리로 쓰는 하루**의 AI 감정 분석을 위해 OpenAI API 키가 필요해요.")
            api_key = st.text_input("OpenAI API 키", type="password", help="sk-로 시작하는 API 키를 입력하세요")
            if st.button("저장"):
                if api_key.startswith("sk-"):
                    st.session_state.openai_api_key = api_key
                    openai_client = openai.OpenAI(api_key=api_key)
                    st.success("API 키가 저장되었습니다!")
                    st.rerun()
                else:
                    st.error("올바른 API 키 형식이 아닙니다.")
        
        st.info("💡 API 키 없이도 기본 감정 분석 기능을 체험할 수 있어요.")

# 사이드바 네비게이션
with st.sidebar:
    st.title("🌟 오늘의 마음")
    
    # 오늘 일기 작성 여부 확인
    today = datetime.now().strftime("%Y-%m-%d")
    today_entries = [entry for entry in st.session_state.diary_entries if entry['date'] == today]
    
    if today_entries:
        st.success(f"✅ 오늘 {len(today_entries)}번의 마음을 기록했어요")
    else:
        st.info("💭 오늘의 이야기를 들려주세요")
    
    page = st.selectbox(
        "페이지 선택",
        ["🎙️ 오늘의 이야기", "💖 마음 분석", "📈 감정 여정", "💡 마음 케어", "📚 나의 이야기들"],
        help="원하는 페이지를 선택하세요"
    )
    
    st.markdown("---")
    
    # 통계 요약
    if st.session_state.diary_entries:
        st.markdown("### 📊 나의 여정")
        total_entries = len(st.session_state.diary_entries)
        st.metric("기록한 이야기", f"{total_entries}개")
        
        if total_entries > 0:
            latest_entry = st.session_state.diary_entries[-1]
            days_since_start = (datetime.now() - datetime.strptime(st.session_state.diary_entries[0]['date'], "%Y-%m-%d")).days + 1
            st.metric("함께한 날들", f"{days_since_start}일째")
            
            # 최근 감정 상태
            recent_mood = latest_entry['analysis'].get('tone', '중립적')
            mood_emoji = {"긍정적": "😊", "중립적": "😐", "부정적": "😔"}
            st.metric("지금의 마음", f"{mood_emoji.get(recent_mood, '😐')} {recent_mood}")

# 페이지별 콘텐츠
if page == "🎙️ 오늘의 이야기":
    st.header("오늘 하루는 어떠셨나요?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **💝 마음을 나누는 시간:**
        - 1분만 투자해보세요, 당신의 이야기가 소중해요
        - 오늘 있었던 일, 느낀 감정을 자유롭게 말해보세요
        - 특별한 일이 없어도 괜찮아요, 평범한 하루도 의미 있어요
        """)
    
    with col2:
        # 오늘 작성한 일기 수
        if today_entries:
            st.info(f"🌟 오늘 {len(today_entries)}번째 이야기")
        else:
            st.info("🌱 오늘 첫 번째 이야기")
    
    # 음성 녹음 섹션
    st.markdown("### 🎙️ 목소리로 들려주세요")
    
    with st.container():
        st.markdown('<div class="recording-container">', unsafe_allow_html=True)
        
        # Streamlit 내장 음성 입력 사용
        audio_value = st.audio_input(
            "🎤 마음을 편하게 말해보세요",
            help="마이크 버튼을 눌러 녹음을 시작하세요. 마음이 편안해질 때까지 천천히 이야기해도 좋아요"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 텍스트 입력 대안
    st.markdown("### ✏️ 또는 글로 적어보세요")
    text_input = st.text_area(
        "마음을 글로 표현해보세요",
        placeholder="오늘은 어떤 하루였나요? 느낀 감정이나 생각을 자유롭게 써보세요...",
        height=120,
        help="목소리 대신 글로 마음을 표현하셔도 좋아요"
    )
    
    # 일기 저장 버튼
    if st.button("💝 마음 분석하고 소중히 보관하기", type="primary", use_container_width=True):
        diary_text = ""
        audio_data = None
        
        # 음성 데이터 처리
        if audio_value is not None:
            audio_bytes = audio_value.read()
            audio_data = base64.b64encode(audio_bytes).decode()
            
            with st.spinner("🤖 당신의 목소리를 마음으로 변환하는 중..."):
                if openai_client:
                    diary_text = transcribe_audio_with_whisper(audio_bytes)
                    if diary_text:
                        st.success("✅ 목소리가 글로 바뀌었어요!")
                        st.info(f"**들은 이야기:** {diary_text}")
                    else:
                        st.error("음성 변환에 실패했어요. 글로 적어주실 수 있나요?")
                else:
                    st.warning("API 키가 없어 음성 변환을 할 수 없어요. 글로 적어주세요.")
        
        # 텍스트 입력 처리
        if not diary_text and text_input.strip():
            diary_text = text_input.strip()
        
        if diary_text:
            with st.spinner("🤖 AI가 당신의 마음을 읽고 있어요..."):
                analysis = analyze_emotion_with_gpt(diary_text)
            
            # 일기 저장
            entry = {
                'id': len(st.session_state.diary_entries) + 1,
                'date': datetime.now().strftime("%Y-%m-%d"),
                'time': datetime.now().strftime("%H:%M"),
                'text': diary_text,
                'analysis': analysis,
                'timestamp': datetime.now(),
                'audio_data': audio_data
            }
            
            st.session_state.diary_entries.append(entry)
            
            # 결과 표시
            st.success("🎉 소중한 이야기가 안전하게 보관되었어요!")
            
            # 분석 결과 표시
            st.markdown("---")
            st.markdown("## 🤖 AI가 읽어드린 당신의 마음")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="emotion-card">
                    <h4>💖 감지된 감정</h4>
                    <p><strong>{', '.join(analysis['emotions'])}</strong></p>
                    <small>핵심 단어: {', '.join(analysis.get('keywords', [])[:3])}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="emotion-card">
                    <h4>📊 마음 상태</h4>
                    <p>스트레스: <strong>{analysis['stress_level']}%</strong></p>
                    <p>활력: <strong>{analysis['energy_level']}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="emotion-card">
                    <h4>🎯 오늘의 컨디션</h4>
                    <p>마음 점수: <strong>{analysis['mood_score']}</strong></p>
                    <p>전체 느낌: <strong>{analysis.get('tone', '중립적')}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # AI 요약
            if 'summary' in analysis:
                st.markdown(f"""
                <div class="feedback-box">
                    <h4>🤖 AI가 전해드리는 말</h4>
                    <p>{analysis['summary']}</p>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ 목소리나 글로 마음을 들려주세요!")

elif page == "💖 마음 분석":
    st.header("마음 분석 대시보드")
    
    if not st.session_state.diary_entries:
        st.info("📝 아직 기록된 이야기가 없어요. 첫 번째 이야기를 들려주세요!")
    else:
        # 필터 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_filter = st.selectbox(
                "기간 필터",
                ["전체", "오늘", "최근 7일", "최근 30일"],
                index=2
            )
        
        with col2:
            emotion_filter = st.selectbox(
                "감정 필터", 
                ["전체"] + list(set([emotion for entry in st.session_state.diary_entries for emotion in entry['analysis']['emotions']]))
            )
        
        # 필터 적용
        filtered_entries = st.session_state.diary_entries.copy()
        
        if date_filter == "오늘":
            today = datetime.now().strftime("%Y-%m-%d")
            filtered_entries = [e for e in filtered_entries if e['date'] == today]
        elif date_filter == "최근 7일":
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            filtered_entries = [e for e in filtered_entries if e['date'] >= week_ago]
        elif date_filter == "최근 30일":
            month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            filtered_entries = [e for e in filtered_entries if e['date'] >= month_ago]
        
        if emotion_filter != "전체":
            filtered_entries = [e for e in filtered_entries if emotion_filter in e['analysis']['emotions']]
        
        if not filtered_entries:
            st.warning("선택한 필터에 해당하는 일기가 없습니다.")
        else:
            # 오늘의 요약 (오늘 일기가 있는 경우)
            today_entries = [e for e in filtered_entries if e['date'] == datetime.now().strftime("%Y-%m-%d")]
            if today_entries and date_filter in ["전체", "오늘", "최근 7일", "최근 30일"]:
                st.markdown("### 📅 오늘의 감정 요약")
                latest = today_entries[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("주요 감정", ', '.join(latest['analysis']['emotions'][:2]))
                with col2:
                    st.metric("스트레스", f"{latest['analysis']['stress_level']}%")
                with col3:
                    st.metric("활력", f"{latest['analysis']['energy_level']}%")
                with col4:
                    mood_emoji = "😊" if latest['analysis']['mood_score'] > 10 else "😐" if latest['analysis']['mood_score'] > -10 else "😔"
                    st.metric("기분", f"{mood_emoji} {latest['analysis']['mood_score']}")
            
            # 일기 목록
            st.markdown(f"### 📝 일기 목록 ({len(filtered_entries)}개)")
            
            # 페이지네이션
            items_per_page = 5
            total_pages = (len(filtered_entries) - 1) // items_per_page + 1
            current_page = st.select_slider("페이지", range(1, total_pages + 1), value=1)
            
            start_idx = (current_page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            current_entries = list(reversed(filtered_entries))[start_idx:end_idx]
            
            for i, entry in enumerate(current_entries):
                with st.expander(
                    f"📅 {entry['date']} {entry['time']} - {', '.join(entry['analysis']['emotions'])} "
                    f"({'😊' if entry['analysis']['mood_score'] > 10 else '😐' if entry['analysis']['mood_score'] > -10 else '😔'})"
                ):
                    # 일기 내용
                    st.markdown(f"**📝 내용:** {entry['text']}")
                    
                    # 음성 파일 재생
                    if entry.get('audio_data'):
                        st.markdown("**🎵 녹음된 음성:**")
                        audio_bytes = base64.b64decode(entry['audio_data'])
                        st.audio(audio_bytes)
                    
                    # 분석 결과
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("스트레스", f"{entry['analysis']['stress_level']}%")
                    with col2:
                        st.metric("활력", f"{entry['analysis']['energy_level']}%")
                    with col3:
                        st.metric("기분 점수", f"{entry['analysis']['mood_score']}")
                    
                    # AI 요약
                    if 'summary' in entry['analysis']:
                        st.info(f"🤖 **AI 분석:** {entry['analysis']['summary']}")
                    
                    # 키워드
                    if entry['analysis'].get('keywords'):
                        st.markdown(f"**🏷️ 키워드:** {', '.join(entry['analysis']['keywords'])}")

elif page == "📈 감정 여정":
    st.header("마음의 변화를 살펴보세요")
    
    if not st.session_state.diary_entries:
        st.info("📊 이야기를 기록하면 마음의 변화를 아름다운 그래프로 볼 수 있어요!")
    else:
        # 기간 선택
        period_options = {
            "최근 7일": 7,
            "최근 30일": 30,
            "최근 90일": 90,
            "전체": None
        }
        
        selected_period = st.selectbox("📅 분석 기간", list(period_options.keys()), index=1)
        
        entries_to_analyze = st.session_state.diary_entries
        if period_options[selected_period]:
            entries_to_analyze = st.session_state.diary_entries[-period_options[selected_period]:]
        
        # 데이터 준비
        df = pd.DataFrame([
            {
                'date': entry['date'],
                'time': entry['time'],
                'datetime': f"{entry['date']} {entry['time']}",
                'stress': entry['analysis']['stress_level'],
                'energy': entry['analysis']['energy_level'],
                'mood': entry['analysis']['mood_score'],
                'emotions': ', '.join(entry['analysis']['emotions'][:2]),
                'tone': entry['analysis'].get('tone', '중립적')
            }
            for entry in entries_to_analyze
        ])
        
        # 일별 평균 계산
        daily_avg = df.groupby('date').agg({
            'stress': 'mean',
            'energy': 'mean',
            'mood': 'mean'
        }).reset_index()
        
        # 메인 그래프들
        col1, col2 = st.columns(2)
        
        with col1:
            # 시간별 감정 변화
            st.subheader("📈 일별 감정 변화")
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=daily_avg['date'],
                    y=daily_avg['stress'],
                    name='스트레스',
                    line=dict(color='#ff6b6b', width=3),
                    hovertemplate='%{x}<br>스트레스: %{y:.1f}%<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter(
                    x=daily_avg['date'],
                    y=daily_avg['energy'],
                    name='활력',
                    line=dict(color='#51cf66', width=3),
                    hovertemplate='%{x}<br>활력: %{y:.1f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="날짜",
                    yaxis_title="수치 (%)",
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Plotly 없이 표 형태로 표시
                st.dataframe(daily_avg[['date', 'stress', 'energy']], use_container_width=True)
        
        with col2:
            # 감정 분포
            st.subheader("😊 감정 분포")
            all_emotions = []
            for entry in entries_to_analyze:
                all_emotions.extend(entry['analysis']['emotions'])
            
            if all_emotions:
                emotion_counts = pd.Series(all_emotions).value_counts()
                
                if PLOTLY_AVAILABLE:
                    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
                    
                    fig_pie = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="감정별 빈도",
                        color_discrete_sequence=colors
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    # Plotly 없이 바차트 형태로 표시
                    st.bar_chart(emotion_counts)
        
        # 추가 분석
        st.subheader("📊 상세 분석")
        
        if PLOTLY_AVAILABLE:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 톤 분포
                tone_counts = df['tone'].value_counts()
                fig_tone = px.bar(
                    x=tone_counts.index,
                    y=tone_counts.values,
                    title="일기 톤 분포",
                    color=tone_counts.values,
                    color_continuous_scale='RdYlGn'
                )
                fig_tone.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_tone, use_container_width=True)
            
            with col2:
                # 기분 점수 히스토그램
                fig_mood = px.histogram(
                    df,
                    x='mood',
                    nbins=15,
                    title="기분 점수 분포",
                    color_discrete_sequence=['#74c0fc']
                )
                fig_mood.update_layout(height=300)
                st.plotly_chart(fig_mood, use_container_width=True)
            
            with col3:
                # 스트레스 vs 활력 산점도
                fig_scatter = px.scatter(
                    df,
                    x='stress',
                    y='energy',
                    title="스트레스 vs 활력 관계",
                    color='mood',
                    color_continuous_scale='RdYlGn',
                    hover_data=['date', 'emotions']
                )
                fig_scatter.update_layout(height=300)
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            # Plotly 없이 간단한 차트로 대체
            st.info("📊 더 자세한 그래프를 보려면 plotly 패키지 설치가 필요합니다.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("톤 분포")
                tone_counts = df['tone'].value_counts()
                st.bar_chart(tone_counts)
                
            with col2:
                st.subheader("기분 점수 분포")
                st.bar_chart(df['mood'].value_counts().sort_index())
        
        # 통계 요약
        st.subheader("📈 통계 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        avg_stress = df['stress'].mean()
        avg_energy = df['energy'].mean()
        avg_mood = df['mood'].mean()
        total_entries = len(df)
        
        with col1:
            st.metric(
                "평균 스트레스", 
                f"{avg_stress:.1f}%",
                delta=f"{avg_stress - 50:.1f}%" if len(df) > 1 else None
            )
        
        with col2:
            st.metric(
                "평균 활력", 
                f"{avg_energy:.1f}%",
                delta=f"{avg_energy - 50:.1f}%" if len(df) > 1 else None
            )
        
        with col3:
            st.metric(
                "평균 기분", 
                f"{avg_mood:.1f}",
                delta=f"{avg_mood:.1f}" if len(df) > 1 else None
            )
        
        with col4:
            st.metric("분석 기간", f"{total_entries}개 일기")

elif page == "💡 마음 케어":
    st.header("당신만을 위한 마음 케어")
    
    if not st.session_state.diary_entries:
        st.info("📝 이야기를 기록하면 AI가 당신만의 맞춤 케어를 추천해드려요!")
    else:
        # AI 피드백
        with st.spinner("🤖 AI가 당신만의 마음 케어 방법을 찾고 있어요..."):
            feedback = generate_personalized_feedback(st.session_state.diary_entries)
        
        st.markdown(f"""
        <div class="feedback-box">
            <h3>🤖 AI 마음 케어 코치의 메시지</h3>
            <p style="font-size: 1.1em; line-height: 1.6;">{feedback}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 개인 통계 카드
        st.subheader("📊 나의 마음 여정 리포트")
        
        recent_entries = st.session_state.diary_entries[-30:]
        if recent_entries:
            avg_stress = sum(entry['analysis']['stress_level'] for entry in recent_entries) / len(recent_entries)
            avg_energy = sum(entry['analysis']['energy_level'] for entry in recent_entries) / len(recent_entries)
            avg_mood = sum(entry['analysis']['mood_score'] for entry in recent_entries) / len(recent_entries)
            
            # 트렌드 계산
            if len(recent_entries) >= 5:
                recent_5 = recent_entries[-5:]
                previous_5 = recent_entries[-10:-5] if len(recent_entries) >= 10 else recent_entries[:-5]
                
                if previous_5:
                    stress_trend = avg_stress - (sum(entry['analysis']['stress_level'] for entry in previous_5) / len(previous_5))
                    energy_trend = avg_energy - (sum(entry['analysis']['energy_level'] for entry in previous_5) / len(previous_5))
                    mood_trend = avg_mood - (sum(entry['analysis']['mood_score'] for entry in previous_5) / len(previous_5))
                else:
                    stress_trend = energy_trend = mood_trend = 0
            else:
                stress_trend = energy_trend = mood_trend = 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #ff6b6b;">😰 평균 스트레스</h3>
                    <h2>{avg_stress:.1f}%</h2>
                    <p style="color: {'red' if stress_trend > 0 else 'green' if stress_trend < 0 else 'gray'};">
                        {'↗️' if stress_trend > 5 else '↘️' if stress_trend < -5 else '→'} 
                        {abs(stress_trend):.1f}% {'증가' if stress_trend > 0 else '감소' if stress_trend < 0 else '유지'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #51cf66;">⚡ 평균 활력</h3>
                    <h2>{avg_energy:.1f}%</h2>
                    <p style="color: {'green' if energy_trend > 0 else 'red' if energy_trend < 0 else 'gray'};">
                        {'↗️' if energy_trend > 5 else '↘️' if energy_trend < -5 else '→'} 
                        {abs(energy_trend):.1f}% {'증가' if energy_trend > 0 else '감소' if energy_trend < 0 else '유지'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #339af0;">😊 평균 기분</h3>
                    <h2>{avg_mood:.1f}</h2>
                    <p style="color: {'green' if mood_trend > 0 else 'red' if mood_trend < 0 else 'gray'};">
                        {'↗️' if mood_trend > 3 else '↘️' if mood_trend < -3 else '→'} 
                        {abs(mood_trend):.1f} {'개선' if mood_trend > 0 else '하락' if mood_trend < 0 else '안정'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # 맞춤 웰빙 가이드
        st.subheader("🧘‍♀️ 맞춤 웰빙 가이드")
        
        if st.session_state.diary_entries:
            latest_entry = st.session_state.diary_entries[-1]
            stress_level = latest_entry['analysis']['stress_level']
            energy_level = latest_entry['analysis']['energy_level']
            recent_emotions = latest_entry['analysis']['emotions']
            
            # 상태에 따른 추천 활동 결정
            if stress_level > 60:
                recommended_activity = "스트레스 해소"
                activity_icon = "🌊"
                activity_description = """
                **4-7-8 호흡법으로 마음 진정하기**
                
                1. **4초 동안** 코로 천천히 숨 들이마시기
                2. **7초 동안** 숨 참기 (편안하게)
                3. **8초 동안** 입으로 천천히 내쉬기
                4. **3-4회 반복**하며 몸의 긴장 풀어주기
                
                *스트레스 호르몬 분비를 줄이고 신경계를 안정시켜줍니다*
                """
                
            elif energy_level < 40:
                recommended_activity = "에너지 충전"
                activity_icon = "☀️"
                activity_description = """
                **활력 충전 시각화 명상**
                
                1. **편안한 자세**로 앉아 눈을 감으세요
                2. **따뜻한 황금빛**이 머리 위에서 내려오는 상상하기
                3. **온몸을 감싸는** 따뜻함과 에너지를 느끼기
                4. **10분간** 이 감각에 집중하며 에너지 흡수하기
                
                *세로토닌 분비를 촉진하고 활력을 회복시켜줍니다*
                """
                
            elif "불안" in recent_emotions:
                recommended_activity = "불안 완화"
                activity_icon = "🌿"
                activity_description = """
                **5-4-3-2-1 그라운딩 기법**
                
                주변에서 찾아보세요:
                - **5개의 것**을 보기 (시각)
                - **4개의 소리** 듣기 (청각)
                - **3개의 질감** 만져보기 (촉각)
                - **2개의 냄새** 맡기 (후각)
                - **1개의 맛** 느끼기 (미각)
                
                *현재에 집중하며 불안을 줄여주는 효과적인 방법입니다*
                """
                
            else:
                recommended_activity = "감사 명상"
                activity_icon = "🙏"
                activity_description = """
                **감사 일기 명상**
                
                1. **오늘 하루** 중 감사한 일 3가지 떠올리기
                2. **작은 것도 포함**하기 (맛있는 커피, 따뜻한 햇살 등)
                3. **각각에 대해** 왜 감사한지 구체적으로 생각하기
                4. **그 감정을** 마음에 깊이 새기기
                
                *행복감을 증진시키고 긍정적인 마음가짐을 기를 수 있어요*
                """
            
            with st.expander(f"{activity_icon} **추천: {recommended_activity}**", expanded=True):
                st.markdown(activity_description)
                
                # 완료 체크
                if st.button(f"✅ {recommended_activity} 완료!", key="wellness_complete"):
                    st.success("🎉 훌륭해요! 자신을 위한 시간을 가져주셔서 감사합니다.")
                    st.balloons()
        
        # 추가 웰빙 리소스
        st.subheader("📚 추가 웰빙 리소스")
        
        wellness_tabs = st.tabs(["🧠 마음챙김", "💪 신체 활동", "🎵 음악 테라피", "📖 자기계발"])
        
        with wellness_tabs[0]:
            st.markdown("""
            **🧘‍♀️ 일일 마음챙김 루틴**
            
            - **아침**: 5분 호흡 명상으로 하루 시작
            - **점심**: 식사할 때 음식의 맛과 향에 집중
            - **저녁**: 하루를 되돌아보는 감사 시간
            - **잠들기 전**: 바디스캔으로 몸과 마음 이완
            """)
        
        with wellness_tabs[1]:
            st.markdown("""
            **🏃‍♀️ 기분 좋아지는 신체 활동**
            
            - **10분 산책**: 자연을 보며 걷기
            - **5분 스트레칭**: 목, 어깨, 허리 풀어주기  
            - **계단 오르기**: 심박수 올려 엔돌핀 분비
            - **춤추기**: 좋아하는 음악에 맞춰 자유롭게
            """)
        
        with wellness_tabs[2]:
            st.markdown("""
            **🎼 상황별 추천 음악**
            
            - **스트레스 해소**: 클래식, 자연 소리, 로파이
            - **에너지 충전**: 업템포 팝, 댄스 뮤직
            - **집중력 향상**: 백색 소음, 포커스 음악
            - **수면 유도**: 명상 음악, ASMR
            """)
        
        with wellness_tabs[3]:
            st.markdown("""
            **📚 성장을 위한 작은 습관**
            
            - **일기 쓰기**: 매일 3줄이라도 감정 기록하기
            - **독서**: 하루 10페이지씩 읽기
            - **새로운 학습**: 온라인 강의 10분씩 듣기
            - **인간관계**: 소중한 사람에게 안부 묻기
            """)

elif page == "📚 나의 이야기들":
    st.header("소중한 이야기 아카이브")
    
    if not st.session_state.diary_entries:
        st.info("📝 아직 기록된 이야기가 없어요.")
    else:
        # 검색 및 필터
        col1, col2 = st.columns(2)
        
        with col1:
            search_query = st.text_input("🔍 이야기 내용 검색", placeholder="찾고 싶은 기억을 검색해보세요")
        
        with col2:
            sort_order = st.selectbox("정렬 순서", ["최신순", "오래된순", "기분 좋은순", "힘들었던순"])
        
        # 데이터 필터링 및 정렬
        filtered_entries = st.session_state.diary_entries.copy()
        
        if search_query:
            filtered_entries = [
                entry for entry in filtered_entries
                if search_query.lower() in entry['text'].lower()
            ]
        
        # 정렬
        if sort_order == "최신순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['timestamp'], reverse=True)
        elif sort_order == "오래된순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['timestamp'])
        elif sort_order == "기분 좋은순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['analysis']['mood_score'], reverse=True)
        elif sort_order == "힘들었던순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['analysis']['mood_score'])
        
        if filtered_entries:
            st.write(f"📊 총 {len(filtered_entries)}개의 소중한 이야기를 찾았어요.")
            
            # 월별 그룹화
            monthly_groups = {}
            for entry in filtered_entries:
                month_key = entry['date'][:7]  # YYYY-MM
                if month_key not in monthly_groups:
                    monthly_groups[month_key] = []
                monthly_groups[month_key].append(entry)
            
            # 월별 표시
            for month, entries in sorted(monthly_groups.items(), reverse=(sort_order == "최신순")):
                with st.expander(f"📅 {month} ({len(entries)}개 이야기)", expanded=(month == max(monthly_groups.keys()))):
                    
                    # 월 요약 통계
                    avg_mood = sum(entry['analysis']['mood_score'] for entry in entries) / len(entries)
                    avg_stress = sum(entry['analysis']['stress_level'] for entry in entries) / len(entries)
                    avg_energy = sum(entry['analysis']['energy_level'] for entry in entries) / len(entries)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("이 달의 평균 마음", f"{avg_mood:.1f}")
                    with col2:
                        st.metric("이 달의 평균 스트레스", f"{avg_stress:.1f}%")
                    with col3:
                        st.metric("이 달의 평균 활력", f"{avg_energy:.1f}%")
                    
                    st.markdown("---")
                    
                    # 해당 월 일기들
                    for entry in entries:
                        mood_emoji = "😊" if entry['analysis']['mood_score'] > 10 else "😐" if entry['analysis']['mood_score'] > -10 else "😔"
                        
                        with st.container():
                            st.markdown(f"""
                            **📅 {entry['date']} {entry['time']} {mood_emoji}**  
                            **마음:** {', '.join(entry['analysis']['emotions'])}  
                            **이야기:** {entry['text'][:100]}{'...' if len(entry['text']) > 100 else ''}
                            """)
                            
                            # 상세 보기 버튼
                            if st.button(f"💝 자세히 보기", key=f"detail_{entry['id']}"):
                                st.markdown("---")
                                st.markdown(f"**📖 전체 이야기:**\n{entry['text']}")
                                
                                if entry.get('audio_data'):
                                    st.markdown("**🎵 당시의 목소리:**")
                                    audio_bytes = base64.b64decode(entry['audio_data'])
                                    st.audio(audio_bytes)
                                
                                if 'summary' in entry['analysis']:
                                    st.info(f"🤖 **AI가 읽어드린 마음:** {entry['analysis']['summary']}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("스트레스", f"{entry['analysis']['stress_level']}%")
                                with col2:
                                    st.metric("활력", f"{entry['analysis']['energy_level']}%")
                                with col3:
                                    st.metric("마음 점수", f"{entry['analysis']['mood_score']}")
                                
                                st.markdown("---")
                            
                            st.markdown("---")
        else:
            st.warning("찾으시는 이야기가 없네요. 다른 검색어로 시도해보세요.")

# 사이드바 - 데이터 관리
with st.sidebar:
    if st.session_state.diary_entries:
        st.markdown("---")
        st.markdown("### 💾 소중한 기록 관리")
        
        # 통계 내보내기
        if st.button("📊 마음 리포트 생성"):
            # 데이터프레임 생성
            df_export = pd.DataFrame([
                {
                    'date': entry['date'],
                    'time': entry['time'],
                    'text': entry['text'],
                    'emotions': ', '.join(entry['analysis']['emotions']),
                    'stress_level': entry['analysis']['stress_level'],
                    'energy_level': entry['analysis']['energy_level'],
                    'mood_score': entry['analysis']['mood_score'],
                    'summary': entry['analysis'].get('summary', ''),
                    'keywords': ', '.join(entry['analysis'].get('keywords', [])),
                    'tone': entry['analysis'].get('tone', '중립적')
                }
                for entry in st.session_state.diary_entries
            ])
            
            csv = df_export.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📁 마음 리포트 다운로드",
                data=csv,
                file_name=f"소리로_쓰는_하루_리포트_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        
        # 백업 저장
        if st.button("💾 전체 이야기 백업"):
            backup_data = {
                'service_name': '소리로 쓰는 하루',
                'entries': st.session_state.diary_entries,
                'export_date': datetime.now().isoformat(),
                'total_count': len(st.session_state.diary_entries)
            }
            backup_json = json.dumps(backup_data, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="📦 백업 파일 다운로드",
                data=backup_json,
                file_name=f"소리로_쓰는_하루_백업_{datetime.now().strftime('%Y%m%d')}.json",
                mime='application/json'
            )
        
        # 데이터 초기화
        st.markdown("---")
        if st.button("🗑️ 모든 기록 삭제", type="secondary"):
            if st.checkbox("⚠️ 정말로 소중한 모든 이야기를 삭제하시겠어요?"):
                st.session_state.diary_entries = []
                st.success("✅ 모든 기록이 삭제되었어요. 새로운 시작이에요!")
                st.rerun()

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🎙️ <strong>소리로 쓰는 하루</strong> - 목소리로 담는 오늘, AI가 읽어주는 마음</p>
    <p>하루 1분, 당신의 소중한 이야기를 들려주세요 ✨</p>
    <small style="color: #999;">Made with ❤️ using Streamlit & OpenAI</small>
</div>
""", unsafe_allow_html=True)
