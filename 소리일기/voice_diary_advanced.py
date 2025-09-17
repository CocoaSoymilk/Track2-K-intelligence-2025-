import streamlit as st
from datetime import datetime, timedelta
import json
import base64
from typing import Dict, List

# 페이지 설정
st.set_page_config(
    page_title="소리일기",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e9ecef;
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
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'diary_entries' not in st.session_state:
    st.session_state.diary_entries = []

def analyze_emotion_simple(text: str) -> Dict:
    """간단한 키워드 기반 감정 분석"""
    emotions_map = {
        '기쁨': ['좋다', '행복', '기쁘다', '즐겁다', '웃음', '성공', '뿌듯', '만족', '사랑', '고마운', '신나다'],
        '슬픔': ['슬프다', '우울', '눈물', '힘들다', '실망', '아프다', '외롭다', '그립다', '안타깝다'],
        '분노': ['화나다', '짜증', '분하다', '억울', '답답', '열받다', '미치겠다', '빡치다'],
        '불안': ['걱정', '불안', '스트레스', '두렵다', '긴장', '무서워', '초조', '조급'],
        '평온': ['평온', '차분', '안정', '편안', '휴식', '여유', '고요', '평화']
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
    stress_keywords = ['스트레스', '힘들다', '피곤', '지쳐', '화나다', '걱정', '바쁘다', '답답', '짜증']
    energy_keywords = ['좋다', '행복', '에너지', '활기', '뿌듯', '즐겁다', '신나다', '만족', '성공']
    
    stress_count = sum(1 for word in stress_keywords if word in text_lower)
    energy_count = sum(1 for word in energy_keywords if word in text_lower)
    
    if stress_count > energy_count:
        stress_level = min(85, 40 + stress_count * 20)
        energy_level = max(15, 60 - stress_count * 15)
        tone = "부정적"
    elif energy_count > stress_count:
        stress_level = max(10, 40 - energy_count * 15)
        energy_level = min(90, 50 + energy_count * 20)
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

def generate_simple_feedback(entries: List[Dict]) -> str:
    """간단한 피드백 생성"""
    if not entries:
        return "첫 번째 음성 일기를 작성해보세요! 🎙️"
    
    recent_entries = entries[-7:]  # 최근 7일
    avg_stress = sum(entry['analysis']['stress_level'] for entry in recent_entries) / len(recent_entries)
    avg_energy = sum(entry['analysis']['energy_level'] for entry in recent_entries) / len(recent_entries)
    
    # 감정 빈도 분석
    all_emotions = []
    for entry in recent_entries:
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
    <h1>🎙️ 소리일기</h1>
    <p>하루를 음성으로 기록하고, AI가 분석해주는 나만의 감정 아카이브</p>
</div>
""", unsafe_allow_html=True)

# 사이드바 네비게이션
with st.sidebar:
    st.title("📱 메뉴")
    
    # 오늘 일기 작성 여부 확인
    today = datetime.now().strftime("%Y-%m-%d")
    today_entries = [entry for entry in st.session_state.diary_entries if entry['date'] == today]
    
    if today_entries:
        st.success(f"✅ 오늘 {len(today_entries)}개 일기 작성됨")
    else:
        st.info("📝 오늘 아직 일기를 작성하지 않았어요")
    
    page = st.selectbox(
        "페이지 선택",
        ["🎙️ 오늘의 일기", "📊 감정 분석", "📈 간단 통계", "💡 개인화 피드백", "📚 일기 목록"]
    )
    
    st.markdown("---")
    
    # 통계 요약
    if st.session_state.diary_entries:
        st.markdown("### 📊 나의 통계")
        total_entries = len(st.session_state.diary_entries)
        st.metric("전체 일기 수", f"{total_entries}개")
        
        if total_entries > 0:
            latest_entry = st.session_state.diary_entries[-1]
            days_since_start = (datetime.now() - datetime.strptime(st.session_state.diary_entries[0]['date'], "%Y-%m-%d")).days + 1
            st.metric("연속 기록", f"{days_since_start}일차")
            
            # 최근 감정 상태
            recent_mood = latest_entry['analysis'].get('tone', '중립적')
            mood_emoji = {"긍정적": "😊", "중립적": "😐", "부정적": "😔"}
            st.metric("최근 기분", f"{mood_emoji.get(recent_mood, '😐')} {recent_mood}")

# 페이지별 콘텐츠
if page == "🎙️ 오늘의 일기":
    st.header("오늘 하루는 어떠셨나요?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **📝 일기 작성 가이드:**
        - 1분 이내로 자유롭게 이야기해보세요
        - 오늘 있었던 일, 느낀 감정을 솔직하게
        - 특별한 일이 없어도 괜찮아요!
        """)
    
    with col2:
        # 오늘 작성한 일기 수
        if today_entries:
            st.info(f"🎯 오늘 {len(today_entries)}번째 일기")
        else:
            st.info("🎯 오늘 첫 번째 일기")
    
    # 음성 녹음 섹션
    st.markdown("### 🎙️ 음성으로 일기 작성하기")
    
    with st.container():
        st.markdown('<div class="recording-container">', unsafe_allow_html=True)
        
        # Streamlit 내장 음성 입력 사용
        audio_value = st.audio_input(
            "🎤 버튼을 눌러 녹음을 시작하세요",
            help="마이크 권한을 허용하고 녹음 버튼을 클릭하세요"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 텍스트 입력 대안
    st.markdown("### ✏️ 또는 텍스트로 작성하기")
    text_input = st.text_area(
        "직접 입력하기",
        placeholder="오늘 하루 어떠셨나요? 자유롭게 써보세요...",
        height=120,
        help="음성 녹음이 어려우신 경우 직접 입력하실 수 있어요"
    )
    
    # 일기 저장 버튼
    if st.button("📝 일기 분석하고 저장하기", type="primary", use_container_width=True):
        diary_text = ""
        audio_data = None
        
        # 음성 데이터 처리
        if audio_value is not None:
            audio_bytes = audio_value.read()
            audio_data = base64.b64encode(audio_bytes).decode()
            st.info("🎵 음성이 저장되었습니다. 텍스트도 함께 입력해주세요.")
        
        # 텍스트 입력 처리
        if text_input.strip():
            diary_text = text_input.strip()
        
        if diary_text:
            with st.spinner("🤖 감정을 분석하는 중..."):
                analysis = analyze_emotion_simple(diary_text)
            
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
            st.success("🎉 일기가 저장되었습니다!")
            
            # 분석 결과 표시
            st.markdown("---")
            st.markdown("## 🤖 AI 분석 결과")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="emotion-card">
                    <h4>😊 감지된 감정</h4>
                    <p><strong>{', '.join(analysis['emotions'])}</strong></p>
                    <small>키워드: {', '.join(analysis.get('keywords', [])[:3])}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="emotion-card">
                    <h4>📊 상태 지수</h4>
                    <p>스트레스: <strong>{analysis['stress_level']}%</strong></p>
                    <p>활력: <strong>{analysis['energy_level']}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="emotion-card">
                    <h4>🎯 전체 평가</h4>
                    <p>기분 점수: <strong>{analysis['mood_score']}</strong></p>
                    <p>톤: <strong>{analysis.get('tone', '중립적')}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # AI 요약
            st.markdown(f"""
            <div class="feedback-box">
                <h4>🤖 AI 요약</h4>
                <p>{analysis['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ 텍스트를 입력해주세요!")

elif page == "📊 감정 분석":
    st.header("감정 분석 대시보드")
    
    if not st.session_state.diary_entries:
        st.info("📝 아직 작성된 일기가 없습니다. 첫 번째 일기를 작성해보세요!")
    else:
        # 오늘의 요약
        if today_entries:
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
        
        # 최근 일기 목록
        st.markdown(f"### 📝 최근 일기 ({len(st.session_state.diary_entries)}개)")
        
        for entry in reversed(st.session_state.diary_entries[-5:]):
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
                st.info(f"🤖 **AI 분석:** {entry['analysis']['summary']}")
                
                # 키워드
                if entry['analysis'].get('keywords'):
                    st.markdown(f"**🏷️ 키워드:** {', '.join(entry['analysis']['keywords'])}")

elif page == "📈 간단 통계":
    st.header("감정 통계")
    
    if not st.session_state.diary_entries:
        st.info("📊 일기를 작성하면 감정 변화를 확인할 수 있어요!")
    else:
        # 기간 선택
        period_days = st.selectbox("📅 분석 기간", [7, 30, 90], format_func=lambda x: f"최근 {x}일")
        
        entries_to_analyze = st.session_state.diary_entries[-period_days:]
        
        # 평균 통계
        avg_stress = sum(entry['analysis']['stress_level'] for entry in entries_to_analyze) / len(entries_to_analyze)
        avg_energy = sum(entry['analysis']['energy_level'] for entry in entries_to_analyze) / len(entries_to_analyze)
        avg_mood = sum(entry['analysis']['mood_score'] for entry in entries_to_analyze) / len(entries_to_analyze)
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("분석 기간", f"{len(entries_to_analyze)}개 일기")
        with col2:
            st.metric("평균 스트레스", f"{avg_stress:.1f}%")
        with col3:
            st.metric("평균 활력", f"{avg_energy:.1f}%")
        with col4:
            st.metric("평균 기분", f"{avg_mood:.1f}")
        
        # 감정 분포
        st.subheader("😊 감정 분포")
        
        all_emotions = []
        for entry in entries_to_analyze:
            all_emotions.extend(entry['analysis']['emotions'])
        
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 간단한 바차트 데이터 준비
        emotion_data = []
        for emotion, count in emotion_counts.items():
            emotion_data.extend([emotion] * count)
        
        if emotion_data:
            # Streamlit 내장 차트 사용
            import pandas as pd
            emotion_df = pd.DataFrame(emotion_data, columns=['감정'])
            emotion_counts_series = emotion_df['감정'].value_counts()
            
            st.bar_chart(emotion_counts_series)
            
            # 톤 분포
            st.subheader("📊 일기 톤 분포")
            tone_counts = {}
            for entry in entries_to_analyze:
                tone = entry['analysis'].get('tone', '중립적')
                tone_counts[tone] = tone_counts.get(tone, 0) + 1
            
            tone_df = pd.DataFrame(list(tone_counts.items()), columns=['톤', '개수'])
            st.bar_chart(tone_df.set_index('톤'))
        
        # 일별 변화 (간단한 라인 차트)
        st.subheader("📈 일별 감정 변화")
        
        daily_data = {}
        for entry in entries_to_analyze:
            date = entry['date']
            if date not in daily_data:
                daily_data[date] = {'stress': [], 'energy': [], 'mood': []}
            
            daily_data[date]['stress'].append(entry['analysis']['stress_level'])
            daily_data[date]['energy'].append(entry['analysis']['energy_level'])
            daily_data[date]['mood'].append(entry['analysis']['mood_score'])
        
        # 일별 평균 계산
        chart_data = []
        for date, values in sorted(daily_data.items()):
            chart_data.append({
                'date': date,
                'stress': sum(values['stress']) / len(values['stress']),
                'energy': sum(values['energy']) / len(values['energy']),
                'mood': sum(values['mood']) / len(values['mood'])
            })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            chart_df = chart_df.set_index('date')
            
            st.line_chart(chart_df[['stress', 'energy']])
            
            st.subheader("📊 기분 점수 변화")
            st.line_chart(chart_df[['mood']])

elif page == "💡 개인화 피드백":
    st.header("개인화된 피드백 & 웰빙 가이드")
    
    if not st.session_state.diary_entries:
        st.info("📝 일기를 작성하면 맞춤 피드백을 받을 수 있어요!")
    else:
        # AI 피드백
        feedback = generate_simple_feedback(st.session_state.diary_entries)
        
        st.markdown(f"""
        <div class="feedback-box">
            <h3>🤖 AI 멘탈 헬스 코치의 피드백</h3>
            <p style="font-size: 1.1em; line-height: 1.6;">{feedback}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 개인 통계
        st.subheader("📊 나의 감정 여정")
        
        recent_entries = st.session_state.diary_entries[-30:]
        if recent_entries:
            avg_stress = sum(entry['analysis']['stress_level'] for entry in recent_entries) / len(recent_entries)
            avg_energy = sum(entry['analysis']['energy_level'] for entry in recent_entries) / len(recent_entries)
            avg_mood = sum(entry['analysis']['mood_score'] for entry in recent_entries) / len(recent_entries)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #ff6b6b;">😰 평균 스트레스</h3>
                    <h2>{avg_stress:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #51cf66;">⚡ 평균 활력</h3>
                    <h2>{avg_energy:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #339af0;">😊 평균 기분</h3>
                    <h2>{avg_mood:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # 맞춤 웰빙 가이드
        st.subheader("🧘‍♀️ 맞춤 웰빙 가이드")
        
        if st.session_state.diary_entries:
            latest_entry = st.session_state.diary_entries[-1]
            stress_level = latest_entry['analysis']['stress_level']
            energy_level = latest_entry['analysis']['energy_level']
            recent_emotions = latest_entry['analysis']['emotions']
            
            # 상태에 따른 추천 활동
            if stress_level > 60:
                recommended_activity = "스트레스 해소"
                activity_description = """
                **🌊 4-7-8 호흡법**
                
                1. **4초 동안** 코로 천천히 숨 들이마시기
                2. **7초 동안** 숨 참기 (편안하게)
                3. **8초 동안** 입으로 천천히 내쉬기
                4. **3-4회 반복**하며 몸의 긴장 풀어주기
                """
                
            elif energy_level < 40:
                recommended_activity = "에너지 충전"
                activity_description = """
                **☀️ 활력 충전 명상**
                
                1. **편안한 자세**로 앉아 눈을 감으세요
                2. **따뜻한 햇살**이 몸을 감싸는 상상하기
                3. **10분간** 이 따뜻함과 에너지를 느끼기
                """
                
            elif "불안" in recent_emotions:
                recommended_activity = "불안 완화"
                activity_description = """
                **🌿 5-4-3-2-1 기법**
                
                - **5개의 것** 보기
                - **4개의 소리** 듣기  
                - **3개의 질감** 만져보기
                - **2개의 냄새** 맡기
                - **1개의 맛** 느끼기
                """
                
            else:
                recommended_activity = "감사 명상"
                activity_description = """
                **🙏 감사 일기**
                
                1. **오늘 감사한 일** 3가지 떠올리기
                2. **작은 것도 포함**하기
                3. **구체적으로** 생각해보기
                4. **마음에 새기기**
                """
            
            with st.expander(f"💡 **오늘의 추천: {recommended_activity}**", expanded=True):
                st.markdown(activity_description)
                
                if st.button(f"✅ {recommended_activity} 완료!", key="wellness_complete"):
                    st.success("🎉 훌륭해요! 자신을 위한 시간을 가져주셔서 감사합니다.")
                    st.balloons()

elif page == "📚 일기 목록":
    st.header("일기 아카이브")
    
    if not st.session_state.diary_entries:
        st.info("📝 아직 작성된 일기가 없습니다.")
    else:
        # 검색
        search_query = st.text_input("🔍 일기 내용 검색", placeholder="찾고 싶은 내용을 입력하세요")
        
        # 정렬 선택
        sort_order = st.selectbox("정렬 순서", ["최신순", "오래된순", "기분 좋은순", "기분 안 좋은순"])
        
        # 필터링
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
        elif sort_order == "기분 안 좋은순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['analysis']['mood_score'])
        
        if filtered_entries:
            st.write(f"📊 총 {len(filtered_entries)}개의 일기를 찾았습니다.")
            
            # 일기 표시
            for entry in filtered_entries:
                mood_emoji = "😊" if entry['analysis']['mood_score'] > 10 else "😐" if entry['analysis']['mood_score'] > -10 else "😔"
                
                with st.expander(f"📅 {entry['date']} {entry['time']} {mood_emoji} - {', '.join(entry['analysis']['emotions'][:2])}"):
                    st.markdown(f"**📝 내용:**\n{entry['text']}")
                    
                    if entry.get('audio_data'):
                        st.markdown("**🎵 음성 녹음:**")
                        audio_bytes = base64.b64decode(entry['audio_data'])
                        st.audio(audio_bytes)
                    
                    st.info(f"🤖 **AI 분석:** {entry['analysis']['summary']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("스트레스", f"{entry['analysis']['stress_level']}%")
                    with col2:
                        st.metric("활력", f"{entry['analysis']['energy_level']}%")
                    with col3:
                        st.metric("기분 점수", f"{entry['analysis']['mood_score']}")
                    
                    if entry['analysis'].get('keywords'):
                        st.markdown(f"**🏷️ 키워드:** {', '.join(entry['analysis']['keywords'])}")
        else:
            st.warning("검색 조건에 맞는 일기를 찾을 수 없습니다.")

# 사이드바 - 데이터 관리
with st.sidebar:
    if st.session_state.diary_entries:
        st.markdown("---")
        st.markdown("### 💾 데이터 관리")
        
        # 통계 리포트 생성
        if st.button("📊 CSV 다운로드"):
            import pandas as pd
            
            df_export = pd.DataFrame([
                {
                    'date': entry['date'],
                    'time': entry['time'],
                    'text': entry['text'],
                    'emotions': ', '.join(entry['analysis']['emotions']),
                    'stress_level': entry['analysis']['stress_level'],
                    'energy_level': entry['analysis']['energy_level'],
                    'mood_score': entry['analysis']['mood_score'],
                    'tone': entry['analysis'].get('tone', '중립적')
                }
                for entry in st.session_state.diary_entries
            ])
            
            csv = df_export.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📁 CSV 파일 다운로드",
                data=csv,
                file_name=f"voice_diary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        
        # 전체 백업
        if st.button("💾 전체 백업"):
            backup_data = {
                'entries': st.session_state.diary_entries,
                'export_date': datetime.now().isoformat(),
                'total_count': len(st.session_state.diary_entries)
            }
            backup_json = json.dumps(backup_data, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="📦 JSON 백업 다운로드",
                data=backup_json,
                file_name=f"voice_diary_backup_{datetime.now().strftime('%Y%m%d')}.json",
                mime='application/json'
            )
        
        # 데이터 초기화
        st.markdown("---")
        if st.button("🗑️ 모든 데이터 삭제"):
            if st.checkbox("⚠️ 정말로 삭제하시겠습니까?"):
                st.session_state.diary_entries = []
                st.success("✅ 모든 데이터가 삭제되었습니다.")
                st.rerun()

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🎙️ <strong>소리일기</strong> - 간소 버전</p>
    <p>Streamlit Cloud 최적화 버전으로 안정적인 서비스를 제공합니다 ✨</p>
</div>
""", unsafe_allow_html=True)
