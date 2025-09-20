import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import json
import os
import base64
import tempfile
import warnings
import calendar
import random
import re
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

# Lightweight stdlib
import numpy as np

# =============================
# Timezone Configuration
# =============================
KST = pytz.timezone('Asia/Seoul')

def get_korean_time():
    """한국 시간 반환"""
    return datetime.now(KST)

def today_key() -> str:
    """오늘 날짜 키 (YYYY-MM-DD)"""
    return get_korean_time().strftime("%Y-%m-%d")

def current_time() -> str:
    """현재 시간 (HH:MM)"""
    return get_korean_time().strftime("%H:%M")

# =============================
# Knowledge Base for RAG System
# =============================
MENTAL_HEALTH_KNOWLEDGE_BASE = {
    "mindfulness": {
        "category": "마음챙김",
        "techniques": {
            "breathing_meditation": {
                "name": "마음챙김 호흡 명상",
                "description": "현재 순간에 온전히 집중하며 판단 없이 경험을 바라보는 연습",
                "steps": [
                    "허리를 편안하게 세우고 편안한 자세로 앉아 눈을 감습니다",
                    "코로 들이마시는 숨과 입으로 내쉬는 숨의 흐름에만 집중합니다",
                    "잡념이나 감정이 떠오르면 억누르지 말고 알아차린 뒤 다시 호흡으로 주의를 돌립니다",
                    "5분 정도 연습하며, 끝나기 전에는 깊게 숨을 들이마시고 천천히 내쉬면서 마음을 정돈합니다"
                ],
                "suitable_for": ["불안", "스트레스", "긴장"],
                "duration": "5분",
                "difficulty": "초급"
            },
            "body_scan": {
                "name": "신체 이완 명상",
                "description": "머리 정수리부터 발끝까지 몸의 감각을 천천히 느끼며 긴장을 풀어주는 방법",
                "steps": [
                    "편안하게 앉거나 누워서 눈을 감습니다",
                    "머리 정수리부터 발끝까지 몸의 감각을 천천히 느끼며 긴장을 풀어줍니다",
                    "각 부위 근육에 남은 긴장을 살며시 내려놓고 이완합니다"
                ],
                "suitable_for": ["긴장", "피로", "스트레스"],
                "duration": "10-15분",
                "difficulty": "초급"
            }
        }
    },
    "stress_relief": {
        "category": "스트레스 완화",
        "techniques": {
            "deep_breathing": {
                "name": "복식호흡",
                "description": "가로막 호흡을 통해 긴장을 완화하는 기법",
                "steps": [
                    "편안한 자세로 앉아 한 손은 배 위에, 다른 손은 가슴 위에 올립니다",
                    "코로 숨을 깊게 들이쉬어 폐가 팽창하면서 배가 부풀어 오르는 것을 느낍니다",
                    "4~5초간 숨을 멈췄다가 입으로 천천히 내쉬며 배가 줄어드는 것을 느낍니다",
                    "들숨과 날숨의 길이를 1:1 비율로 맞추어 반복합니다"
                ],
                "suitable_for": ["스트레스", "불안", "긴장"],
                "duration": "5-10분",
                "difficulty": "초급"
            },
            "progressive_relaxation": {
                "name": "점진적 근육 이완법",
                "description": "몸을 16개 근육 부위로 나누어 한 부분씩 수축과 이완을 반복하는 기법",
                "steps": [
                    "몸을 16개 근육 부위로 나누어 한 부분씩 5초간 힘껏 수축시킵니다",
                    "10초 이상 이완합니다 (예: 주먹을 꽉 쥐었다가 풀고, 어깨를 으쓱했다가 내리기)",
                    "긴장과 이완의 차이를 느끼며 몸 전체를 차례로 이완합니다"
                ],
                "suitable_for": ["긴장", "스트레스", "불안"],
                "duration": "15-20분",
                "difficulty": "중급"
            },
            "4_7_8_breathing": {
                "name": "4-7-8 호흡법",
                "description": "Dr. Andrew Weil이 제안한 심호흡 기법으로 숨을 길게 내쉬는 것을 강조",
                "steps": [
                    "먼저 입을 통해 숨을 완전히 내뱉습니다",
                    "코로 4초 동안 천천히 숨을 들이마신 후 7초 동안 숨을 멈춥니다",
                    "마지막으로 8초 동안 입으로 천천히 숨을 내쉬며 복부가 완전히 꺼질 때까지 숨을 뺍니다",
                    "해당 호흡을 8회 반복하여 한 세트로 하고, 하루 2차례 연습을 권장합니다"
                ],
                "suitable_for": ["고스트레스", "불안", "과흥분"],
                "duration": "5-10분",
                "difficulty": "중급",
                "warning": "초보자는 현기증을 느낄 수 있으므로 처음에는 천천히 시도해야 합니다"
            }
        }
    },
    "positive_psychology": {
        "category": "긍정 심리학",
        "techniques": {
            "gratitude_journal": {
                "name": "감사 일기",
                "description": "감사를 표현하는 습관으로 스트레스와 우울을 줄이고 낙관주의를 증진",
                "steps": [
                    "매일 잠자리에 들기 전에 그날 감사했던 일 세 가지를 적어봅니다",
                    "차 한 잔의 여유를 즐길 수 있었던 순간 등을 떠올리며 마음속으로 감사함을 전합니다",
                    "손가락을 하나씩 따뜻하게 감싸쥐며 감사하는 마음을 보내보는 등의 간단한 감사 의식을 합니다"
                ],
                "suitable_for": ["우울", "부정적 감정", "스트레스"],
                "duration": "5-10분",
                "difficulty": "초급"
            }
        }
    },
    "sleep_recovery": {
        "category": "수면·휴식·리커버리",
        "techniques": {
            "sleep_hygiene": {
                "name": "수면 위생",
                "description": "건강한 수면을 위한 생활 습관 개선",
                "steps": [
                    "성인은 하루 7~9시간의 수면을 취합니다",
                    "매일 같은 시간에 기상하는 것이 중요합니다 (주말에도 일정한 기상 시간 유지)",
                    "취침 1시간 전에는 스마트폰과 TV 등 모든 전자기기를 끕니다",
                    "잠자리에서는 내일 할 일이나 걱정을 내려놓고, 온몸의 근육을 차례로 이완시킵니다",
                    "침실의 온도와 습도를 적절하게 유지합니다 (18~20℃에서는 습도 50%, 21~23℃에서는 습도 40%)"
                ],
                "suitable_for": ["피로", "스트레스", "불안"],
                "duration": "지속적 실천",
                "difficulty": "초급"
            },
            "pomodoro": {
                "name": "포모도로 기법",
                "description": "25분간 온전히 몰입한 후 5분간 휴식하는 주기를 반복하여 집중도를 높이는 기법",
                "steps": [
                    "25분간 온전히 몰입한 후 5분간 휴식하는 주기를 반복합니다",
                    "4회 주기가 끝나면 15~30분의 긴 휴식을 취합니다",
                    "집중 시간에는 미리 정해둔 작업 리스트를 차례로 수행합니다",
                    "휴식 시간에는 걷거나 스트레칭을 통해 신체와 마음을 가볍게 해줍니다"
                ],
                "suitable_for": ["스트레스", "과부하", "집중력 저하"],
                "duration": "2시간 사이클",
                "difficulty": "초급"
            }
        }
    }
}

def get_recommendations_from_knowledge_base(state: str, emotions: List[str], stress_level: int, energy_level: int) -> List[Dict]:
    """지식베이스에서 상황에 맞는 추천 기법들을 검색"""
    recommendations = []
    
    # 감정과 상태에 따른 매핑
    emotion_mapping = {
        "불안": ["mindfulness", "stress_relief"],
        "스트레스": ["stress_relief", "mindfulness"],
        "긴장": ["stress_relief", "mindfulness"],
        "우울": ["positive_psychology", "mindfulness"],
        "슬픔": ["positive_psychology", "mindfulness"],
        "피로": ["sleep_recovery", "stress_relief"],
        "분노": ["stress_relief", "mindfulness"],
        "짜증": ["stress_relief", "mindfulness"]
    }
    
    # 상태별 매핑
    state_mapping = {
        "고스트레스": ["stress_relief"],
        "긴장 과다": ["stress_relief", "mindfulness"],
        "과흥분/과부하 가능": ["stress_relief", "sleep_recovery"],
        "저활력": ["positive_psychology", "sleep_recovery"],
        "저각성": ["positive_psychology", "sleep_recovery"]
    }
    
    # 추천 카테고리 결정
    recommended_categories = set()
    
    # 감정 기반 추천
    for emotion in emotions:
        if emotion in emotion_mapping:
            recommended_categories.update(emotion_mapping[emotion])
    
    # 상태 기반 추천
    if state in state_mapping:
        recommended_categories.update(state_mapping[state])
    
    # 스트레스 수준 기반 추천
    if stress_level > 70:
        recommended_categories.add("stress_relief")
    elif stress_level > 50:
        recommended_categories.add("mindfulness")
    
    # 에너지 수준 기반 추천
    if energy_level < 30:
        recommended_categories.add("sleep_recovery")
    elif energy_level < 50:
        recommended_categories.add("positive_psychology")
    
    # 기본 추천 (아무것도 매칭되지 않을 때)
    if not recommended_categories:
        recommended_categories.add("mindfulness")
    
    # 추천 기법들 수집
    for category in recommended_categories:
        if category in MENTAL_HEALTH_KNOWLEDGE_BASE:
            category_data = MENTAL_HEALTH_KNOWLEDGE_BASE[category]
            for technique_key, technique in category_data["techniques"].items():
                # 적합성 검사
                is_suitable = False
                for emotion in emotions:
                    if emotion in technique.get("suitable_for", []):
                        is_suitable = True
                        break
                
                if state in technique.get("suitable_for", []) or is_suitable:
                    recommendations.append({
                        "category": category_data["category"],
                        "name": technique["name"],
                        "description": technique["description"],
                        "steps": technique["steps"],
                        "duration": technique["duration"],
                        "difficulty": technique["difficulty"],
                        "warning": technique.get("warning", None)
                    })
    
    # 중복 제거 및 최대 3개로 제한
    unique_recommendations = []
    seen_names = set()
    for rec in recommendations:
        if rec["name"] not in seen_names:
            unique_recommendations.append(rec)
            seen_names.add(rec["name"])
            if len(unique_recommendations) >= 3:
                break
    
    return unique_recommendations

# =============================
# Page / App Config
# =============================
st.set_page_config(
    page_title="소리로 쓰는 하루 - AI 감정 코치",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Lazy Import Helpers (cached)
# =============================
@st.cache_resource(show_spinner=False)
def get_librosa():
    try:
        import librosa  # type: ignore
        return librosa
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_parselmouth():
    try:
        import parselmouth  # type: ignore
        from parselmouth.praat import call  # noqa: F401
        return parselmouth
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_openai_client():
    try:
        import openai  # type: ignore
        if "OPENAI_API_KEY" in st.secrets:
            return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # type: ignore
        elif "openai_api_key" in st.session_state and st.session_state.openai_api_key:
            return openai.OpenAI(api_key=st.session_state.openai_api_key)  # type: ignore
        else:
            return None
    except Exception:
        return None

openai_client = get_openai_client()

# =============================
# Session State Initialization
# =============================
def initialize_session_state():
    """세션 상태 초기화"""
    if "diary_entries" not in st.session_state:
        st.session_state.diary_entries: List[Dict] = []
    if "prosody_baseline" not in st.session_state:
        st.session_state.prosody_baseline: Dict[str, float] = {}
    if "user_goals" not in st.session_state:
        st.session_state.user_goals: List[Dict] = []
    if "show_disclaimer" not in st.session_state:
        st.session_state.show_disclaimer = True
    if "onboarding_completed" not in st.session_state:
        st.session_state.onboarding_completed = False
    if "demo_data_loaded" not in st.session_state:
        st.session_state.demo_data_loaded = False

initialize_session_state()

# =============================
# Demo Data Generation
# =============================
def generate_demo_data():
    """20대 남성 대학생 페르소나 기반 가상 데이터 생성"""
    if st.session_state.demo_data_loaded:
        return
    
    demo_scenarios = [
        {
            "text": "오늘은 중간고사 마지막 날이었어요. 수학 시험이 정말 어려웠는데 그래도 나름 준비한 만큼은 쓴 것 같아요. 시험 끝나고 친구들이랑 치킨 먹으면서 스트레스 풀었는데, 역시 친구들과 함께 있으니까 기분이 좋아지네요. 내일부터는 좀 여유롭게 지낼 수 있을 것 같아요.",
            "emotions": ["기쁨", "평온"],
            "stress": 35, "energy": 70, "mood": 25, "tone": "긍정적"
        },
        {
            "text": "요즘 좋아하는 친구한테 고백할까 말까 고민이 너무 많아요. 같이 스터디도 하고 밥도 먹는데 나한테 관심이 있는 건지 잘 모르겠어요. 오늘도 같이 도서관에서 공부했는데 자꾸 의식하게 되서 집중이 안 되더라구요. 용기를 내야 하는데 거절당할까봐 무섭기도 하고...",
            "emotions": ["불안", "설렘"],
            "stress": 65, "energy": 50, "mood": -5, "tone": "중립적"
        },
        {
            "text": "과제 마감이 내일인데 아직 반도 못했어요. 교수님이 까다로우시기로 유명한데 대충 낼 수도 없고... 밤새워야겠어요. 커피 마시고 각성제처럼 쓰면서 버텨야겠는데, 요즘 계속 이런 식으로 사니까 컨디션이 너무 안 좋아요. 규칙적으로 살고 싶은데 현실적으로 어렵네요.",
            "emotions": ["스트레스", "피로"],
            "stress": 85, "energy": 25, "mood": -35, "tone": "부정적"
        },
        {
            "text": "드디어 좋아하는 친구한테 고백했어요! 생각보다 담담하게 받아줘서 다행이었고, 연인까지는 아니어도 더 가까워질 수 있을 것 같다고 하더라구요. 완전히 성공은 아니지만 그래도 마음이 후련해요. 친구들도 용기냈다고 칭찬해줘서 기분이 좋아요. 오늘은 정말 뜻깊은 하루였어요.",
            "emotions": ["기쁨", "만족"],
            "stress": 40, "energy": 75, "mood": 30, "tone": "긍정적"
        },
        {
            "text": "오늘은 정말 평범한 하루였어요. 수업 듣고, 도서관에서 공부하고, 집에 와서 넷플릭스 보고... 특별할 건 없었지만 그래도 평온한 게 좋기도 해요. 요즘 너무 바빴거든요. 이렇게 여유있는 시간이 있다는 게 감사하네요. 내일은 친구들이랑 볼링 치러 가기로 했어요.",
            "emotions": ["평온", "만족"],
            "stress": 20, "energy": 60, "mood": 15, "tone": "긍정적"
        },
        {
            "text": "팀 프로젝트에서 조원 한 명이 갑자기 연락두절되었어요. 발표가 다음 주인데 정말 당황스러워요. 남은 조원들끼리 그 친구 몫까지 해야 하는 상황이 되었는데 시간은 부족하고... 화가 나면서도 어쩔 수 없이 해야 하니까 스트레스가 이만저만이 아니에요. 이런 상황이 정말 힘들어요.",
            "emotions": ["분노", "스트레스"],
            "stress": 90, "energy": 40, "mood": -40, "tone": "부정적"
        },
        {
            "text": "친구들이랑 MT 다녀왔어요! 정말 오랜만에 스트레스 완전히 날려버리고 왔네요. 게임도 하고 바베큐도 먹고, 밤새 이야기도 하고... 역시 친구들과 함께하는 시간이 최고예요. 좋아하는 친구도 같이 가서 더 가까워진 것 같고요. 내일부터 다시 현실이지만 오늘만큼은 정말 행복했어요.",
            "emotions": ["기쁨", "행복"],
            "stress": 15, "energy": 85, "mood": 45, "tone": "긍정적"
        }
    ]
    
    # 지난 7일간 데이터 생성
    base_date = get_korean_time() - timedelta(days=6)
    
    for i, scenario in enumerate(demo_scenarios):
        entry_date = base_date + timedelta(days=i)
        entry = {
            "id": i + 1,
            "date": entry_date.strftime("%Y-%m-%d"),
            "time": f"{random.randint(18, 22):02d}:{random.randint(0, 59):02d}",
            "text": scenario["text"],
            "analysis": {
                "emotions": scenario["emotions"],
                "stress_level": scenario["stress"],
                "energy_level": scenario["energy"],
                "mood_score": scenario["mood"],
                "summary": f"{scenario['tone']} 상태의 하루를 보냈습니다.",
                "keywords": [],
                "tone": scenario["tone"],
                "confidence": round(random.uniform(0.7, 0.9), 2)
            },
            "audio_data": None,
            "mental_state": {
                "state": "안정/회복" if scenario["stress"] < 40 else ("고스트레스" if scenario["stress"] > 70 else "중립"),
                "summary": f"스트레스 {scenario['stress']}%, 에너지 {scenario['energy']}% 상태입니다.",
                "positives": ["친구들과의 시간", "성취감"] if scenario["tone"] == "긍정적" else [],
                "recommendations": [
                    "충분한 휴식 취하기",
                    "친구들과 시간 보내기",
                    "규칙적인 생활 패턴 유지하기"
                ],
                "motivation": "대학생활도 인생의 소중한 시간이에요. 하루하루 최선을 다하세요!"
            }
        }
        st.session_state.diary_entries.append(entry)
    
    # 목표 데이터 추가
    st.session_state.user_goals = [
        {
            "id": 1,
            "type": "stress",
            "target": 50,
            "description": "스트레스 지수를 50 이하로 유지하기",
            "created_date": (get_korean_time() - timedelta(days=5)).strftime("%Y-%m-%d"),
            "active": True
        },
        {
            "id": 2,
            "type": "consistency",
            "target": 5,
            "description": "일주일에 5번 이상 기록하기",
            "created_date": (get_korean_time() - timedelta(days=5)).strftime("%Y-%m-%d"),
            "active": True
        }
    ]
    
    st.session_state.demo_data_loaded = True

# =============================
# Styles
# =============================
st.markdown(
    """
    <style>
      .main-header{ 
        text-align:center; 
        padding:1.5rem; 
        background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); 
        color:#fff; 
        border-radius:15px; 
        margin-bottom:20px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
      }
      .card{ 
        background:#fff; 
        border:1px solid #e0e6ed; 
        border-left:4px solid #667eea; 
        border-radius:12px; 
        padding:1.2rem; 
        box-shadow:0 2px 10px rgba(0,0,0,0.05); 
        margin-bottom:1rem;
      }
      .success-card{ 
        background:linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
        border-left:4px solid #28a745; 
      }
      .warning-card{ 
        background:linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
        border-left:4px solid #ffc107; 
      }
      .technique-card{ 
        background:linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%); 
        border-left:4px solid #17a2b8; 
        margin: 1rem 0;
        padding: 1.2rem;
        border-radius: 12px;
      }
      .calendar-day{
        text-align: center;
        padding: 8px;
        margin: 2px;
        border-radius: 8px;
        min-height: 40px;
        cursor: pointer;
      }
      .calendar-today{
        border: 2px solid #667eea;
        font-weight: bold;
      }
      .disclaimer-banner{
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
      }
      .goal-progress{
        background: linear-gradient(90deg, #4caf50 0%, #81c784 100%);
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
      }
      .note{ color:#666; font-size:0.85rem; font-style: italic; }
      .metric-positive { color: #28a745; font-weight: bold; }
      .metric-negative { color: #dc3545; font-weight: bold; }
      .metric-neutral { color: #6c757d; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Disclaimer and Safety Notice
# =============================
def show_disclaimer():
    """서비스 안내 및 고지사항"""
    if st.session_state.show_disclaimer:
        st.markdown(
            """
            <div class="disclaimer-banner">
                <h4>🛡️ 서비스 이용 안내</h4>
                <ul>
                    <li><strong>의료적 한계:</strong> 본 서비스는 자기 성찰을 돕는 보조 도구이며, 의료적 진단이나 치료를 대체하지 않습니다.</li>
                    <li><strong>데이터 보안:</strong> 모든 기록은 세션 내에서만 저장되며, 브라우저 종료 시 삭제됩니다.</li>
                    <li><strong>AI 한계:</strong> AI 분석 결과는 참고용이며, 개인의 판단이 우선됩니다.</li>
                    <li><strong>긴급상황:</strong> 심각한 정신건강 문제는 전문가와 상담하시기 바랍니다.</li>
                    <li><strong>추천 기법:</strong> 제공되는 심리 건강 기법들은 의료기관 검증 자료 기반이나, 개인차가 있을 수 있습니다.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 이해했습니다", type="primary"):
                st.session_state.show_disclaimer = False
                st.rerun()
        
        with col2:
            if st.button("📊 데모 데이터로 시작하기"):
                generate_demo_data()
                st.session_state.show_disclaimer = False
                st.success("20대 대학생 페르소나의 가상 데이터가 로드되었습니다!")
                st.rerun()

# =============================
# Header
# =============================
if not st.session_state.show_disclaimer:
    st.markdown(
        f"""
        <div class="main-header">
          <h1>🎙️ 소리로 쓰는 하루 – AI 감정 코치</h1>
          <p>📅 {get_korean_time().strftime('%Y년 %m월 %d일 %A')} | ⏰ {current_time()}</p>
          <p>감정 라벨은 텍스트 분석 기반, 목소리는 <b>보조 지표</b>로 활용합니다</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

show_disclaimer()

# =============================
# Feature Extraction Classes
# =============================
class VoiceFeatureExtractor:
    """Prosody feature extractor with graceful fallbacks."""

    def __init__(self, target_sr: int = 22050):
        self.sample_rate = target_sr

    def _load_audio(self, audio_bytes: bytes):
        librosa = get_librosa()
        if not librosa:
            return None, None
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            y, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True, res_type="kaiser_fast")
            return y, sr
        except Exception:
            return None, None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def extract(self, audio_bytes: bytes) -> Dict:
        librosa = get_librosa()
        if not librosa:
            return self._default_features()
        try:
            y, sr = self._load_audio(audio_bytes)
            if y is None or y.size == 0 or sr is None:
                return self._default_features()

            duration_sec = max(0.001, float(len(y) / sr))
            
            # RMS energy
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            energy_mean = float(np.mean(rms))
            energy_max = float(np.max(rms))
            
            # Tempo (guard for short audio)
            if duration_sec >= 2.5:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            else:
                tempo = 110.0
                
            # ZCR
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)[0]
            zcr_mean = float(np.mean(zcr))
            
            # Spectral centroid
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = float(np.mean(sc))
            
            # Pitch proxy
            pitch_mean = 150.0
            pitch_var = 0.13
            try:
                if duration_sec >= 1.0:
                    pitches, mags = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=400)
                    valid = []
                    for t in range(pitches.shape[1]):
                        idx = int(np.argmax(mags[:, t]))
                        p = float(pitches[idx, t])
                        if p > 0:
                            valid.append(p)
                    if valid:
                        va = np.array(valid, dtype=float)
                        pitch_mean = float(np.mean(va))
                        pitch_var = float(np.std(va) / (np.mean(va) + 1e-6))
            except Exception:
                pass

            # HNR / Jitter via parselmouth (optional)
            hnr = 15.0
            jitter = 0.012
            parselmouth = get_parselmouth()
            if parselmouth:
                tmp2 = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t2:
                        t2.write(audio_bytes)
                        tmp2 = t2.name
                    snd = parselmouth.Sound(tmp2)
                    harm = snd.to_harmonicity_cc()
                    hnr = float(np.nan_to_num(harm.values.mean(), nan=15.0))
                    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
                    jitter = float(parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
                except Exception:
                    pass
                finally:
                    if tmp2 and os.path.exists(tmp2):
                        try:
                            os.unlink(tmp2)
                        except Exception:
                            pass

            return {
                "duration_sec": duration_sec,
                "pitch_mean": pitch_mean,
                "pitch_variation": pitch_var,
                "energy_mean": energy_mean,
                "energy_max": energy_max,
                "tempo": float(tempo),
                "zcr_mean": zcr_mean,
                "spectral_centroid_mean": spectral_centroid_mean,
                "hnr": hnr,
                "jitter": jitter,
            }
        except Exception:
            return self._default_features()

    def _default_features(self) -> Dict:
        return {
            "duration_sec": 0.0,
            "pitch_mean": 150.0,
            "pitch_variation": 0.13,
            "energy_mean": 0.08,
            "energy_max": 0.12,
            "tempo": 110.0,
            "zcr_mean": 0.10,
            "spectral_centroid_mean": 2000.0,
            "hnr": 15.0,
            "jitter": 0.012,
        }

# =============================
# Voice Analysis Functions
# =============================
def prosody_to_dimensions(f: Dict, baseline: Optional[Dict] = None) -> Dict:
    def norm(key: str, val: float) -> float:
        if not baseline or key not in baseline:
            return val
        b = float(baseline.get(key, 0.0))
        return (val / b) if b else val

    tempo = norm("tempo", float(f.get("tempo", 110.0)))
    energy = norm("energy_mean", float(f.get("energy_mean", 0.08)))
    hnr = norm("hnr", float(f.get("hnr", 15.0)))
    jitter = float(f.get("jitter", 0.012))
    zcr = float(f.get("zcr_mean", 0.10))
    sc = norm("spectral_centroid_mean", float(f.get("spectral_centroid_mean", 2000.0)))

    arousal = float(np.clip(35 + 120 * energy + 0.06 * (tempo - 110) + 0.004 * (sc - 2000), 0, 100))
    tension = float(np.clip(28 + 120 * jitter + 0.55 * (zcr - 0.10) * 100, 0, 100))
    stability = float(np.clip(60 + 1.3 * (hnr - 15) - 85 * jitter, 0, 100))

    duration = float(f.get("duration_sec", 4.0))
    quality = float(np.clip(
        0.28 * (duration / 8.0) + 0.42 * np.clip((hnr - 10) / 15, 0, 1) + 0.30 * np.clip((energy - 0.06) / 0.20, 0, 1),
        0, 1
    ))

    return {"arousal": arousal, "tension": tension, "stability": stability, "quality": quality}

def analyze_voice_as_cues(voice_features: Dict, baseline: Optional[Dict] = None) -> Dict:
    dims = prosody_to_dimensions(voice_features, baseline)
    return {"voice_cues": dims, "voice_features": voice_features}

# =============================
# Text Analysis Functions
# =============================
def safe_json_parse(content: str) -> Dict:
    """안전한 JSON 파싱 with 다양한 포맷 지원"""
    if not content or not content.strip():
        return {}
    
    content = content.strip()
    
    # 코드블록 제거
    if content.startswith("```"):
        lines = content.split('\n')
        content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
        
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 여러 JSON 객체가 있는 경우 첫 번째 시도
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        
        # 마지막으로 기본값 반환
        return {}

def analyze_text_with_llm(text: str, voice_cues_for_prompt: Optional[Dict] = None) -> Dict:
    if not openai_client:
        return analyze_text_simulation(text)
    
    cues_text = ""
    if voice_cues_for_prompt:
        cues = voice_cues_for_prompt
        cues_text = (
            f"\n(보조지표) 각성:{int(cues.get('arousal',0))}, 긴장:{int(cues.get('tension',0))}, "
            f"안정:{int(cues.get('stability',0))}, 품질:{cues.get('quality',0):.2f}"
        )
    
    try:
        sys_msg = (
            "감정 라벨(기쁨/슬픔/분노/불안/평온/중립)은 텍스트로만 판단하세요. "
            "음성 지표는 수치 보조로만 참고하고 라벨을 바꾸지 마세요. "
            "다음 JSON 형식으로만 응답하세요: "
            '{"emotions": ["감정1", "감정2"], "stress_level": 숫자, "energy_level": 숫자, "mood_score": 숫자, "summary": "요약", "keywords": ["키워드"], "tone": "톤", "confidence": 숫자}'
        )
        
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=500,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"오늘의 이야기: {text}{cues_text}"}
            ],
        )
        
        content = resp.choices[0].message.content
        if not content:
            return analyze_text_simulation(text)
            
        data = safe_json_parse(content)
        
        if not data:
            return analyze_text_simulation(text)
            
        # 필수 필드 보장
        data.setdefault("emotions", ["중립"])  
        data.setdefault("stress_level", 30)
        data.setdefault("energy_level", 50)
        data.setdefault("mood_score", 0)
        data.setdefault("summary", "일반적인 상태입니다.")
        data.setdefault("keywords", [])
        data.setdefault("tone", "중립적")
        data.setdefault("confidence", 0.7)
        
        return data
        
    except Exception as e:
        print(f"LLM 분석 오류: {e}")
        return analyze_text_simulation(text)

def analyze_text_simulation(text: str) -> Dict:
    """LLM이 없을 때 사용하는 시뮬레이션 분석"""
    t = text.lower()
    pos_kw = ["좋", "행복", "뿌듯", "기쁨", "즐겁", "평온", "만족", "감사", "성공", "좋아"]
    neg_kw = ["힘들", "불안", "걱정", "짜증", "화", "우울", "슬픔", "스트레스", "피곤", "어려"]
    
    pos = sum(k in t for k in pos_kw)
    neg = sum(k in t for k in neg_kw)
    
    if pos > neg:
        tone = "긍정적"
        stress = max(10, 40 - 8 * pos)
        energy = min(85, 50 + 10 * pos)
        emos = ["기쁨"]
    elif neg > pos:
        tone = "부정적"
        stress = min(85, 40 + 10 * neg)
        energy = max(20, 55 - 8 * neg)
        emos = ["슬픔"]
    else:
        tone = "중립적"
        stress = 30
        energy = 50
        emos = ["중립"]
    
    mood = int(np.clip(energy - stress, -70, 70))
    
    return {
        "emotions": emos,
        "stress_level": int(stress),
        "energy_level": int(energy),
        "mood_score": int(mood),
        "summary": f"{tone} 상태로 보입니다.",
        "keywords": [],
        "tone": tone,
        "confidence": 0.55,
    }

# =============================
# Fusion Functions
# =============================
def combine_text_and_voice(text_analysis: Dict, voice_analysis: Optional[Dict]) -> Dict:
    """텍스트 분석(주)과 음성 분석(보조) 결합"""
    if not voice_analysis or "voice_cues" not in voice_analysis:
        return text_analysis

    cues = voice_analysis["voice_cues"]
    quality = float(cues.get("quality", 0.5))
    base_alpha = 0.25 * quality
    
    tone = text_analysis.get("tone", "중립적")
    if tone == "긍정적":
        base_alpha *= 0.6  # 긍정적일 때는 음성 영향 감소
    elif tone == "부정적":
        base_alpha *= 0.9  # 부정적일 때는 음성 영향 증가

    MAX_DS, MAX_DE, MAX_DM = 12, 12, 10

    stress = text_analysis.get("stress_level", 30)
    energy = text_analysis.get("energy_level", 50)
    mood = text_analysis.get("mood_score", 0)

    # 음성 신호로부터 조정값 계산
    delta_energy = base_alpha * ((cues["arousal"] - 50) / 50.0) * 12
    delta_stress = base_alpha * (((cues["tension"] - 50) / 50.0) * 12 - ((cues["stability"] - 50) / 50.0) * 6)
    delta_mood = base_alpha * ((cues["stability"] - 50) / 50.0) * 8 - base_alpha * ((cues["tension"] - 50) / 50.0) * 6

    # 조정값 범위 제한
    delta_stress = float(np.clip(delta_stress, -MAX_DS, MAX_DS))
    delta_energy = float(np.clip(delta_energy, -MAX_DE, MAX_DE))
    delta_mood = float(np.clip(delta_mood, -MAX_DM, MAX_DM))

    # 최종 결과 생성
    combined = dict(text_analysis)
    combined["stress_level"] = int(np.clip(stress + delta_stress, 0, 100))
    combined["energy_level"] = int(np.clip(energy + delta_energy, 0, 100))
    combined["mood_score"] = int(np.clip(mood + delta_mood, -70, 70))
    combined["confidence"] = float(np.clip(text_analysis.get("confidence", 0.7) + 0.12 * quality, 0, 1))
    combined["voice_analysis"] = voice_analysis

    return combined

# =============================
# Transcription Functions
# =============================
def transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    """Whisper를 사용한 음성 전사"""
    if not openai_client:
        return None
    try:
        tmp = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t:
            t.write(audio_bytes)
            tmp = t.name
        
        with open(tmp, "rb") as fh:
            out = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=fh, 
                language="ko"
            )
        
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        
        return out.text
    except Exception as e:
        print(f"음성 전사 오류: {e}")
        return None

# =============================
# Baseline Update Functions
# =============================
def update_baseline(vf: Dict):
    """베이스라인 업데이트 (이동평균)"""
    keys = ["pitch_mean", "tempo", "energy_mean", "hnr", "spectral_centroid_mean"]
    b = st.session_state.prosody_baseline
    count = int(b.get("_count", 0))
    new_count = min(20, count + 1)
    alpha = 1.0 / new_count
    
    for k in keys:
        v = float(vf.get(k, 0.0))
        prev = float(b.get(k, v))
        b[k] = (1 - alpha) * prev + alpha * v
    
    b["_count"] = new_count

# =============================
# Enhanced Coaching Functions with RAG
# =============================
def generate_enhanced_coach_report(text: str, combined: Dict, recent: Optional[List[Dict]] = None) -> Dict:
    """RAG 시스템을 활용한 향상된 코칭 리포트 생성"""
    if not openai_client:
        return assess_mental_state_with_rag(text, combined)

    # 1차 분석 결과에서 상태 정보 추출
    emotions = combined.get("emotions", [])
    stress_level = combined.get("stress_level", 30)
    energy_level = combined.get("energy_level", 50)
    mood_score = combined.get("mood_score", 0)
    
    # 상태 판정
    state = determine_mental_state(stress_level, energy_level, mood_score, emotions)
    
    # RAG: 지식베이스에서 추천 기법들 검색
    rag_recommendations = get_recommendations_from_knowledge_base(
        state, emotions, stress_level, energy_level
    )
    
    # 2차 LLM: RAG 정보를 활용한 개인화된 추천 생성
    try:
        cues = combined.get("voice_analysis", {}).get("voice_cues", {})
        
        # 최근 기록 요약
        history_blob: List[Dict] = []
        if recent:
            for e in recent[-5:]:
                a = e.get("analysis", {})
                history_blob.append({
                    "date": e.get("date"),
                    "tone": a.get("tone"),
                    "stress": a.get("stress_level"),
                    "energy": a.get("energy_level"),
                    "mood": a.get("mood_score")
                })

        sys_msg = (
            "당신은 전문적인 한국어 심리 코치입니다. "
            "제공된 검증된 심리 건강 기법들을 바탕으로 개인화된 추천을 생성하세요. "
            "의료적 진단이 아닌 자기 돌봄 차원의 조언을 제공하세요. "
            "다음 JSON 형식으로만 답하세요: "
            '{"state": "상태", "summary": "요약", "positives": ["긍정요소"], "recommendations": ["추천사항"], "motivation": "격려메시지", "wellness_techniques": [{"name": "기법명", "description": "설명", "priority": "높음/보통/낮음"}]}'
        )
        
        user_payload = {
            "text": text,
            "analysis": {
                "emotions": emotions,
                "stress": stress_level,
                "energy": energy_level,
                "mood": mood_score,
                "tone": combined.get("tone", "중립적"),
            },
            "voice_cues": {
                "arousal": int(cues.get("arousal", 50)),
                "tension": int(cues.get("tension", 50)),
                "stability": int(cues.get("stability", 50)),
                "quality": float(cues.get("quality", 0.5)),
            },
            "available_techniques": rag_recommendations,
            "recent_summary": history_blob,
        }

        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=800,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
        )

        content = resp.choices[0].message.content
        if not content:
            return assess_mental_state_with_rag(text, combined, rag_recommendations)
        
        data = safe_json_parse(content)
        
        if not data:
            return assess_mental_state_with_rag(text, combined, rag_recommendations)

        # 필수 필드 보장
        data.setdefault("state", state)
        data.setdefault("summary", "오늘의 상태를 차분히 정리했어요.")
        data.setdefault("positives", [])
        data.setdefault("recommendations", [])
        data.setdefault("motivation", "작은 걸음이 큰 변화를 만듭니다.")
        data.setdefault("wellness_techniques", rag_recommendations[:2])  # 상위 2개 기법
        
        # 길이 제한
        data["recommendations"] = data.get("recommendations", [])[:4]
        data["positives"] = data.get("positives", [])[:4]
        
        return data
        
    except Exception as e:
        print(f"향상된 코칭 리포트 생성 오류: {e}")
        return assess_mental_state_with_rag(text, combined, rag_recommendations)

def determine_mental_state(stress_level: int, energy_level: int, mood_score: int, emotions: List[str]) -> str:
    """정신 상태 판정"""
    if stress_level >= 80:
        return "고스트레스"
    elif stress_level >= 60:
        return "긴장 과다"
    elif energy_level < 30:
        return "저활력"
    elif energy_level < 40 and mood_score < -10:
        return "저각성"
    elif stress_level > 60 and energy_level > 70:
        return "과흥분/과부하 가능"
    elif mood_score >= 20 and stress_level < 40:
        return "안정/회복"
    else:
        return "중립"

def assess_mental_state_with_rag(text: str, combined: Dict, rag_recommendations: Optional[List[Dict]] = None) -> Dict:
    """RAG 기반 정신 상태 평가 (폴백)"""
    tone = combined.get("tone", "중립적")
    stress = combined.get("stress_level", 30)
    energy = combined.get("energy_level", 50)
    mood = combined.get("mood_score", 0)
    emotions = combined.get("emotions", [])
    
    state = determine_mental_state(stress, energy, mood, emotions)
    
    if not rag_recommendations:
        rag_recommendations = get_recommendations_from_knowledge_base(state, emotions, stress, energy)
    
    # 긍정적 요소 추출
    positives = extract_positive_events(text)
    
    # 기본 추천사항
    recs: List[str] = []
    if tone == "긍정적" or positives:
        recs.append("오늘의 긍정적인 경험을 감사 일기에 기록해보세요.")
    
    # RAG 기법들을 추천사항으로 변환
    for technique in rag_recommendations[:2]:
        recs.append(f"{technique['name']}: {technique['description']}")
    
    # 동기부여 메시지
    mot = "작은 습관이 오늘의 좋은 흐름을 내일로 이어줍니다."
    if state in ("고스트레스", "긴장 과다"):
        mot = "호흡을 고르고, 천천히. 당신의 속도로 충분합니다."
    elif state in ("저활력", "저각성"):
        mot = "작은 한 걸음이 에너지를 깨웁니다. 10분만 움직여볼까요?"

    summary = f"상태: {state} · 스트레스 {stress} · 에너지 {energy}"

    return {
        "state": state,
        "summary": summary,
        "positives": positives,
        "recommendations": recs[:4],
        "motivation": mot,
        "wellness_techniques": rag_recommendations[:2]
    }

def extract_positive_events(text: str) -> List[str]:
    """텍스트에서 긍정적 이벤트 추출"""
    t = text.lower()
    keys = [
        ("좋았", "오늘 좋았던 점"),
        ("행복", "행복한 순간"),
        ("고마", "감사한 일"),
        ("즐겁", "즐거웠던 활동"),
        ("평온", "평온했던 순간"),
        ("성공", "성취"),
        ("뿌듯", "뿌듯했던 일"),
        ("만족", "만족스러운 일"),
        ("친구", "친구들과의 시간")
    ]
    
    tags: List[str] = []
    for k, v in keys:
        if k in t:
            tags.append(v)
    
    return list(dict.fromkeys(tags))[:4]

# =============================
# Weekly Report Functions
# =============================
def generate_weekly_report(entries: List[Dict]) -> Dict:
    """주간 리포트 생성"""
    if not openai_client or len(entries) < 7:
        return generate_simple_weekly_report(entries)

    try:
        # 데이터 요약
        week_data = []
        for entry in entries[-7:]:
            analysis = entry.get("analysis", {})
            week_data.append({
                "date": entry.get("date"),
                "emotions": analysis.get("emotions", []),
                "stress": analysis.get("stress_level", 0),
                "energy": analysis.get("energy_level", 0),
                "mood": analysis.get("mood_score", 0),
                "tone": analysis.get("tone", "중립적"),
                "text_summary": entry.get("text", "")[:100] + "..." if len(entry.get("text", "")) > 100 else entry.get("text", "")
            })

        sys_msg = (
            "당신은 전문적인 웰빙 코치입니다. 지난 7일간의 감정 기록을 분석하여 개인화된 주간 리포트를 작성해주세요. "
            "다음 JSON 형식으로만 답하세요: "
            '{"overall_trend": "추세", "key_insights": ["인사이트"], "patterns": {"best_days": ["날짜"], "challenging_days": ["날짜"], "emotional_patterns": "패턴설명"}, "recommendations": {"priority_actions": ["행동"], "wellness_tips": ["팁"], "goals_for_next_week": ["목표"]}, "encouragement": "격려메시지"}'
        )

        user_content = f"지난 7일간의 데이터: {json.dumps(week_data, ensure_ascii=False)}"

        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=800,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content}
            ]
        )

        content = resp.choices[0].message.content
        if not content:
            return generate_simple_weekly_report(entries)
        
        data = safe_json_parse(content)
        
        if not data:
            return generate_simple_weekly_report(entries)
            
        return data
        
    except Exception as e:
        print(f"주간 리포트 생성 오류: {e}")
        return generate_simple_weekly_report(entries)

def generate_simple_weekly_report(entries: List[Dict]) -> Dict:
    """간단한 주간 리포트 (폴백)"""
    if len(entries) < 3:
        return {
            "overall_trend": "안정적",
            "key_insights": ["아직 분석하기에 충분한 데이터가 없습니다."],
            "patterns": {
                "best_days": [],
                "challenging_days": [],
                "emotional_patterns": "더 많은 기록이 필요합니다."
            },
            "recommendations": {
                "priority_actions": ["꾸준한 기록 유지하기"],
                "wellness_tips": ["하루 10분 자기 성찰 시간 갖기"],
                "goals_for_next_week": ["매일 감정 기록하기"]
            },
            "encouragement": "좋은 시작입니다! 꾸준히 기록해보세요."
        }

    # 최근 7일 데이터
    recent = entries[-7:]
    
    # 평균 계산
    avg_stress = np.mean([e.get("analysis", {}).get("stress_level", 0) for e in recent])
    avg_energy = np.mean([e.get("analysis", {}).get("energy_level", 0) for e in recent])
    avg_mood = np.mean([e.get("analysis", {}).get("mood_score", 0) for e in recent])

    # 트렌드 분석
    if avg_stress < 40 and avg_energy > 60:
        trend = "개선됨"
    elif avg_stress > 70 or avg_energy < 30:
        trend = "주의필요"
    else:
        trend = "안정적"

    # 최고/최악의 날
    best_day = max(recent, key=lambda x: x.get("analysis", {}).get("mood_score", 0))
    worst_day = min(recent, key=lambda x: x.get("analysis", {}).get("mood_score", 0))

    return {
        "overall_trend": trend,
        "key_insights": [
            f"평균 스트레스: {avg_stress:.0f}점",
            f"평균 에너지: {avg_energy:.0f}점",
            f"평균 기분: {avg_mood:.0f}점"
        ],
        "patterns": {
            "best_days": [best_day.get("date", "")],
            "challenging_days": [worst_day.get("date", "")],
            "emotional_patterns": f"이번 주는 전반적으로 {trend} 상태를 보였습니다."
        },
        "recommendations": {
            "priority_actions": ["스트레스 관리에 집중", "규칙적인 운동"] if avg_stress > 50 else ["현재 상태 유지"],
            "wellness_tips": ["명상 5분", "자연 산책", "충분한 수면"],
            "goals_for_next_week": ["스트레스 지수 40 이하 유지", "매일 감정 기록"]
        },
        "encouragement": "매일 기록하고 계시는 노력이 대단합니다!"
    }

# =============================
# Calendar Functions
# =============================
def get_emotion_color(emotions: List[str]) -> str:
    """감정에 따른 색상 반환"""
    if not emotions:
        return "#f8f9fa"
    
    primary_emotion = emotions[0].lower()
    color_map = {
        "기쁨": "#28a745",      # 초록
        "행복": "#28a745",
        "평온": "#17a2b8",      # 청록
        "만족": "#6f42c1",      # 보라
        "슬픔": "#6c757d",      # 회색
        "불안": "#ffc107",      # 노랑
        "걱정": "#ffc107",
        "분노": "#dc3545",      # 빨강
        "짜증": "#fd7e14",      # 주황
        "스트레스": "#dc3545",  # 빨강
        "피로": "#6c757d",      # 회색
        "설렘": "#e83e8c",      # 핑크
        "중립": "#e9ecef",      # 연회색
    }
    
    for emotion in emotions:
        if emotion in color_map:
            return color_map[emotion]
    
    return "#e9ecef"

def get_emotion_emoji(emotions: List[str]) -> str:
    """감정에 따른 이모지 반환"""
    if not emotions:
        return "😐"
    
    emoji_map = {
        "기쁨": "😊", "행복": "😊", "평온": "😌", "만족": "🙂",
        "슬픔": "😢", "불안": "😰", "걱정": "😟", "분노": "😠",
        "짜증": "😤", "스트레스": "😵", "피로": "😴", "설렘": "😍",
        "중립": "😐"
    }
    
    for emotion in emotions:
        if emotion in emoji_map:
            return emoji_map[emotion]
    
    return "😐"

def create_emotion_calendar():
    """감정 캘린더 생성"""
    st.subheader("📅 나의 감정 캘린더")
    
    if not st.session_state.diary_entries:
        st.info("기록이 쌓이면 캘린더로 감정 패턴을 확인할 수 있어요!")
        return

    # 현재 날짜
    today = get_korean_time()
    
    # 기록이 있는 년-월 목록 생성
    available_months = set()
    for entry in st.session_state.diary_entries:
        entry_date = entry.get("date", "")
        if entry_date:
            year_month = entry_date[:7]  # YYYY-MM 형식
            available_months.add(year_month)
    
    # 현재 월도 추가
    current_month = today.strftime("%Y-%m")
    available_months.add(current_month)
    
    # 정렬된 월 목록
    sorted_months = sorted(list(available_months), reverse=True)
    
    # 월 선택
    col1, col2 = st.columns([1, 3])
    with col1:
        if sorted_months:
            selected_month_str = st.selectbox(
                "월 선택",
                sorted_months,
                index=0,
                format_func=lambda x: f"{x.split('-')[0]}년 {int(x.split('-')[1])}월"
            )
            # 선택된 월을 datetime 객체로 변환
            year, month = map(int, selected_month_str.split('-'))
            selected_month = datetime(year, month, 1).date()
        else:
            selected_month = today.date().replace(day=1)
    
    # 선택된 월의 데이터 필터링
    month_str = selected_month.strftime("%Y-%m")
    month_entries = {}
    for entry in st.session_state.diary_entries:
        entry_date = entry.get("date", "")
        if entry_date.startswith(month_str):
            try:
                day = int(entry_date.split("-")[2])
                if day not in month_entries:
                    month_entries[day] = []
                month_entries[day].append(entry)
            except (IndexError, ValueError):
                continue

    # 캘린더 그리드 생성
    year = selected_month.year
    month = selected_month.month
    
    # 월 정보 표시
    with col2:
        st.markdown(f"### {year}년 {month}월")
        total_entries_this_month = sum(len(entries) for entries in month_entries.values())
        st.caption(f"이번 달 총 {total_entries_this_month}개의 기록")
    
    try:
        cal = calendar.monthcalendar(year, month)
    except Exception:
        st.error("캘린더 생성 중 오류가 발생했습니다.")
        return
    
    # 요일 헤더
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    cols = st.columns(7)
    for i, day in enumerate(weekdays):
        cols[i].markdown(f"<div style='text-align: center; font-weight: bold; padding: 8px;'>{day}</div>", 
                        unsafe_allow_html=True)

    # 캘린더 날짜들
    for week_idx, week in enumerate(cal):
        cols = st.columns(7)
        for day_idx, day in enumerate(week):
            if day == 0:
                # 빈 날짜 (이전/다음 달)
                cols[day_idx].markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
            else:
                entries_for_day = month_entries.get(day, [])
                
                # 오늘 날짜 확인
                is_today = (day == today.day and month == today.month and year == today.year)
                
                if entries_for_day:
                    # 해당 날짜에 기록이 있는 경우
                    latest_entry = entries_for_day[-1]  # 가장 최근 기록
                    emotions = latest_entry.get("analysis", {}).get("emotions", [])
                    emoji = get_emotion_emoji(emotions)
                    color = get_emotion_color(emotions)
                    
                    # 버튼 키 생성 (고유하게)
                    button_key = f"cal_{year}_{month}_{day}_{week_idx}_{day_idx}"
                    
                    with cols[day_idx]:
                        button_clicked = st.button(
                            f"{emoji}\n{day}",
                            key=button_key,
                            help=f"{', '.join(emotions)} ({len(entries_for_day)}개 기록)",
                            use_container_width=True
                        )
                        
                        if button_clicked:
                            # 날짜 클릭 시 상세 정보 표시
                            st.session_state[f"show_day_{year}_{month}_{day}"] = True
                        
                        # 배경색 표시를 위한 스타일
                        border_style = "border: 2px solid #667eea;" if is_today else "border: 1px solid #ddd;"
                        background_style = f"background: {color}; opacity: 0.3;"
                        
                        st.markdown(
                            f"""
                            <div style='{background_style} {border_style} 
                                       border-radius: 8px; height: 10px; margin-top: 2px;'>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                else:
                    # 기록이 없는 날짜
                    border_style = "border: 2px solid #667eea;" if is_today else "border: 1px solid #ddd;"
                    
                    cols[day_idx].markdown(
                        f"""
                        <div style='{border_style} border-radius: 8px; padding: 20px; margin: 2px; 
                                   text-align: center; background: #fafafa; color: #999;'>
                            {day}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # 선택된 날짜 상세 정보 표시
    show_day_details = False
    for day in range(1, 32):
        session_key = f"show_day_{year}_{month}_{day}"
        if st.session_state.get(session_key, False):
            show_day_details = True
            entries_for_day = month_entries.get(day, [])
            if entries_for_day:
                st.markdown(f"### {year}년 {month}월 {day}일 기록")
                
                for i, entry in enumerate(entries_for_day):
                    emotions_str = ', '.join(entry.get('analysis', {}).get('emotions', []))
                    with st.expander(f"📝 {entry.get('time', '')} - {emotions_str}", expanded=(i==0)):
                        st.write(entry.get("text", ""))
                        analysis = entry.get("analysis", {})
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("스트레스", f"{analysis.get('stress_level', 0)}%")
                        col2.metric("에너지", f"{analysis.get('energy_level', 0)}%")
                        col3.metric("기분", f"{analysis.get('mood_score', 0)}")
                
                # 닫기 버튼
                if st.button("닫기", key=f"close_{year}_{month}_{day}"):
                    st.session_state[session_key] = False
                    st.rerun()
                
                break  # 한 번에 하나의 날짜만 표시

# =============================
# Goal Management Functions
# =============================
def add_goal(goal_type: str, target_value: float, description: str):
    """목표 추가"""
    goal = {
        "id": len(st.session_state.user_goals) + 1,
        "type": goal_type,  # "stress", "energy", "mood", "consistency"
        "target": target_value,
        "description": description,
        "created_date": today_key(),
        "active": True
    }
    st.session_state.user_goals.append(goal)

def check_goal_progress(goal: Dict) -> Dict:
    """목표 진행률 확인"""
    recent_entries = st.session_state.diary_entries[-7:]  # 최근 7일
    if not recent_entries:
        return {"progress": 0, "current_value": 0, "status": "진행중"}

    goal_type = goal["type"]
    target = goal["target"]
    
    if goal_type == "consistency":
        # 일관성 목표 (주간 기록 횟수)
        current_value = len(recent_entries)
        progress = min(100, (current_value / target) * 100)
    else:
        # 수치 목표
        values = []
        for entry in recent_entries:
            analysis = entry.get("analysis", {})
            if goal_type == "stress":
                values.append(analysis.get("stress_level", 0))
            elif goal_type == "energy":
                values.append(analysis.get("energy_level", 0))
            elif goal_type == "mood":
                values.append(analysis.get("mood_score", 0))
        
        if values:
            current_value = np.mean(values)
            if goal_type == "stress":
                # 스트레스는 낮을수록 좋음
                progress = max(0, min(100, (target - current_value) / target * 100)) if current_value <= target else 100
            else:
                # 에너지, 기분은 높을수록 좋음
                progress = min(100, (current_value / target) * 100)
        else:
            current_value = 0
            progress = 0

    status = "달성!" if progress >= 100 else "진행중"
    
    return {
        "progress": progress,
        "current_value": current_value,
        "status": status
    }

def create_goals_page():
    """목표 설정 및 추적 페이지"""
    st.header("🎯 나의 목표 설정 & 추적")
    
    # 새 목표 추가
    with st.expander("➕ 새로운 목표 추가하기"):
        col1, col2 = st.columns(2)
        
        with col1:
            goal_type = st.selectbox(
                "목표 유형",
                ["stress", "energy", "mood", "consistency"],
                format_func=lambda x: {
                    "stress": "스트레스 관리 (낮추기)",
                    "energy": "에너지 증진 (높이기)", 
                    "mood": "기분 개선 (높이기)",
                    "consistency": "꾸준한 기록 (일주일 기준)"
                }[x]
            )
        
        with col2:
            if goal_type == "consistency":
                target_value = st.slider("주간 목표 기록 횟수", 1, 7, 5)
                description = f"일주일에 {target_value}번 이상 기록하기"
            elif goal_type == "stress":
                target_value = st.slider("목표 스트레스 지수 (이하)", 10, 50, 30)
                description = f"스트레스 지수를 {target_value} 이하로 유지하기"
            elif goal_type == "energy":
                target_value = st.slider("목표 에너지 지수 (이상)", 50, 90, 70)
                description = f"에너지 지수를 {target_value} 이상으로 유지하기"
            else:  # mood
                target_value = st.slider("목표 기분 점수 (이상)", 0, 50, 20)
                description = f"기분 점수를 {target_value} 이상으로 유지하기"

        custom_desc = st.text_input("목표 설명 (선택사항)", value=description)
        
        if st.button("목표 추가"):
            add_goal(goal_type, target_value, custom_desc)
            st.success("새로운 목표가 추가되었습니다!")
            st.rerun()

    # 현재 목표들 표시
    active_goals = [g for g in st.session_state.user_goals if g.get("active", True)]
    
    if not active_goals:
        st.info("아직 설정된 목표가 없습니다. 위에서 새로운 목표를 추가해보세요!")
        return

    st.subheader("📊 목표 진행 상황")
    
    for goal in active_goals:
        progress_info = check_goal_progress(goal)
        progress = progress_info["progress"]
        current = progress_info["current_value"]
        status = progress_info["status"]
        
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{goal['description']}**")
                st.progress(progress / 100)
                st.caption(f"진행률: {progress:.1f}% | 현재값: {current:.1f}")
            
            with col2:
                if status == "달성!":
                    st.success(status)
                else:
                    st.info(status)
            
            with col3:
                if st.button("🗑️", key=f"delete_goal_{goal['id']}", help="목표 삭제"):
                    goal["active"] = False
                    st.success("목표가 삭제되었습니다!")
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Onboarding Functions
# =============================
def show_onboarding_guide():
    """온보딩 가이드 표시"""
    if not st.session_state.onboarding_completed:
        with st.expander("🌟 어떤 이야기를 할지 막막하신가요? (처음 사용자 가이드)", expanded=True):
            st.markdown("""
            ### 💭 이런 이야기들을 나눠보세요
            
            **🌅 하루 시작/마무리**
            - "오늘 하루 가장 기억에 남는 순간은 언제였나요?"
            - "오늘 가장 감사했던 일은 무엇인가요?"
            - "내일 가장 기대되는 일은 무엇인가요?"
            
            **💚 감정과 기분**
            - "지금 이 순간 어떤 기분이신가요?"
            - "요즘 나를 가장 힘들게 하는 것은 무엇인가요?"
            - "사소하지만 나를 웃게 만들었던 일은 무엇이었나요?"
            
            **🎯 목표와 성장**
            - "오늘 한 일 중에서 가장 뿌듯했던 것은?"
            - "내가 최근에 성장했다고 느끼는 부분이 있나요?"
            - "지금 가장 집중하고 싶은 것은 무엇인가요?"
            
            **🔄 일상과 루틴**
            - "오늘의 컨디션을 10점 만점에 몇 점으로 평가하시나요?"
            - "최근 잠은 잘 주무시고 계신가요?"
            - "스트레스를 받을 때 어떻게 해소하시나요?"
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎯 바로 시작하기"):
                    st.session_state.onboarding_completed = True
                    st.rerun()
            with col2:
                if st.button("📚 더 많은 팁 보기"):
                    st.session_state.show_more_tips = True

        if st.session_state.get("show_more_tips", False):
            with st.expander("📋 효과적인 기록을 위한 팁"):
                st.markdown("""
                ### 🎙️ 음성 녹음 팁
                - **조용한 환경**에서 녹음하세요
                - **자연스럽게** 말해주세요 (연기할 필요 없어요!)
                - **2-3분** 정도가 적당합니다
                - **핸드폰을 입에서 20cm** 정도 떨어뜨려 주세요
                
                ### ✍️ 텍스트 입력 팁  
                - **솔직한 감정**을 표현해주세요
                - **구체적인 상황**을 포함하면 더 정확한 분석이 가능해요
                - **5-10문장** 정도면 충분합니다
                - **어려운 날도** 기록해보세요 - 패턴을 찾는 데 도움됩니다
                """)

# =============================
# Technique Display Functions
# =============================
def display_wellness_techniques(techniques: List[Dict]):
    """웰빙 기법들을 보기 좋게 표시"""
    if not techniques:
        return
    
    st.markdown("### 🌱 추천 웰빙 기법")
    
    for technique in techniques:
        with st.container():
            st.markdown("<div class='technique-card'>", unsafe_allow_html=True)
            
            # 기법 이름과 난이도
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**🎯 {technique['name']}**")
            with col2:
                difficulty = technique.get('difficulty', '초급')
                color = {"초급": "🟢", "중급": "🟡", "고급": "🔴"}.get(difficulty, "🟢")
                st.markdown(f"{color} {difficulty}")
            
            # 설명
            st.markdown(f"*{technique['description']}*")
            
            # 단계별 안내
            if 'steps' in technique:
                st.markdown("**실천 방법:**")
                for i, step in enumerate(technique['steps'], 1):
                    st.markdown(f"{i}. {step}")
            
            # 소요시간과 주의사항
            col1, col2 = st.columns(2)
            with col1:
                if 'duration' in technique:
                    st.markdown(f"⏱️ **소요 시간:** {technique['duration']}")
            with col2:
                if 'category' in technique:
                    st.markdown(f"📂 **분야:** {technique['category']}")
            
            # 주의사항 (있는 경우)
            if technique.get('warning'):
                st.warning(f"⚠️ {technique['warning']}")
            
            st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Sidebar
# =============================
if not st.session_state.show_disclaimer:
    with st.sidebar:
        st.markdown("### 🔧 시스템 상태")
        librosa = get_librosa()
        parselmouth = get_parselmouth()
        
        st.markdown(f"- {'✅' if openai_client else '⚠️'} OpenAI API")
        st.markdown(f"- {'✅' if librosa else '⚠️'} 음성 분석(Librosa)")
        st.markdown(f"- {'✅' if parselmouth else 'ℹ️'} 고급 음성학(Praat)")
        st.markdown(f"- ✅ RAG 심리건강 지식베이스")
        
        if not openai_client:
            with st.expander("🔑 OpenAI API 키 입력"):
                api_key = st.text_input("OpenAI API 키", type="password")
                if st.button("저장"):
                    if api_key.startswith("sk-"):
                        st.session_state.openai_api_key = api_key
                        st.success("API 키가 저장되었습니다. 상단 Rerun 버튼으로 새로고침 해주세요.")
                    else:
                        st.error("올바른 키 형식이 아닙니다.")

        st.markdown("---")
        
        # 페이지 선택
        page = st.selectbox(
            "페이지", 
            [
                "🎙️ 오늘의 이야기", 
                "💖 마음 분석", 
                "📈 감정 여정", 
                "📅 감정
