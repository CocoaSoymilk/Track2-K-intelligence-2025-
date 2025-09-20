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
warnings.filterwarnings("ignore")

# Lightweight stdlib
import numpy as np
from typing import Dict, List, Optional, Tuple

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
# Coaching Functions
# =============================
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

def assess_mental_state(text: str, combined: Dict) -> Dict:
    """정신 상태 평가 (규칙 기반 폴백)"""
    tone = combined.get("tone", "중립적")
    stress = combined.get("stress_level", 30)
    energy = combined.get("energy_level", 50)
    mood = combined.get("mood_score", 0)
    
    cues = combined.get("voice_analysis", {}).get("voice_cues", {})
    arousal = float(cues.get("arousal", 50))
    tension = float(cues.get("tension", 50))
    stability = float(cues.get("stability", 50))
    quality = float(cues.get("quality", 0.5))

    positives = extract_positive_events(text)

    # 상태 판정
    state = "중립"
    if tone == "긍정적" and mood >= 15 and stress < 40:
        state = "안정/회복"
    if energy < 40 and mood < 0:
        state = "저활력"
    if stress >= 60:
        state = "고스트레스"
    
    if quality > 0.4:
        if tension > 65 and stability < 45:
            state = "긴장 과다"
        elif arousal > 70 and stress > 45:
            state = "과흥분/과부하 가능"
        elif arousal < 40 and energy < 45:
            state = "저각성"

    # 추천사항 생성
    recs: List[str] = []
    if tone == "긍정적" or positives:
        if positives:
            recs.append("오늘 좋았던 포인트를 3줄로 기록해 보세요 (감사/성취/즐거움).")
        recs.append("좋았던 활동을 내일 10분만 더 해보기.")
    
    if tension > 60:
        recs.append("4-7-8 호흡 3회: 4초 들이마시고, 7초 멈추고, 8초 내쉬기.")
    
    if stability < 50:
        recs.append("목/어깨 이완 스트레칭 2분 (상체 회전, 목 옆선 늘리기).")
    
    if arousal < 45 or energy < 45:
        recs.append("햇빛 10분 산책 + 가벼운 워킹 (Step 800~1000).")
    
    if arousal > 65 and stress > 50:
        recs.append("알림/자극 줄이기: 25분 집중 + 5분 휴식(포모도로 2회).")

    recs = recs[:4]

    # 동기부여 메시지
    mot = "작은 습관이 오늘의 좋은 흐름을 내일로 이어줍니다."
    if state in ("고스트레스", "긴장 과다"):
        mot = "호흡을 고르고, 천천히. 당신의 속도로 충분합니다."
    elif state in ("저활력", "저각성"):
        mot = "작은 한 걸음이 에너지를 깨웁니다. 10분만 움직여볼까요?"

    summary = (
        f"상태: {state} · 스트레스 {stress} · 에너지 {energy} · "
        f"각성 {int(arousal)} / 긴장 {int(tension)} / 안정 {int(stability)}"
    )

    return {
        "state": state,
        "summary": summary,
        "positives": positives,
        "recommendations": recs,
        "motivation": mot,
        "voice_cues": {
            "arousal": arousal,
            "tension": tension,
            "stability": stability,
            "quality": quality
        }
    }

def generate_llm_coach_report(text: str, combined: Dict, recent: Optional[List[Dict]] = None) -> Dict:
    """LLM을 사용한 풍부한 코칭 리포트 생성"""
    if not openai_client:
        return assess_mental_state(text, combined)

    cues = combined.get("voice_analysis", {}).get("voice_cues", {})
    
    try:
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
            "당신은 따뜻한 한국어 코치입니다. 텍스트는 감정 라벨의 기준이며, 음성은 각성/긴장/안정의 보조지표로만 고려하세요. "
            "다음 JSON 형식으로만 답하세요: "
            '{"state": "상태", "summary": "요약", "positives": ["긍정요소"], "recommendations": ["추천사항"], "motivation": "격려메시지"}'
        )
        
        user_payload = {
            "text": text,
            "text_analysis": {
                "emotions": combined.get("emotions", []),
                "stress": combined.get("stress_level", 30),
                "energy": combined.get("energy_level", 50),
                "mood": combined.get("mood_score", 0),
                "tone": combined.get("tone", "중립적"),
            },
            "voice_cues": {
                "arousal": int(cues.get("arousal", 50)),
                "tension": int(cues.get("tension", 50)),
                "stability": int(cues.get("stability", 50)),
                "quality": float(cues.get("quality", 0.5)),
            },
            "recent_summary": history_blob,
        }

        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=500,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
        )

        content = resp.choices[0].message.content
        if not content:
            return assess_mental_state(text, combined)
        
        data = safe_json_parse(content)
        
        if not data:
            return assess_mental_state(text, combined)

        # 필수 필드 보장
        data.setdefault("state", "중립")
        data.setdefault("summary", "오늘의 상태를 차분히 정리했어요.")
        data.setdefault("positives", [])
        data.setdefault("recommendations", [])
        data.setdefault("motivation", "작은 걸음이 큰 변화를 만듭니다.")
        
        # 길이 제한
        data["recommendations"] = data.get("recommendations", [])[:4]
        data["positives"] = data.get("positives", [])[:4]
        
        return data
        
    except Exception as e:
        print(f"코칭 리포트 생성 오류: {e}")
        return assess_mental_state(text, combined)

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
                "📅 감정 캘린더",
                "🎯 나의 목표", 
                "🎵 목소리 보조지표", 
                "📚 나의 이야기들"
            ]
        )

        # 현재 상태 요약
        if st.session_state.diary_entries:
            st.markdown("### 📊 현재 상태")
            latest = st.session_state.diary_entries[-1]
            analysis = latest.get("analysis", {})
            
            st.metric("기록 수", f"{len(st.session_state.diary_entries)}개")
            st.metric("최근 스트레스", f"{analysis.get('stress_level', 0)}%")
            st.metric("최근 에너지", f"{analysis.get('energy_level', 0)}%")
            
            # 주간 리포트 버튼
            if len(st.session_state.diary_entries) >= 7:
                st.markdown("---")
                if st.button("📋 주간 리포트 생성"):
                    st.session_state.show_weekly_report = True

        # 서비스 안내
        st.markdown("---")
        with st.expander("ℹ️ 서비스 안내"):
            st.markdown("""
            **🛡️ 데이터 보안**
            - 모든 기록은 세션에만 저장
            - 브라우저 종료 시 자동 삭제
            
            **⚕️ 의료적 한계**
            - 자기 성찰 보조 도구
            - 의료 진단/치료 대체 불가
            
            **🤖 AI 분석**
            - 감정 라벨: 텍스트 기반
            - 음성: 보조 지표로만 활용
            """)

# =============================
# Main Pages
# =============================
if not st.session_state.show_disclaimer:
    extractor = VoiceFeatureExtractor()

    if page == "🎙️ 오늘의 이야기":
        st.header("오늘 하루는 어떠셨나요?")
        
        # 온보딩 가이드 표시
        show_onboarding_guide()
        
        # 입력 섹션
        audio_val = st.audio_input("🎤 마음을 편하게 말해보세요", help="녹음 후 업로드")
        text_input = st.text_area(
            "✏️ 글로 표현해도 좋아요", 
            placeholder="오늘의 이야기를 적어주세요...", 
            height=120
        )

        if st.button("💝 분석하고 저장", type="primary"):
            diary_text = text_input.strip()
            voice_analysis = None
            audio_b64 = None
            
            # 음성 처리
            if audio_val is not None:
                audio_bytes = audio_val.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                
                with st.spinner("🎵 목소리 신호를 계산하는 중..."):
                    vf = extractor.extract(audio_bytes)
                    update_baseline(vf)
                    voice_analysis = analyze_voice_as_cues(vf, st.session_state.prosody_baseline)
                
                # 음성 전사 (텍스트가 없을 때)
                if openai_client and not diary_text:
                    with st.spinner("🤖 음성을 텍스트로 변환 중..."):
                        tx = transcribe_audio(audio_bytes)
                        if tx:
                            diary_text = tx
                            st.info(f"🎤 들은 이야기: {tx}")

            if not diary_text:
                st.warning("텍스트를 입력하거나 음성을 녹음해 주세요.")
            else:
                # 분석 수행
                cues_for_prompt = voice_analysis["voice_cues"] if voice_analysis else None
                
                with st.spinner("🤖 텍스트 기반 감정 분석 중..."):
                    t_res = analyze_text_with_llm(diary_text, cues_for_prompt)
                
                # 텍스트와 음성 결합
                final = combine_text_and_voice(t_res, voice_analysis)
                
                # 코칭 리포트 생성
                recent_entries = st.session_state.diary_entries[-7:] if st.session_state.diary_entries else []
                ms_card = generate_llm_coach_report(diary_text, final, recent_entries)
                
                # 엔트리 저장
                entry = {
                    "id": len(st.session_state.diary_entries) + 1,
                    "date": today_key(),
                    "time": current_time(),
                    "text": diary_text,
                    "analysis": final,
                    "audio_data": audio_b64,
                    "mental_state": ms_card,
                }
                st.session_state.diary_entries.append(entry)
                st.success("🎉 소중한 이야기가 저장되었습니다!")
                
                # 결과 표시
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("💖 감정 (텍스트 기반)")
                    emotions_text = ", ".join(final.get("emotions", []))
                    st.write(emotions_text)
                    st.caption("감정 라벨은 텍스트만으로 판정합니다.")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("📊 마음 상태")
                    stress_color = "metric-negative" if final['stress_level'] > 60 else ("metric-positive" if final['stress_level'] < 30 else "metric-neutral")
                    energy_color = "metric-positive" if final['energy_level'] > 60 else ("metric-negative" if final['energy_level'] < 40 else "metric-neutral")
                    
                    st.markdown(f"**스트레스:** <span class='{stress_color}'>{final['stress_level']}%</span>", unsafe_allow_html=True)
                    st.markdown(f"**활력:** <span class='{energy_color}'>{final['energy_level']}%</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("🎯 컨디션")
                    mood_color = "metric-positive" if final['mood_score'] > 10 else ("metric-negative" if final['mood_score'] < -10 else "metric-neutral")
                    st.markdown(f"**마음 점수:** <span class='{mood_color}'>{final['mood_score']}</span>", unsafe_allow_html=True)
                    st.metric("분석 신뢰도", f"{final.get('confidence', 0.6):.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # 음성 분석 결과
                if voice_analysis:
                    st.markdown("### 🎵 목소리 신호 (보조 지표)")
                    cues = final["voice_analysis"]["voice_cues"]
                    quality_text = "높음" if cues["quality"] > 0.7 else ("보통" if cues["quality"] > 0.4 else "낮음")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("각성도", f"{int(cues['arousal'])}/100")
                    c2.metric("긴장도", f"{int(cues['tension'])}/100")
                    c3.metric("안정도", f"{int(cues['stability'])}/100")
                    c4.metric("녹음 품질", quality_text)
                    st.caption("※ 목소리 신호는 보조 지표입니다. 감정 판단은 텍스트에 기반합니다.")
                
                # 코칭 카드
                st.markdown("### 🧠 오늘의 마음 코치")
                card_class = "success-card" if ms_card.get("state") == "안정/회복" else ("warning-card" if "스트레스" in ms_card.get("state", "") else "card")
                
                with st.container():
                    st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
                    st.write(f"**상태:** {ms_card.get('state', '중립')}")
                    st.write(ms_card.get("summary", "오늘의 상태를 차분히 정리했어요."))
                    
                    if ms_card.get("positives"):
                        st.write("**🌟 오늘의 밝은 포인트**")
                        for positive in ms_card["positives"]:
                            st.write(f"• {positive}")
                    
                    st.write("**💡 추천 행동**")
                    for i, rec in enumerate(ms_card.get("recommendations", []), 1):
                        st.write(f"{i}. {rec}")
                    
                    st.info(f"💪 {ms_card.get('motivation', '오늘도 잘 해내셨어요.')}")
                    st.markdown("</div>", unsafe_allow_html=True)

    elif page == "📅 감정 캘린더":
        create_emotion_calendar()

    elif page == "🎯 나의 목표":
        create_goals_page()

    elif page == "💖 마음 분석":
        st.header("마음 분석 대시보드")
        
        if not st.session_state.diary_entries:
            st.info("기록이 아직 없어요. 첫 번째 이야기를 들려주세요! 📝")
        else:
            # 전체 통계
            st.subheader("📊 전체 통계")
            
            total_entries = len(st.session_state.diary_entries)
            recent_entries = st.session_state.diary_entries[-30:]  # 최근 30개
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 기록 수", f"{total_entries}개")
            
            with col2:
                avg_stress = np.mean([e["analysis"].get("stress_level", 0) for e in recent_entries])
                st.metric("평균 스트레스", f"{avg_stress:.0f}%")
            
            with col3:
                avg_energy = np.mean([e["analysis"].get("energy_level", 0) for e in recent_entries])
                st.metric("평균 에너지", f"{avg_energy:.0f}%")
            
            with col4:
                avg_mood = np.mean([e["analysis"].get("mood_score", 0) for e in recent_entries])
                st.metric("평균 기분", f"{avg_mood:.0f}")
            
            # 감정 분포
            st.subheader("😊 감정 분포 (최근 30개 기록)")
            emotion_counts = {}
            for entry in recent_entries:
                emotions = entry["analysis"].get("emotions", [])
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if emotion_counts:
                emotion_df = pd.DataFrame(list(emotion_counts.items()), columns=["감정", "횟수"])
                st.bar_chart(emotion_df.set_index("감정"))
            
            # 상세 데이터 테이블
            st.subheader("📋 상세 기록")
            df = pd.DataFrame([
                {
                    "날짜": e["date"],
                    "시간": e["time"],
                    "감정": ", ".join(e["analysis"].get("emotions", [])),
                    "스트레스": e["analysis"].get("stress_level", 0),
                    "에너지": e["analysis"].get("energy_level", 0),
                    "기분": e["analysis"].get("mood_score", 0),
                    "톤": e["analysis"].get("tone", "중립적"),
                    "신뢰도": f"{e['analysis'].get('confidence', 0.6):.2f}"
                }
                for e in st.session_state.diary_entries
            ])
            
            # 필터링 옵션
            col1, col2 = st.columns(2)
            with col1:
                date_filter = st.date_input("날짜 필터 (이후)", value=None)
            with col2:
                emotion_filter = st.selectbox("감정 필터", ["전체"] + list(emotion_counts.keys()))
            
            # 필터 적용
            filtered_df = df.copy()
            if date_filter:
                filtered_df = filtered_df[pd.to_datetime(filtered_df["날짜"]) >= pd.to_datetime(date_filter)]
            if emotion_filter != "전체":
                filtered_df = filtered_df[filtered_df["감정"].str.contains(emotion_filter)]
            
            st.dataframe(
                filtered_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "스트레스": st.column_config.ProgressColumn("스트레스", max_value=100),
                    "에너지": st.column_config.ProgressColumn("에너지", max_value=100),
                }
            )

    elif page == "📈 감정 여정":
        st.header("시간에 따른 변화")
        
        if not st.session_state.diary_entries:
            st.info("기록이 쌓이면 추세를 보여드릴게요. 꾸준히 기록해보세요! 📈")
        else:
            # 기간 선택
            col1, col2 = st.columns(2)
            with col1:
                period = st.selectbox("기간 선택", ["전체", "최근 30일", "최근 14일", "최근 7일"])
            
            # 데이터 필터링
            entries = st.session_state.diary_entries
            if period == "최근 30일":
                entries = entries[-30:]
            elif period == "최근 14일":
                entries = entries[-14:]
            elif period == "최근 7일":
                entries = entries[-7:]
            
            if len(entries) < 2:
                st.warning("추세 분석을 위해서는 최소 2개 이상의 기록이 필요합니다.")
            else:
                # 차트 데이터 준비
                df = pd.DataFrame([
                    {
                        "날짜시간": f"{e['date']} {e['time']}",
                        "날짜": e['date'],
                        "스트레스": e["analysis"].get("stress_level", 0),
                        "에너지": e["analysis"].get("energy_level", 0),
                        "기분": e["analysis"].get("mood_score", 0) + 70  # 시각화를 위해 0-140 범위로 조정
                    }
                    for e in entries
                ])
                
                # 메트릭 선택
                with col2:
                    metric = st.selectbox("지표 선택", ["전체", "스트레스", "에너지", "기분"])
                
                # 차트 표시
                if metric == "전체":
                    st.line_chart(df.set_index("날짜시간")[["스트레스", "에너지", "기분"]])
                else:
                    if metric == "기분":
                        st.line_chart(df.set_index("날짜시간")[["기분"]])
                        st.caption("※ 기분 점수는 시각화를 위해 +70 조정되었습니다 (실제: -70~70)")
                    else:
                        st.line_chart(df.set_index("날짜시간")[[metric]])
                
                # 추세 분석
                st.subheader("📊 추세 분석")
                
                stress_trend = np.polyfit(range(len(entries)), [e["analysis"].get("stress_level", 0) for e in entries], 1)[0]
                energy_trend = np.polyfit(range(len(entries)), [e["analysis"].get("energy_level", 0) for e in entries], 1)[0]
                mood_trend = np.polyfit(range(len(entries)), [e["analysis"].get("mood_score", 0) for e in entries], 1)[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trend_icon = "📉" if stress_trend < -0.1 else ("📈" if stress_trend > 0.1 else "➡️")
                    trend_text = "감소" if stress_trend < -0.1 else ("증가" if stress_trend > 0.1 else "안정")
                    st.metric("스트레스 추세", f"{trend_icon} {trend_text}", delta=f"{stress_trend:.2f}")
                
                with col2:
                    trend_icon = "📈" if energy_trend > 0.1 else ("📉" if energy_trend < -0.1 else "➡️")
                    trend_text = "증가" if energy_trend > 0.1 else ("감소" if energy_trend < -0.1 else "안정")
                    st.metric("에너지 추세", f"{trend_icon} {trend_text}", delta=f"{energy_trend:.2f}")
                
                with col3:
                    trend_icon = "📈" if mood_trend > 0.1 else ("📉" if mood_trend < -0.1 else "➡️")
                    trend_text = "개선" if mood_trend > 0.1 else ("하락" if mood_trend < -0.1 else "안정")
                    st.metric("기분 추세", f"{trend_icon} {trend_text}", delta=f"{mood_trend:.2f}")
                
                # 인사이트
                st.subheader("🔍 인사이트")
                insights = []
                
                if stress_trend < -0.5:
                    insights.append("✨ 스트레스가 꾸준히 감소하고 있어요! 현재 방식을 유지해보세요.")
                elif stress_trend > 0.5:
                    insights.append("⚠️ 스트레스가 증가하는 추세입니다. 휴식과 스트레스 관리가 필요해 보여요.")
                
                if energy_trend > 0.5:
                    insights.append("🔋 에너지 레벨이 상승하고 있어요! 좋은 습관들을 계속 이어가세요.")
                elif energy_trend < -0.5:
                    insights.append("😴 에너지가 떨어지고 있습니다. 충분한 휴식과 운동을 고려해보세요.")
                
                if mood_trend > 0.5:
                    insights.append("😊 기분이 전반적으로 좋아지고 있어요! 긍정적인 변화네요.")
                elif mood_trend < -0.5:
                    insights.append("💙 기분이 다소 가라앉는 추세입니다. 자신을 돌보는 시간을 가져보세요.")
                
                if not insights:
                    insights.append("📊 전반적으로 안정적인 상태를 유지하고 있어요.")
                
                for insight in insights:
                    st.info(insight)

    elif page == "🎵 목소리 보조지표":
        st.header("목소리 신호 상세 분석")
        
        entries_with_voice = [e for e in st.session_state.diary_entries if e.get("analysis", {}).get("voice_analysis")]
        
        if not entries_with_voice:
            st.info("음성으로 기록된 항목이 아직 없습니다. 첫 음성 기록을 남겨보세요! 🎤")
        else:
            # 최신 음성 기록 선택
            selected_entry = st.selectbox(
                "분석할 기록 선택",
                entries_with_voice,
                format_func=lambda x: f"{x['date']} {x['time']} - {', '.join(x['analysis'].get('emotions', []))}",
                index=len(entries_with_voice) - 1
            )
            
            voice = selected_entry["analysis"]["voice_analysis"]
            vf = voice["voice_features"]
            cues = voice["voice_cues"]
            
            # 보조지표 메트릭
            st.subheader("🎯 음성 보조지표")
            quality_text = "높음" if cues["quality"] > 0.7 else ("보통" if cues["quality"] > 0.4 else "낮음")
            quality_color = "🟢" if cues["quality"] > 0.7 else ("🟡" if cues["quality"] > 0.4 else "🔴")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("각성도", f"{int(cues['arousal'])}/100", help="음성의 활기찬 정도")
            c2.metric("긴장도", f"{int(cues['tension'])}/100", help="음성의 긴장된 정도")
            c3.metric("안정도", f"{int(cues['stability'])}/100", help="음성의 안정된 정도")
            c4.metric("녹음 품질", f"{quality_color} {quality_text}", help="분석 신뢰도에 영향")
            
            # 음성 특성 상세
            st.subheader("🔬 기초 음성 특성")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 기본 측정값**")
                d1, d2 = st.columns(2)
                d1.metric("피치 평균", f"{vf.get('pitch_mean',0):.1f} Hz")
                d2.metric("피치 변동", f"{vf.get('pitch_variation',0):.3f}")
                
                d3, d4 = st.columns(2)
                d3.metric("음성 에너지", f"{vf.get('energy_mean',0):.3f}")
                d4.metric("최대 에너지", f"{vf.get('energy_max',0):.3f}")
            
            with col2:
                st.markdown("**🎵 고급 측정값**")
                d5, d6 = st.columns(2)
                d5.metric("말하기 속도", f"{vf.get('tempo',0):.0f} BPM")
                d6.metric("영교차율", f"{vf.get('zcr_mean',0):.3f}")
                
                d7, d8 = st.columns(2)
                d7.metric("HNR (명료도)", f"{vf.get('hnr',0):.1f} dB")
                d8.metric("Jitter (안정성)", f"{vf.get('jitter',0):.4f}")
            
            # 해석 가이드
            st.subheader("📖 해석 가이드")
            
            with st.expander("🎯 보조지표 의미"):
                st.markdown("""
                - **각성도**: 음성의 활기참과 에너지 수준을 나타냅니다
                - **긴장도**: 음성에 나타나는 긴장이나 스트레스 정도를 측정합니다  
                - **안정도**: 음성의 일관성과 안정성을 평가합니다
                - **품질**: 분석의 신뢰도에 영향을 주는 녹음 품질입니다
                """)
            
            with st.expander("🔬 기술적 지표 설명"):
                st.markdown("""
                - **피치 (Pitch)**: 목소리의 높낮이 (Hz 단위)
                - **에너지 (Energy)**: 목소리의 크기와 강도
                - **속도 (Tempo)**: 말하는 빠르기 (BPM 단위)
                - **HNR**: 목소리의 명료도와 떨림 정도
                - **Jitter**: 음성 주파수의 변동성 (안정성 지표)
                """)
            
            # 베이스라인 정보
            if st.session_state.prosody_baseline:
                st.subheader("📈 개인 베이스라인")
                baseline = st.session_state.prosody_baseline
                baseline_count = baseline.get("_count", 0)
                
                st.info(f"현재 {baseline_count}개 기록을 바탕으로 개인 베이스라인이 설정되어 있습니다.")
                
                if st.button("베이스라인 초기화"):
                    st.session_state.prosody_baseline = {}
                    st.success("베이스라인이 초기화되었습니다.")
                    st.rerun()
            
            st.caption("※ 이 수치들은 감정 분석을 위한 보조 지표로만 사용되며, 텍스트 기반 감정 라벨을 직접 결정하지 않습니다.")

    elif page == "📚 나의 이야기들":
        st.header("나의 이야기 아카이브")
        
        if not st.session_state.diary_entries:
            st.info("아직 기록된 이야기가 없어요. 첫 번째 이야기를 들려주세요! ✨")
        else:
            # 주간 리포트 버튼 (7개 이상일 때)
            if len(st.session_state.diary_entries) >= 7:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("📋 주간 리포트 생성", type="primary"):
                        with st.spinner("📊 주간 리포트를 생성하고 있습니다..."):
                            report = generate_weekly_report(st.session_state.diary_entries)
                            st.session_state.weekly_report = report
                            st.session_state.show_weekly_report = True
            
            # 주간 리포트 표시
            if st.session_state.get("show_weekly_report", False) and "weekly_report" in st.session_state:
                report = st.session_state.weekly_report
                
                st.markdown("### 📊 주간 웰빙 리포트")
                
                # 전체 추세
                trend_color = {"개선됨": "🟢", "안정적": "🟡", "주의필요": "🔴"}
                trend_icon = trend_color.get(report.get("overall_trend", "안정적"), "🟡")
                
                st.markdown(f"**전체 추세:** {trend_icon} {report.get('overall_trend', '안정적')}")
                
                # 주요 인사이트
                if report.get("key_insights"):
                    st.markdown("**🔍 주요 발견사항**")
                    for insight in report["key_insights"]:
                        st.write(f"• {insight}")
                
                # 패턴 분석
                patterns = report.get("patterns", {})
                if patterns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if patterns.get("best_days"):
                            st.markdown("**🌟 좋았던 날들**")
                            for day in patterns["best_days"]:
                                st.write(f"• {day}")
                    
                    with col2:
                        if patterns.get("challenging_days"):
                            st.markdown("**💪 도전적이었던 날들**")
                            for day in patterns["challenging_days"]:
                                st.write(f"• {day}")
                    
                    if patterns.get("emotional_patterns"):
                        st.markdown("**📈 감정 패턴**")
                        st.write(patterns["emotional_patterns"])
                
                # 추천사항
                recommendations = report.get("recommendations", {})
                if recommendations:
                    st.markdown("### 💡 다음 주를 위한 추천")
                    
                    if recommendations.get("priority_actions"):
                        st.markdown("**🎯 우선순위 행동**")
                        for i, action in enumerate(recommendations["priority_actions"], 1):
                            st.write(f"{i}. {action}")
                    
                    if recommendations.get("wellness_tips"):
                        st.markdown("**🌱 웰빙 팁**")
                        for tip in recommendations["wellness_tips"]:
                            st.write(f"• {tip}")
                    
                    if recommendations.get("goals_for_next_week"):
                        st.markdown("**🎯 다음 주 목표**")
                        for goal in recommendations["goals_for_next_week"]:
                            st.write(f"• {goal}")
                
                # 격려 메시지
                if report.get("encouragement"):
                    st.success(f"💪 {report['encouragement']}")
                
                if st.button("리포트 닫기"):
                    st.session_state.show_weekly_report = False
                    st.rerun()
                
                st.markdown("---")
            
            # 검색 및 필터
            st.subheader("🔍 기록 탐색")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                search_text = st.text_input("🔍 텍스트 검색", placeholder="키워드로 검색...")
            with col2:
                emotion_options = ["전체"] + list(set([
                    emotion for entry in st.session_state.diary_entries 
                    for emotion in entry.get("analysis", {}).get("emotions", [])
                ]))
                emotion_filter = st.selectbox("😊 감정 필터", emotion_options)
            with col3:
                date_filter = st.date_input("📅 날짜 이후", value=None)
            
            # 필터링된 엔트리
            filtered_entries = st.session_state.diary_entries
            
            if search_text:
                filtered_entries = [e for e in filtered_entries if search_text.lower() in e.get("text", "").lower()]
            
            if emotion_filter != "전체":
                filtered_entries = [e for e in filtered_entries if emotion_filter in e.get("analysis", {}).get("emotions", [])]
            
            if date_filter:
                filtered_entries = [e for e in filtered_entries if e.get("date", "") >= date_filter.strftime("%Y-%m-%d")]
            
            st.write(f"**총 {len(filtered_entries)}개의 기록** (전체 {len(st.session_state.diary_entries)}개 중)")
            
            # 기록 표시 (최신 20개)
            display_entries = list(reversed(filtered_entries[-20:]))
            
            for i, entry in enumerate(display_entries):
                analysis = entry.get("analysis", {})
                emotions = analysis.get("emotions", [])
                state = entry.get("mental_state", {}).get("state", "")
                
                # 상태에 따른 카드 스타일 결정
                if state == "안정/회복":
                    card_style = "success-card"
                elif any(keyword in state for keyword in ["스트레스", "긴장", "과부하"]):
                    card_style = "warning-card"
                else:
                    card_style = "card"
                
                # 감정 이모지
                emotion_emoji = get_emotion_emoji(emotions)
                
                with st.expander(
                    f"{emotion_emoji} {entry['date']} {entry['time']} · {', '.join(emotions)} · {state}",
                    expanded=(i == 0)  # 첫 번째만 기본 확장
                ):
                    st.markdown(f"<div class='{card_style}'>", unsafe_allow_html=True)
                    
                    # 텍스트 내용
                    st.markdown("**📝 기록 내용**")
                    st.write(entry["text"])
                    
                    # 분석 결과
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        stress_color = "🔴" if analysis.get("stress_level", 0) > 60 else ("🟡" if analysis.get("stress_level", 0) > 30 else "🟢")
                        st.write(f"**스트레스:** {stress_color} {analysis.get('stress_level', 0)}%")
                    
                    with col2:
                        energy_color = "🟢" if analysis.get("energy_level", 0) > 60 else ("🟡" if analysis.get("energy_level", 0) > 40 else "🔴")
                        st.write(f"**에너지:** {energy_color} {analysis.get('energy_level', 0)}%")
                    
                    with col3:
                        mood_score = analysis.get("mood_score", 0)
                        mood_color = "🟢" if mood_score > 10 else ("🟡" if mood_score > -10 else "🔴")
                        st.write(f"**기분:** {mood_color} {mood_score}")
                    
                    # 코칭 요약
                    mental_state = entry.get("mental_state", {})
                    if mental_state.get("summary"):
                        st.markdown("**🧠 코치 요약**")
                        st.info(mental_state["summary"])
                    
                    # 음성 분석 (있는 경우)
                    if analysis.get("voice_analysis"):
                        voice_cues = analysis["voice_analysis"]["voice_cues"]
                        st.markdown("**🎵 음성 보조지표**")
                        vc1, vc2, vc3 = st.columns(3)
                        vc1.write(f"각성: {int(voice_cues.get('arousal', 0))}")
                        vc2.write(f"긴장: {int(voice_cues.get('tension', 0))}")
                        vc3.write(f"안정: {int(voice_cues.get('stability', 0))}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)

    # =============================
    # Sidebar: Export / Reset / Additional Info
    # =============================
    with st.sidebar:
        if st.session_state.diary_entries:
            st.markdown("---")
            st.markdown("### 📁 데이터 관리")
            
            # CSV 내보내기
            if st.button("📊 CSV 내보내기"):
                rows: List[Dict] = []
                for e in st.session_state.diary_entries:
                    a = e["analysis"]
                    row = {
                        "날짜": e["date"],
                        "시간": e["time"],
                        "텍스트": e["text"],
                        "감정": ", ".join(a.get("emotions", [])),
                        "스트레스": a.get("stress_level", 0),
                        "에너지": a.get("energy_level", 0),
                        "기분": a.get("mood_score", 0),
                        "톤": a.get("tone", "중립적"),
                        "신뢰도": a.get("confidence", 0.6)
                    }
                    
                    # 정신 상태 정보 추가
                    if e.get("mental_state"):
                        ms = e["mental_state"]
                        row.update({
                            "상태": ms.get("state", ""),
                            "코치요약": ms.get("summary", ""),
                            "추천사항": " | ".join(ms.get("recommendations", []))
                        })
                    
                    # 음성 분석 정보 추가
                    if a.get("voice_analysis"):
                        v = a["voice_analysis"]
                        vc = v["voice_cues"]
                        vf = v["voice_features"]
                        row.update({
                            "각성도": vc.get("arousal", ""),
                            "긴장도": vc.get("tension", ""),
                            "안정도": vc.get("stability", ""),
                            "음질": vc.get("quality", ""),
                            "피치평균": vf.get("pitch_mean", ""),
                            "음성에너지": vf.get("energy_mean", ""),
                            "말속도": vf.get("tempo", ""),
                            "HNR": vf.get("hnr", "")
                        })
                    
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                csv = df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 다운로드",
                    csv,
                    file_name=f"voice_diary_{get_korean_time().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            # JSON 내보내기 (전체 데이터)
            if st.button("📋 JSON 내보내기"):
                export_data = {
                    "exported_at": get_korean_time().isoformat(),
                    "total_entries": len(st.session_state.diary_entries),
                    "entries": st.session_state.diary_entries,
                    "goals": st.session_state.user_goals,
                    "baseline": st.session_state.prosody_baseline
                }
                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 전체 데이터 다운로드",
                    json_str,
                    file_name=f"voice_diary_full_{get_korean_time().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            
            # 데이터 삭제
            st.markdown("---")
            if st.button("🗑️ 모든 기록 삭제", type="secondary"):
                if st.button("⚠️ 정말 삭제하시겠습니까?", type="secondary"):
                    st.session_state.diary_entries = []
                    st.session_state.user_goals = []
                    st.session_state.prosody_baseline = {}
                    st.success("모든 기록이 삭제되었습니다.")
                    st.rerun()

        # 추가 정보
        st.markdown("---")
        st.markdown("### ℹ️ 앱 정보")
        st.markdown(f"**버전:** v2.0")
        st.markdown(f"**현재 시간:** {current_time()}")
        st.markdown(f"**시간대:** 한국 표준시 (KST)")
        
        # 도움말
        with st.expander("❓ 도움말"):
            st.markdown("""
            **🎙️ 음성 녹음 팁**
            - 조용한 환경에서 녹음
            - 핸드폰을 입에서 20cm 거리
            - 2-3분 정도가 적당
            
            **📝 텍스트 입력 팁**
            - 솔직한 감정 표현
            - 구체적인 상황 포함
            - 5-10문장 정도면 충분
            
            **📊 분석 이해하기**
            - 감정 라벨: 텍스트 기반 판정
            - 음성 지표: 보조 참고 자료
            - 신뢰도: 분석 정확도 추정치
            """)

# =============================
# Footer
# =============================
if not st.session_state.show_disclaimer:
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>
            Made with ❤️ | 감정 라벨은 <strong>텍스트 우선</strong> · 목소리는 <strong>보조 지표</strong><br>
            마지막 업데이트: {get_korean_time().strftime('%Y-%m-%d %H:%M KST')} | 
            기록 수: {len(st.session_state.diary_entries)}개 | 
            목표 수: {len([g for g in st.session_state.user_goals if g.get('active', True)])}개
        </div>
        """,
        unsafe_allow_html=True,
    )
