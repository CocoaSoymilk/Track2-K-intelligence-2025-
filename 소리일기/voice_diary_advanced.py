import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import base64
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Lightweight stdlib
import numpy as np
from typing import Dict, List, Optional

# =============================
# Page / App Config
# =============================
st.set_page_config(
    page_title="소리로 쓰는 하루 - LLM 코치 버전(안정화)",
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
# Session State
# =============================
if "diary_entries" not in st.session_state:
    st.session_state.diary_entries: List[Dict] = []
if "prosody_baseline" not in st.session_state:
    st.session_state.prosody_baseline: Dict[str, float] = {}

# =============================
# Styles (minimal)
# =============================
st.markdown(
    """
    <style>
      .main-header{ text-align:center; padding:1.1rem; background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); color:#fff; border-radius:12px; margin-bottom:12px; }
      .card{ background:#fff; border:1px solid #eee; border-left:4px solid #667eea; border-radius:12px; padding:1rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); }
      .note{ color:#666; font-size:0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
      <h1>🎙️ 소리로 쓰는 하루 – LLM 코치(안정화)</h1>
      <p>감정 라벨은 텍스트가 결정하고, 목소리는 각성/긴장/안정의 <b>보조지표</b>로만 반영합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Feature Extraction
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
# Voice Cues (Arousal/Tension/Stability/Quality)
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
# Text Analysis (LLM or fallback)
# =============================

def analyze_text_with_llm(text: str, voice_cues_for_prompt: Optional[Dict] = None) -> Dict:
    if not openai_client:
        return analyze_text_simulation(text)
    cues_text = ""
    if voice_cues_for_prompt:
        cues = voice_cues_for_prompt
        cues_text = (
            f"\n(보조지표) 각성:{int(cues.get('arousal',0))}, 긴장:{int(cues.get('tension',0))}, 안정:{int(cues.get('stability',0))}, 품질:{cues.get('quality',0):.2f}"
        )
    try:
        sys = (
            "감정 라벨(기쁨/슬픔/분노/불안/평온/중립)은 텍스트로만 판단하세요. "
            "음성 지표는 수치 보조로만 참고하고 라벨을 바꾸지 마세요. JSON으로만 답하세요."
        )
        schema = (
            "{\n  \"emotions\": [..최대3개..],\n  \"stress_level\": 0-100,\n  \"energy_level\": 0-100,\n  \"mood_score\": -70~70,\n  \"summary\": \"한두 문장\",\n  \"keywords\": [..],\n  \"tone\": \"긍정적/중립적/부정적\",\n  \"confidence\": 0.0-1.0\n}"
        )
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=800,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"오늘의 이야기: {text}{cues_text}"},
                {"role": "system", "content": schema},
            ],
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            parts = content.split("```")
            data = None
            for part in parts:
                try:
                    data = json.loads(part)
                    break
                except Exception:
                    continue
            if data is None:
                data = json.loads(parts[-1])
        else:
            data = json.loads(content)
        data.setdefault("emotions", ["중립"])  
        data.setdefault("stress_level", 30)
        data.setdefault("energy_level", 50)
        data.setdefault("mood_score", 0)
        data.setdefault("summary", "일반적인 상태입니다.")
        data.setdefault("keywords", [])
        data.setdefault("tone", "중립적")
        data.setdefault("confidence", 0.7)
        return data
    except Exception:
        return analyze_text_simulation(text)


def analyze_text_simulation(text: str) -> Dict:
    t = text.lower()
    pos_kw = ["좋", "행복", "뿌듯", "기쁨", "즐겁", "평온", "만족"]
    neg_kw = ["힘들", "불안", "걱정", "짜증", "화", "우울", "슬픔"]
    pos = sum(k in t for k in pos_kw)
    neg = sum(k in t for k in neg_kw)
    if pos > neg:
        tone = "긍정적"; stress = max(10, 40 - 8 * pos); energy = min(85, 50 + 10 * pos)
    elif neg > pos:
        tone = "부정적"; stress = min(85, 40 + 10 * neg); energy = max(20, 55 - 8 * neg)
    else:
        tone = "중립적"; stress = 30; energy = 50
    mood = int(np.clip(energy - stress, -70, 70))
    emos = ["기쁨"] if tone == "긍정적" else (["슬픔"] if tone == "부정적" else ["중립"])
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
# Fusion: Text (anchor) + Voice Cues (aux)
# =============================

def combine_text_and_voice(text_analysis: Dict, voice_analysis: Optional[Dict]) -> Dict:
    if not voice_analysis or "voice_cues" not in voice_analysis:
        return text_analysis
    cues = voice_analysis["voice_cues"]
    quality = float(cues.get("quality", 0.5))
    base_alpha = 0.25 * quality
    tone = text_analysis.get("tone", "중립적")
    if tone == "긍정적":
        base_alpha *= 0.6
    elif tone == "부정적":
        base_alpha *= 0.9
    MAX_DS, MAX_DE, MAX_DM = 12, 12, 10
    stress = text_analysis.get("stress_level", 30)
    energy = text_analysis.get("energy_level", 50)
    mood = text_analysis.get("mood_score", 0)
    delta_energy = base_alpha * ((cues["arousal"] - 50) / 50.0) * 12
    delta_stress = base_alpha * (((cues["tension"] - 50) / 50.0) * 12 - ((cues["stability"] - 50) / 50.0) * 6)
    delta_mood = base_alpha * ((cues["stability"] - 50) / 50.0) * 8 - base_alpha * ((cues["tension"] - 50) / 50.0) * 6
    delta_stress = float(np.clip(delta_stress, -MAX_DS, MAX_DS))
    delta_energy = float(np.clip(delta_energy, -MAX_DE, MAX_DE))
    delta_mood = float(np.clip(delta_mood, -MAX_DM, MAX_DM))
    combined = dict(text_analysis)
    combined["stress_level"] = int(np.clip(stress + delta_stress, 0, 100))
    combined["energy_level"] = int(np.clip(energy + delta_energy, 0, 100))
    combined["mood_score"] = int(np.clip(mood + delta_mood, -70, 70))
    combined["confidence"] = float(np.clip(text_analysis.get("confidence", 0.7) + 0.12 * quality, 0, 1))
    combined["voice_analysis"] = voice_analysis
    return combined

# =============================
# Whisper Transcription (optional)
# =============================

def transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    if not openai_client:
        return None
    try:
        tmp = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t:
            t.write(audio_bytes)
            tmp = t.name
        with open(tmp, "rb") as fh:
            out = openai_client.audio.transcriptions.create(model="whisper-1", file=fh, language="ko")
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return out.text
    except Exception:
        return None

# =============================
# Baseline Update
# =============================

def update_baseline(vf: Dict):
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
# Simple Rules Coaching (fallback)
# =============================

def extract_positive_events(text: str) -> List[str]:
    t = text.lower()
    keys = [("좋았", "오늘 좋았던 점"), ("행복", "행복한 순간"), ("고마", "감사한 일"), ("즐겁", "즐거웠던 활동"), ("평온", "평온했던 순간"), ("성공", "성취"), ("뿌듯", "뿌듯했던 일")]
    tags: List[str] = []
    for k, v in keys:
        if k in t:
            tags.append(v)
    return list(dict.fromkeys(tags))[:4]


def assess_mental_state(text: str, combined: Dict) -> Dict:
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
    mot = "작은 습관이 오늘의 좋은 흐름을 내일로 이어줍니다."
    if state in ("고스트레스", "긴장 과다"):
        mot = "호흡을 고르고, 천천히. 당신의 속도로 충분합니다."
    elif state in ("저활력", "저각성"):
        mot = "작은 한 걸음이 에너지를 깨웁니다. 10분만 움직여볼까요?"
    summary = f"상태: {state} · 스트레스 {stress} · 에너지 {energy} · 각성 {int(arousal)} / 긴장 {int(tension)} / 안정 {int(stability)}"
    return {"state": state, "summary": summary, "positives": positives, "recommendations": recs, "motivation": mot, "voice_cues": {"arousal": arousal, "tension": tension, "stability": stability, "quality": quality}}

# =============================
# LLM Coach Report (primary)
# =============================

def generate_llm_coach_report(text: str, combined: Dict, recent: Optional[List[Dict]] = None) -> Dict:
    """Use LLM to produce a rich, structured coaching card. Falls back to rules if needed."""
    if not openai_client:
        return assess_mental_state(text, combined)
    cues = combined.get("voice_analysis", {}).get("voice_cues", {})
    try:
        history_blob: List[Dict] = []
        if recent:
            for e in recent[-5:]:
                a = e.get("analysis", {})
                history_blob.append({"date": e.get("date"), "tone": a.get("tone"), "stress": a.get("stress_level"), "energy": a.get("energy_level"), "mood": a.get("mood_score")})
        sys_msg = """
당신은 따뜻한 한국어 코치입니다. 텍스트는 감정 라벨의 기준이며, 음성은 각성/긴장/안정의 보조지표로만 고려하세요.
아래 정보를 종합해 JSON으로만 답하세요. 라벨(기쁨/슬픔/분노/불안/평온/중립)은 바꾸지 말고,
상태 요약과 2~4개의 구체 추천, 동기부여 한 줄을 생성하세요. 최근 기록이 있으면 짧게 반영하세요.
"""
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
        schema_hint = """
다음 JSON 스키마로만 응답:
{
  "state": "안정/회복|고스트레스|긴장 과다|과흥분/과부하 가능|저활력|저각성|중립",
  "summary": "한두 문장 요약",
  "positives": ["..."],
  "recommendations": ["간결한 실행 문장"],
  "motivation": "짧은 격려 문장"
}
"""
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=700,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                {"role": "system", "content": schema_hint},
            ],
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            parts = content.split("```")
            data = None
            for part in parts:
                try:
                    data = json.loads(part)
                    break
                except Exception:
                    continue
            if data is None:
                data = json.loads(parts[-1])
        else:
            data = json.loads(content)
        data.setdefault("state", "중립")
        data.setdefault("summary", "오늘의 상태를 차분히 정리했어요.")
        data.setdefault("positives", [])
        data.setdefault("recommendations", [])
        data.setdefault("motivation", "작은 걸음이 큰 변화를 만듭니다.")
        data["recommendations"] = data.get("recommendations", [])[:4]
        data["positives"] = data.get("positives", [])[:4]
        return data
    except Exception:
        return assess_mental_state(text, combined)

# =============================
# UI Helpers
# =============================

def today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# =============================
# Sidebar (status + nav)
# =============================
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
    page = st.selectbox("페이지", ["🎙️ 오늘의 이야기", "💖 마음 분석", "📈 감정 여정", "🎵 목소리 보조지표", "📚 나의 이야기들"])

# =============================
# Main Pages
# =============================
extractor = VoiceFeatureExtractor()

if page == "🎙️ 오늘의 이야기":
    st.header("오늘 하루는 어떠셨나요?")
    audio_val = st.audio_input("🎤 마음을 편하게 말해보세요", help="녹음 후 업로드")
    text_input = st.text_area("✏️ 글로 표현해도 좋아요", placeholder="오늘의 이야기를 적어주세요...", height=120)

    if st.button("💝 분석하고 저장", type="primary"):
        diary_text = text_input.strip()
        voice_analysis = None
        audio_b64 = None
        if audio_val is not None:
            audio_bytes = audio_val.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            with st.spinner("🎵 목소리 신호를 계산하는 중..."):
                vf = extractor.extract(audio_bytes)
                update_baseline(vf)
                voice_analysis = analyze_voice_as_cues(vf, st.session_state.prosody_baseline)
            if openai_client and not diary_text:
                with st.spinner("🤖 음성을 텍스트로 변환 중..."):
                    tx = transcribe_audio(audio_bytes)
                    if tx:
                        diary_text = tx
                        st.info(f"들은 이야기: {tx}")
        if not diary_text:
            st.warning("텍스트를 입력하거나 음성을 녹음해 주세요.")
        else:
            cues_for_prompt = voice_analysis["voice_cues"] if voice_analysis else None
            with st.spinner("🤖 텍스트 기반 감정 분석 중..."):
                t_res = analyze_text_with_llm(diary_text, cues_for_prompt)
            final = combine_text_and_voice(t_res, voice_analysis)
            recent_entries = st.session_state.diary_entries[-7:] if st.session_state.diary_entries else []
            ms_card = generate_llm_coach_report(diary_text, final, recent_entries)
            entry = {
                "id": len(st.session_state.diary_entries) + 1,
                "date": today_key(),
                "time": datetime.now().strftime("%H:%M"),
                "text": diary_text,
                "analysis": final,
                "audio_data": audio_b64,
                "mental_state": ms_card,
            }
            st.session_state.diary_entries.append(entry)
            st.success("🎉 소중한 이야기가 저장되었습니다!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("💖 감정 (텍스트 기반)")
                st.write(", ".join(final.get("emotions", [])))
                st.caption("감정 라벨은 텍스트만으로 판정합니다.")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📊 마음 상태")
                st.metric("스트레스", f"{final['stress_level']}%")
                st.metric("활력", f"{final['energy_level']}%")
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🎯 컨디션")
                st.metric("마음 점수", f"{final['mood_score']}")
                st.metric("분석 신뢰도", f"{final.get('confidence', 0.6):.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            if voice_analysis:
                st.markdown("### 🎵 목소리 신호 (보조 지표)")
                cues = final["voice_analysis"]["voice_cues"]
                qtxt = "높음" if cues["quality"] > 0.7 else ("보통" if cues["quality"] > 0.4 else "낮음")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("각성도", f"{int(cues['arousal'])}/100")
                c2.metric("긴장도", f"{int(cues['tension'])}/100")
                c3.metric("안정도", f"{int(cues['stability'])}/100")
                c4.metric("녹음 품질", qtxt)
                st.caption("※ 목소리 신호는 보조 지표입니다. 감정 판단은 텍스트에 기반합니다.")
            st.markdown("### 🧠 오늘의 마음 코치")
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write(ms_card.get("summary", "오늘의 상태를 차분히 정리했어요."))
                if ms_card.get("positives"):
                    st.write("**오늘의 밝은 포인트**: " + ", ".join(ms_card["positives"]))
                st.write("**추천**")
                for r in ms_card.get("recommendations", []):
                    st.write(f"- {r}")
                st.info(ms_card.get("motivation", "오늘도 잘 해내셨어요."))
                st.markdown("</div>", unsafe_allow_html=True)

elif page == "🎵 목소리 보조지표":
    st.header("최근 녹음의 보조지표 상세")
    entries = [e for e in st.session_state.diary_entries if e.get("analysis", {}).get("voice_analysis")]
    if not entries:
        st.info("음성으로 기록된 항목이 아직 없습니다.")
    else:
        latest = entries[-1]
        voice = latest["analysis"]["voice_analysis"]
        vf = voice["voice_features"]
        cues = voice["voice_cues"]
        qtxt = "높음" if cues["quality"] > 0.7 else ("보통" if cues["quality"] > 0.4 else "낮음")
        st.subheader("보조지표")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("각성도", f"{int(cues['arousal'])}/100")
        c2.metric("긴장도", f"{int(cues['tension'])}/100")
        c3.metric("안정도", f"{int(cues['stability'])}/100")
        c4.metric("녹음 품질", qtxt)
        st.subheader("기초 음성 특성")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("피치 평균(Hz)", f"{vf.get('pitch_mean',0):.1f}")
        d2.metric("에너지(mean)", f"{vf.get('energy_mean',0):.3f}")
        d3.metric("말하기 속도(BPM)", f"{vf.get('tempo',0):.0f}")
        d4.metric("HNR", f"{vf.get('hnr',0):.1f}")
        st.caption("※ 이 수치는 상태 단서로만 사용되며 감정 라벨을 직접 결정하지 않습니다.")

elif page == "💖 마음 분석":
    st.header("마음 분석 대시보드")
    if not st.session_state.diary_entries:
        st.info("기록이 아직 없어요.")
    else:
        df = pd.DataFrame([
            {"date": e["date"], "time": e["time"], "emotions": ", ".join(e["analysis"].get("emotions", [])), "stress": e["analysis"].get("stress_level", 0), "energy": e["analysis"].get("energy_level", 0), "mood": e["analysis"].get("mood_score", 0), "tone": e["analysis"].get("tone", "중립적"), "confidence": e["analysis"].get("confidence", 0.6)}
            for e in st.session_state.diary_entries
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "📈 감정 여정":
    st.header("시간에 따른 변화")
    if not st.session_state.diary_entries:
        st.info("기록이 쌓이면 추세를 보여드릴게요.")
    else:
        df = pd.DataFrame([
            {"dt": f"{e['date']} {e['time']}", "stress": e["analysis"].get("stress_level", 0), "energy": e["analysis"].get("energy_level", 0), "mood": e["analysis"].get("mood_score", 0)}
            for e in st.session_state.diary_entries
        ])
        st.line_chart(df.set_index("dt"))
        st.caption("※ 성능을 위해 내장 차트를 사용합니다.")

elif page == "📚 나의 이야기들":
    st.header("아카이브")
    if not st.session_state.diary_entries:
        st.info("아직 기록된 이야기가 없어요.")
    else:
        for e in reversed(st.session_state.diary_entries[-20:]):
            with st.expander(f"{e['date']} {e['time']} · {', '.join(e['analysis'].get('emotions', []))}"):
                st.write(e["text"])
                st.json({k: e["analysis"][k] for k in ["stress_level", "energy_level", "mood_score", "tone", "confidence"]})
                if e.get("mental_state"):
                    st.write("— 코치 요약: " + e["mental_state"].get("summary", ""))

# =============================
# Sidebar: Export / Reset
# =============================
with st.sidebar:
    if st.session_state.diary_entries:
        st.markdown("---")
        if st.button("📁 CSV 내보내기"):
            rows: List[Dict] = []
            for e in st.session_state.diary_entries:
                a = e["analysis"]
                row = {"date": e["date"], "time": e["time"], "text": e["text"], "emotions": ", ".join(a.get("emotions", [])), "stress": a.get("stress_level", 0), "energy": a.get("energy_level", 0), "mood": a.get("mood_score", 0), "tone": a.get("tone", "중립적"), "confidence": a.get("confidence", 0.6)}
                if e.get("mental_state"):
                    ms = e["mental_state"]
                    row.update({"state": ms.get("state", ""), "ms_summary": ms.get("summary", ""), "ms_recommendations": " | ".join(ms.get("recommendations", []))})
                if a.get("voice_analysis"):
                    v = a["voice_analysis"]; vc = v["voice_cues"]; vf = v["voice_features"]
                    row.update({"arousal": vc.get("arousal", ""), "tension": vc.get("tension", ""), "stability": vc.get("stability", ""), "quality": vc.get("quality", ""), "pitch_mean": vf.get("pitch_mean", ""), "energy_mean": vf.get("energy_mean", ""), "tempo": vf.get("tempo", ""), "hnr": vf.get("hnr", "")})
                rows.append(row)
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("다운로드", csv, file_name=f"voice_diary_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        if st.button("🗑️ 모든 기록 삭제"):
            st.session_state.diary_entries = []
            st.success("모든 기록이 삭제되었습니다.")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Made with ❤️ · 감정 라벨은 텍스트 우선 · 목소리는 각성/긴장/안정의 보조 지표로만 사용됩니다.")
