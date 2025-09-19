import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import base64
import tempfile
import warnings
warnings.filterwarnings("ignore")

# --- Lightweight imports first (heavy libs are lazy-imported) ---
import numpy as np
from typing import Dict, List, Optional

# =============================
# Page / App Config
# =============================
st.set_page_config(
    page_title="소리로 쓰는 하루 - 차원형 보조지표 버전",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Utility: Lazy Importers (avoid heavy import unless needed)
# =============================
@st.cache_resource(show_spinner=False)
def get_librosa():
    try:
        import librosa  # type: ignore
        return librosa
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_scipy():
    try:
        import scipy  # type: ignore
        from scipy import signal, stats  # noqa: F401
        return scipy
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
        # OpenAI Python SDK v1 style client
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
# Global Session State
# =============================
if "diary_entries" not in st.session_state:
    st.session_state.diary_entries: List[Dict] = []

# 개인 기준선(초기 3~5회 평균) 저장 공간
if "prosody_baseline" not in st.session_state:
    st.session_state.prosody_baseline: Dict[str, float] = {}

# =============================
# Styles (minimal; keep light for perf)
# =============================
st.markdown(
    """
    <style>
      .main-header{ text-align:center; padding:1.2rem; background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); color:#fff; border-radius:10px; margin-bottom:1rem; }
      .metric-card{ background:#fff; border:1px solid #eee; border-left:4px solid #667eea; border-radius:12px; padding:1rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); }
      .note{ color:#666; font-size:0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
      <h1>🎙️ 소리로 쓰는 하루 – 보조지표(차원형) 기반</h1>
      <p>목소리는 감정 라벨을 ‘결정’하지 않고, 각성/긴장/안정의 <b>보조 단서</b>로만 반영합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Feature Extraction (Optimized)
# =============================
class VoiceFeatureExtractor:
    """경량/안정 위주 Prosody Feature Extractor (librosa 우선, Praat 선택)
    - Lazy import 사용
    - 필수 최소 feature만 계산
    - 짧은 오디오/저품질 오디오는 품질 낮게 처리
    """

    def __init__(self, target_sr: int = 22050):
        self.sample_rate = target_sr

    def _load_audio(self, audio_bytes: bytes):
        librosa = get_librosa()
        if not librosa:
            return None, None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        try:
            y, sr = librosa.load(path, sr=self.sample_rate, mono=True, res_type="kaiser_fast")
            return y, sr
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    @st.cache_data(show_spinner=False)
    def _hann(self):
        # small cached window if ever needed later
        return np.hanning(2048)

    def extract(self, audio_bytes: bytes) -> Dict:
        librosa = get_librosa()
        if not librosa:
            return self._default_features()

        try:
            y, sr = self._load_audio(audio_bytes)
            if y is None or y.size == 0:
                return self._default_features()

            duration_sec = max(0.001, float(len(y) / sr))

            # --- Cheap/robust features ---
            # RMS energy (frame-wise -> mean/max)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            energy_mean = float(np.mean(rms))
            energy_max = float(np.max(rms))

            # Tempo (coarse) – librosa.beat is relatively heavy; guard by duration
            if duration_sec >= 2.5:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            else:
                tempo = 0.0

            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)[0]
            zcr_mean = float(np.mean(zcr))

            # Spectral centroid (coarse brightness)
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = float(np.mean(sc))

            # Pitch proxy (cheap): use piptrack sparsely and choose max per frame
            pitch_mean = 0.0
            pitch_var = 0.0
            try:
                if duration_sec >= 1.0:
                    pitches, mags = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=400)
                    valid = []
                    for t in range(pitches.shape[1]):
                        idx = np.argmax(mags[:, t])
                        p = pitches[idx, t]
                        if p > 0:
                            valid.append(p)
                    if valid:
                        va = np.array(valid, dtype=float)
                        pitch_mean = float(np.mean(va))
                        pitch_var = float(np.std(va) / (np.mean(va) + 1e-6))
            except Exception:
                pass

            # Simple phonation quality proxies (Praat optional)
            parselmouth = get_parselmouth()
            hnr = 15.0
            jitter = 0.012
            if parselmouth:
                try:
                    # write temp wav once more (avoid reusing deleted)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
                        tmp2.write(audio_bytes)
                        p2 = tmp2.name
                    snd = parselmouth.Sound(p2)
                    # harmonicity (HNR)
                    harm = snd.to_harmonicity_cc()
                    hnr = float(np.nan_to_num(harm.values.mean(), nan=15.0))
                    # jitter local (approx) via PointProcess
                    point_proc = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
                    jitter = float(parselmouth.praat.call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
                except Exception:
                    pass
                finally:
                    try:
                        os.unlink(p2)
                    except Exception:
                        pass

            return {
                "duration_sec": duration_sec,
                "pitch_mean": float(pitch_mean if pitch_mean > 0 else 150.0),
                "pitch_variation": float(pitch_var if pitch_var > 0 else 0.13),
                "energy_mean": float(energy_mean),
                "energy_max": float(energy_max),
                "tempo": float(tempo if tempo > 0 else 110.0),
                "zcr_mean": float(zcr_mean),
                "spectral_centroid_mean": float(spectral_centroid_mean),
                "hnr": float(hnr),
                "jitter": float(jitter),
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
# Dimensional Voice Cues (Arousal/Tension/Stability + Quality)
# =============================

def prosody_to_dimensions(f: Dict, baseline: Optional[Dict] = None) -> Dict:
    def norm(key: str, val: float) -> float:
        if not baseline or key not in baseline:
            return val
        b = baseline[key]
        return (val / b) if b else val

    pitch = norm("pitch_mean", f.get("pitch_mean", 150.0))  # noqa: F841 (reserved if needed later)
    tempo = norm("tempo", f.get("tempo", 110.0))
    energy = norm("energy_mean", f.get("energy_mean", 0.08))
    hnr = norm("hnr", f.get("hnr", 15.0))
    jitter = f.get("jitter", 0.012)
    zcr = f.get("zcr_mean", 0.10)
    sc = norm("spectral_centroid_mean", f.get("spectral_centroid_mean", 2000.0))

    # Scaled 0~100 dimensional cues (empirical scaling)
    arousal = np.clip(35 + 120 * energy + 0.06 * (tempo - 110) + 0.004 * (sc - 2000), 0, 100)
    tension = np.clip(28 + 120 * jitter + 0.55 * (zcr - 0.10) * 100, 0, 100)
    stability = np.clip(60 + 1.3 * (hnr - 15) - 85 * jitter, 0, 100)

    duration = f.get("duration_sec", 4.0)
    quality = np.clip(
        0.28 * (duration / 8.0) + 0.42 * np.clip((hnr - 10) / 15, 0, 1) + 0.30 * np.clip((energy - 0.06) / 0.20, 0, 1),
        0,
        1,
    )

    return {
        "arousal": float(arousal),
        "tension": float(tension),
        "stability": float(stability),
        "quality": float(quality),
    }


def analyze_voice_as_cues(voice_features: Dict, baseline: Optional[Dict] = None) -> Dict:
    dims = prosody_to_dimensions(voice_features, baseline)
    return {
        "voice_cues": dims,  # arousal/tension/stability/quality
        "voice_features": voice_features,
    }

# =============================
# Text Analysis (LLM or fallback)
# =============================

def analyze_text_with_llm(text: str, voice_cues_for_prompt: Optional[Dict] = None) -> Dict:
    """LLM 사용 시: 감정 라벨은 텍스트로만 판단. 음성은 보조지표로만 활용하도록 지시."""
    if not openai_client:
        return analyze_text_simulation(text)

    cues_text = ""
    if voice_cues_for_prompt:
        cues = voice_cues_for_prompt
        cues_text = (
            f"\n(참고용 보조지표) 각성:{int(cues.get('arousal',0))}, "
            f"긴장:{int(cues.get('tension',0))}, 안정:{int(cues.get('stability',0))}, 품질:{cues.get('quality',0):.2f}"
        )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=800,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 차분한 마음 분석가입니다. 감정 라벨(기쁨/슬픔/분노/불안/평온/중립)은 오직 텍스트로만 판단하세요. "
                        "음성 관련 정보는 각성/긴장/안정의 보조지표로만 참고하며 라벨을 바꾸지 마세요.\n"
                        "JSON으로만 응답: {\n"
                        "  \"emotions\": [..최대3개..],\n"
                        "  \"stress_level\": 0-100,\n"
                        "  \"energy_level\": 0-100,\n"
                        "  \"mood_score\": -70~70,\n"
                        "  \"summary\": \"한두 문장\",\n"
                        "  \"keywords\": [..],\n"
                        "  \"tone\": \"긍정적/중립적/부정적\",\n"
                        "  \"confidence\": 0.0-1.0\n"
                        "}"
                    ),
                },
                {"role": "user", "content": f"오늘의 이야기: {text}{cues_text}"},
            ],
        )
        content = resp.choices[0].message.content.strip()
        # Try to find JSON
        if "```" in content:
            content = content.split("```")[-2]
        result = json.loads(content)
        # guard defaults
        result.setdefault("emotions", ["중립"])  
        result.setdefault("stress_level", 30)
        result.setdefault("energy_level", 50)
        result.setdefault("mood_score", 0)
        result.setdefault("summary", "일반적인 상태입니다.")
        result.setdefault("keywords", [])
        result.setdefault("tone", "중립적")
        result.setdefault("confidence", 0.7)
        return result
    except Exception:
        return analyze_text_simulation(text)


def analyze_text_simulation(text: str) -> Dict:
    text_l = text.lower()
    pos_kw = ["좋", "행복", "뿌듯", "기쁨", "즐겁", "평온", "만족"]
    neg_kw = ["힘들", "불안", "걱정", "짜증", "화", "우울", "슬픔"]

    pos = sum(k in text_l for k in pos_kw)
    neg = sum(k in text_l for k in neg_kw)

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

    # base impact scaled by quality and tone
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            p = tmp.name
        with open(p, "rb") as fh:
            out = openai_client.audio.transcriptions.create(model="whisper-1", file=fh, language="ko")
        try:
            os.unlink(p)
        except Exception:
            pass
        return out.text
    except Exception:
        return None

# =============================
# Baseline Update (fast and simple)
# =============================

def update_baseline(vf: Dict):
    keys = ["pitch_mean", "tempo", "energy_mean", "hnr", "spectral_centroid_mean"]
    b = st.session_state.prosody_baseline
    # keep running mean with count (store count inside baseline)
    count = int(b.get("_count", 0))
    new_count = min(20, count + 1)  # cap to avoid drift
    alpha = 1.0 / new_count
    for k in keys:
        v = float(vf.get(k, 0.0))
        prev = float(b.get(k, v))
        b[k] = (1 - alpha) * prev + alpha * v
    b["_count"] = new_count

# =============================
# Sidebar: System Status & Nav
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
    page = st.selectbox(
        "페이지",
        ["🎙️ 오늘의 이야기", "💖 마음 분석", "📈 감정 여정", "🎵 목소리 보조지표", "📚 나의 이야기들"],
    )

# =============================
# Main Pages
# =============================
extractor = VoiceFeatureExtractor()

def today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")

if page == "🎙️ 오늘의 이야기":
    st.header("오늘 하루는 어떠셨나요?")

    # Input widgets
    audio_val = st.audio_input("🎤 마음을 편하게 말해보세요", help="녹음 후 업로드")
    text_input = st.text_area(
        "✏️ 글로 표현해도 좋아요",
        placeholder="오늘의 이야기를 적어주세요...",
        height=120,
    )

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

            # Whisper (optional, fast-fail)
            if openai_client and not diary_text:
                with st.spinner("🤖 음성을 텍스트로 변환 중..."):
                    tx = transcribe_audio(audio_bytes)
                    if tx:
                        diary_text = tx
                        st.info(f"들은 이야기: {tx}")

        if not diary_text:
            st.warning("텍스트를 입력하거나 음성을 녹음해 주세요.")
        else:
            # Text analysis first (anchor). Pass cues only for context string, not to change labels.
            cues_for_prompt = voice_analysis["voice_cues"] if voice_analysis else None
            with st.spinner("🤖 텍스트 기반 감정 분석 중..."):
                t_res = analyze_text_with_llm(diary_text, cues_for_prompt)

            # Fuse with voice cues (aux)
            final = combine_text_and_voice(t_res, voice_analysis)

            entry = {
                "id": len(st.session_state.diary_entries) + 1,
                "date": today_key(),
                "time": datetime.now().strftime("%H:%M"),
                "text": diary_text,
                "analysis": final,
                "audio_data": audio_b64,
            }
            st.session_state.diary_entries.append(entry)

            st.success("🎉 소중한 이야기가 저장되었습니다!")

            # Quick summary cards (lightweight UI)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("💖 감정 (텍스트 기반)")
                st.write(", ".join(final.get("emotions", [])))
                st.caption("감정 라벨은 텍스트만으로 판정합니다.")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("📊 마음 상태")
                st.metric("스트레스", f"{final['stress_level']}%")
                st.metric("활력", f"{final['energy_level']}%")
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("🎯 컨디션")
                st.metric("마음 점수", f"{final['mood_score']}")
                conf = final.get("confidence", 0.6)
                st.metric("분석 신뢰도", f"{conf:.2f}")
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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("피치 평균(Hz)", f"{vf.get('pitch_mean',0):.1f}")
        c2.metric("에너지(mean)", f"{vf.get('energy_mean',0):.3f}")
        c3.metric("말하기 속도(BPM)", f"{vf.get('tempo',0):.0f}")
        c4.metric("HNR", f"{vf.get('hnr',0):.1f}")
        st.caption("※ 이 수치는 상태 단서로만 사용되며 감정 라벨을 직접 결정하지 않습니다.")

elif page == "💖 마음 분석":
    st.header("마음 분석 대시보드")
    if not st.session_state.diary_entries:
        st.info("기록이 아직 없어요.")
    else:
        df = pd.DataFrame([
            {
                "date": e["date"],
                "time": e["time"],
                "emotions": ", ".join(e["analysis"].get("emotions", [])),
                "stress": e["analysis"].get("stress_level", 0),
                "energy": e["analysis"].get("energy_level", 0),
                "mood": e["analysis"].get("mood_score", 0),
                "tone": e["analysis"].get("tone", "중립적"),
                "confidence": e["analysis"].get("confidence", 0.6),
            }
            for e in st.session_state.diary_entries
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "📈 감정 여정":
    st.header("시간에 따른 변화")
    if not st.session_state.diary_entries:
        st.info("기록이 쌓이면 추세를 보여드릴게요.")
    else:
        df = pd.DataFrame([
            {
                "dt": f"{e['date']} {e['time']}",
                "stress": e["analysis"].get("stress_level", 0),
                "energy": e["analysis"].get("energy_level", 0),
                "mood": e["analysis"].get("mood_score", 0),
            }
            for e in st.session_state.diary_entries
        ])
        st.line_chart(df.set_index("dt"))
        st.caption("※ 간단한 기본 차트로 표시합니다 (성능을 위해 내장 차트 사용).")

elif page == "📚 나의 이야기들":
    st.header("아카이브")
    if not st.session_state.diary_entries:
        st.info("아직 기록된 이야기가 없어요.")
    else:
        for e in reversed(st.session_state.diary_entries[-20:]):  # 최근 20개만 렌더 (퍼포먼스)
            with st.expander(f"{e['date']} {e['time']} · {', '.join(e['analysis'].get('emotions', []))}"):
                st.write(e["text"])
                st.json({k: e["analysis"][k] for k in ["stress_level", "energy_level", "mood_score", "tone", "confidence"]})

# =============================
# Sidebar: Export / Backup / Reset
# =============================
with st.sidebar:
    if st.session_state.diary_entries:
        st.markdown("---")
        if st.button("📁 CSV 내보내기"):
            rows = []
            for e in st.session_state.diary_entries:
                a = e["analysis"]
                row = {
                    "date": e["date"],
                    "time": e["time"],
                    "text": e["text"],
                    "emotions": ", ".join(a.get("emotions", [])),
                    "stress": a.get("stress_level", 0),
                    "energy": a.get("energy_level", 0),
                    "mood": a.get("mood_score", 0),
                    "tone": a.get("tone", "중립적"),
                    "confidence": a.get("confidence", 0.6),
                }
                if a.get("voice_analysis"):
                    v = a["voice_analysis"]
                    vc = v["voice_cues"]
                    vf = v["voice_features"]
                    row.update(
                        {
                            "arousal": vc.get("arousal", ""),
                            "tension": vc.get("tension", ""),
                            "stability": vc.get("stability", ""),
                            "quality": vc.get("quality", ""),
                            "pitch_mean": vf.get("pitch_mean", ""),
                            "energy_mean": vf.get("energy_mean", ""),
                            "tempo": vf.get("tempo", ""),
                            "hnr": vf.get("hnr", ""),
                        }
                    )
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
