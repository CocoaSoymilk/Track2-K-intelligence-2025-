# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz, io, os, json, base64, tempfile, hashlib, random, calendar, warnings
warnings.filterwarnings("ignore")

# =============================
# Optional Heavy Imports (lazy)
# =============================
@st.cache_resource(show_spinner=False)
def get_librosa():
    try:
        import librosa
        return librosa
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_parselmouth():
    try:
        import parselmouth
        from parselmouth.praat import call  # noqa
        return parselmouth
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_soundfile():
    try:
        import soundfile as sf
        return sf
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_webrtcvad():
    try:
        import webrtcvad
        return webrtcvad
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_pypdf2():
    try:
        import PyPDF2
        return PyPDF2
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_openai_client():
    # OpenAI v1 SDK-style client
    try:
        import openai
        if "OPENAI_API_KEY" in st.secrets:
            return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        elif "openai_api_key" in st.session_state and st.session_state.openai_api_key:
            return openai.OpenAI(api_key=st.session_state.openai_api_key)
        else:
            return None
    except Exception:
        return None

openai_client = get_openai_client()
librosa = get_librosa()
parselmouth = get_parselmouth()
sf = get_soundfile()
webrtcvad = get_webrtcvad()
PyPDF2 = get_pypdf2()

# =============================
# Timezone / Keys
# =============================
KST = pytz.timezone("Asia/Seoul")
def kst_now(): return datetime.now(KST)
def today_key(): return kst_now().strftime("%Y-%m-%d")
def current_time(): return kst_now().strftime("%H:%M")

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="소리로 쓰는 하루 - AI 감정 코치",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Session State
# =============================
def init_ss():
    ss = st.session_state
    ss.setdefault("diary_entries", [])
    ss.setdefault("prosody_baseline", {})
    ss.setdefault("user_goals", [])
    ss.setdefault("show_disclaimer", True)
    ss.setdefault("onboarding_completed", False)
    ss.setdefault("demo_data_loaded", False)
    ss.setdefault("kb_index", None)
    ss.setdefault("kb_meta", None)
    ss.setdefault("kb_ready", False)
    ss.setdefault("show_weekly_report", False)
    ss.setdefault("weekly_report", None)
    ss.setdefault("openai_api_key", "")
init_ss()

# =============================
# Styles
# =============================
st.markdown("""
<style>
  .main-header{ text-align:center; padding:1.5rem; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:#fff; border-radius:15px; margin-bottom:20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);}
  .card{ background:#fff; border:1px solid #e0e6ed; border-left:4px solid #667eea; border-radius:12px; padding:1.2rem; box-shadow:0 2px 10px rgba(0,0,0,0.05); margin-bottom:1rem;}
  .success-card{ background:linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-left:4px solid #28a745;}
  .warning-card{ background:linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border-left:4px solid #ffc107;}
  .disclaimer-banner{ background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-left: 4px solid #2196f3; padding: 1rem; border-radius: 8px; margin: 1rem 0;}
  .metric-positive { color: #28a745; font-weight: bold; }
  .metric-negative { color: #dc3545; font-weight: bold; }
  .metric-neutral { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================
# Disclaimer
# =============================
def show_disclaimer():
    if st.session_state.show_disclaimer:
        st.markdown("""
        <div class="disclaimer-banner">
            <h4>🛡️ 서비스 이용 안내</h4>
            <ul>
                <li><strong>의료적 한계:</strong> 본 서비스는 자기 성찰 보조 도구이며, 의료적 진단/치료가 아닙니다.</li>
                <li><strong>데이터 보안:</strong> 모든 기록은 세션에만 저장, 브라우저 종료 시 삭제됩니다.</li>
                <li><strong>AI 한계:</strong> 분석 결과는 참고용입니다. 최종 판단은 사용자에게 있습니다.</li>
                <li><strong>긴급상황:</strong> 심각한 정신건강 이슈는 반드시 전문가와 상담하세요.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 이해했습니다", type="primary"):
                st.session_state.show_disclaimer = False
                st.rerun()
        with c2:
            if st.button("📊 데모 데이터로 시작하기"):
                load_demo_data()
                st.session_state.show_disclaimer = False
                st.success("가상 데이터가 로드되었습니다!")
                st.rerun()

# =============================
# Demo Data
# =============================
def load_demo_data():
    if st.session_state.demo_data_loaded: return
    scenarios = [
        {"t":"오늘은 중간고사 마지막 날이었어요. 수학 시험이 정말 어려웠는데 그래도 나름 준비한 만큼은 쓴 것 같아요. 친구들이랑 치킨 먹으면서 스트레스 풀었어요.",
         "e":["기쁨","평온"],"S":35,"E":70,"M":25,"tone":"긍정적"},
        {"t":"좋아하는 친구에게 고백할까 고민이 많아요. 거절이 두려워 불안하지만 설레요.",
         "e":["불안","설렘"],"S":65,"E":50,"M":-5,"tone":"중립적"},
        {"t":"과제 마감이 내일인데 아직 반도 못했어요. 밤새워야 할 듯. 컨디션이 안 좋아요.",
         "e":["스트레스","피로"],"S":85,"E":25,"M":-35,"tone":"부정적"},
        {"t":"고백했는데 담담히 받아줬어요. 연인은 아니지만 가까워질 수 있을 것 같대요. 마음이 후련해요.",
         "e":["기쁨","만족"],"S":40,"E":75,"M":30,"tone":"긍정적"},
        {"t":"평범한 하루. 수업 듣고 공부하고 넷플릭스. 바빴던 요즘에 평온함이 좋네요.",
         "e":["평온","만족"],"S":20,"E":60,"M":15,"tone":"긍정적"},
        {"t":"팀 프로젝트 조원 한 명이 잠수. 발표가 다음 주라 당황. 스트레스가 큽니다.",
         "e":["분노","스트레스"],"S":90,"E":40,"M":-40,"tone":"부정적"},
        {"t":"친구들이랑 MT 다녀왔어요! 밤새 이야기하며 행복했어요.",
         "e":["기쁨","행복"],"S":15,"E":85,"M":45,"tone":"긍정적"},
    ]
    base = kst_now() - timedelta(days=6)
    for i, s in enumerate(scenarios):
        d = (base + timedelta(days=i))
        entry = {
            "id": i+1,
            "date": d.strftime("%Y-%m-%d"),
            "time": f"{random.randint(18,22):02d}:{random.randint(0,59):02d}",
            "text": s["t"],
            "analysis": {
                "emotions": s["e"],
                "stress_level": s["S"],
                "energy_level": s["E"],
                "mood_score": s["M"],
                "summary": f"{s['tone']} 상태의 하루.",
                "keywords": [],
                "tone": s["tone"],
                "confidence": round(random.uniform(0.7,0.9),2)
            },
            "audio_data": None,
            "mental_state": {
                "state": "안정/회복" if s["S"]<40 else ("고스트레스" if s["S"]>70 else "중립"),
                "summary": f"스트레스 {s['S']}%, 에너지 {s['E']}%.",
                "positives": ["친구와의 시간","성취"] if s["tone"]=="긍정적" else [],
                "recommendations": ["휴식","친구와 시간","규칙적 생활"],
                "motivation": "하루하루 최선을 다해요!"
            }
        }
        st.session_state.diary_entries.append(entry)
    st.session_state.user_goals = [
        {"id":1,"type":"stress","target":50,"description":"스트레스 50 이하 유지","created_date":(kst_now()-timedelta(days=5)).strftime("%Y-%m-%d"),"active":True},
        {"id":2,"type":"consistency","target":5,"description":"주 5회 이상 기록","created_date":(kst_now()-timedelta(days=5)).strftime("%Y-%m-%d"),"active":True},
    ]
    st.session_state.demo_data_loaded = True

# =============================
# Header
# =============================
def header():
    if not st.session_state.show_disclaimer:
        st.markdown(f"""
        <div class="main-header">
          <h1>🎙️ 소리로 쓰는 하루 – AI 감정 코치</h1>
          <p>📅 {kst_now().strftime('%Y년 %m월 %d일 %A')} | ⏰ {current_time()}</p>
          <p>감정 라벨은 <b>텍스트 기반</b>, 목소리는 <b>보조 지표</b>로만 사용합니다</p>
        </div>
        """, unsafe_allow_html=True)

# =============================
# Audio: Feature Extractor
# =============================
class VoiceFeatureExtractor:
    def __init__(self, target_sr=22050): self.sample_rate=target_sr

    def _load_audio(self, audio_bytes:bytes):
        if not librosa: return None, None
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate, mono=True)
            return y, sr
        except Exception:
            return None, None

    def extract(self, audio_bytes:bytes):
        if not librosa:
            return self._default()
        try:
            y, sr = self._load_audio(audio_bytes)
            if y is None or y.size==0 or sr is None: return self._default()
            dur = max(0.001, len(y)/sr)

            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            energy_mean = float(np.mean(rms)); energy_max = float(np.max(rms))

            tempo = 110.0
            if dur>=2.5:
                tempo,_ = librosa.beat.beat_track(y=y, sr=sr)

            zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)[0]
            zcr_mean = float(np.mean(zcr))

            sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = float(np.mean(sc))

            pitch_mean=150.0; pitch_var=0.13
            try:
                if dur>=1.0:
                    pitches, mags = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=400)
                    valid=[]
                    for t in range(pitches.shape[1]):
                        idx=int(np.argmax(mags[:,t])); p=float(pitches[idx,t])
                        if p>0: valid.append(p)
                    if valid:
                        va=np.array(valid,dtype=float)
                        pitch_mean=float(np.mean(va)); pitch_var=float(np.std(va)/(np.mean(va)+1e-6))
            except Exception: pass

            hnr=15.0; jitter=0.012
            if parselmouth:
                tmp=None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t2:
                        if sf: sf.write(t2, y, sr, subtype="PCM_16", format="WAV")
                        else:
                            # fallback write via numpy (no header) – skip parselmouth if no soundfile
                            raise RuntimeError("soundfile not available")
                        tmp=t2.name
                    snd=parselmouth.Sound(tmp)
                    harm=snd.to_harmonicity_cc()
                    hnr=float(np.nan_to_num(harm.values.mean(), nan=15.0))
                    pp=parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
                    jitter=float(parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
                except Exception: pass
                finally:
                    if tmp and os.path.exists(tmp):
                        try: os.unlink(tmp)
                        except Exception: pass

            return {
                "duration_sec":float(dur),
                "pitch_mean":float(pitch_mean),
                "pitch_variation":float(pitch_var),
                "energy_mean":float(energy_mean),
                "energy_max":float(energy_max),
                "tempo":float(tempo),
                "zcr_mean":float(zcr_mean),
                "spectral_centroid_mean":float(spectral_centroid_mean),
                "hnr":float(hnr),
                "jitter":float(jitter),
            }
        except Exception:
            return self._default()

    def _default(self):
        return {"duration_sec":0.0,"pitch_mean":150.0,"pitch_variation":0.13,
                "energy_mean":0.08,"energy_max":0.12,"tempo":110.0,
                "zcr_mean":0.10,"spectral_centroid_mean":2000.0,"hnr":15.0,"jitter":0.012}

# =============================
# Prosody → Cues
# =============================
def prosody_to_dimensions(f, baseline=None):
    def norm(k,v):
        if not baseline or k not in baseline: return v
        b=float(baseline.get(k,0.0)); return (v/b) if b else v
    tempo=norm("tempo", float(f.get("tempo",110.0)))
    energy=norm("energy_mean", float(f.get("energy_mean",0.08)))
    hnr=norm("hnr", float(f.get("hnr",15.0)))
    jitter=float(f.get("jitter",0.012)); zcr=float(f.get("zcr_mean",0.10))
    sc=norm("spectral_centroid_mean", float(f.get("spectral_centroid_mean",2000.0)))
    arousal=float(np.clip(35 + 120*energy + 0.06*(tempo-110) + 0.004*(sc-2000),0,100))
    tension=float(np.clip(28 + 120*jitter + 0.55*(zcr-0.10)*100,0,100))
    stability=float(np.clip(60 + 1.3*(hnr-15) - 85*jitter,0,100))
    duration=float(f.get("duration_sec",4.0))
    quality=float(np.clip(0.28*(duration/8.0) + 0.42*np.clip((hnr-10)/15,0,1) + 0.30*np.clip((energy-0.06)/0.20,0,1),0,1))
    return {"arousal":arousal,"tension":tension,"stability":stability,"quality":quality}

def analyze_voice_as_cues(vf, baseline=None):
    return {"voice_cues":prosody_to_dimensions(vf, baseline),"voice_features":vf}

def update_baseline(vf):
    keys=["pitch_mean","tempo","energy_mean","hnr","spectral_centroid_mean"]
    b=st.session_state.prosody_baseline; cnt=int(b.get("_count",0)); new_cnt=min(20,cnt+1); alpha=1.0/new_cnt
    for k in keys:
        v=float(vf.get(k,0.0)); prev=float(b.get(k,v)); b[k]=(1-alpha)*prev + alpha*v
    b["_count"]=new_cnt

# =============================
# Text Analysis (LLM 1차) – Prompt Engineering 강화
# =============================
TEXT_ANALYZER_SYSTEM = (
    "당신은 한국어 감정 분석가입니다. 다음 규칙을 준수하세요.\n"
    "1) 감정 라벨(emotions)은 오직 텍스트 내용만으로 판단합니다(보이스 지표는 참고용일 뿐 라벨 변경 금지).\n"
    "2) 허용 라벨: 기쁨/슬픔/분노/불안/평온/중립. 최대 2개.\n"
    "3) 의료적 진단/약물/자해/위험 판단을 하지 않습니다. 필요 시 일반적 전문가 상담 권고만 가능합니다.\n"
    "4) 아래 JSON 스키마로만 응답하세요(추가 텍스트 금지)."
)

def safe_json_parse(content:str)->dict:
    if not content: return {}
    s=content.strip()
    if s.startswith("```"): s="\n".join(s.split("\n")[1:-1])
    if s.startswith("```json"): s=s[7:]
    if s.endswith("```"): s=s[:-3]
    s=s.strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            i=s.find("{"); j=s.rfind("}")+1
            if i>=0 and j>i: return json.loads(s[i:j])
        except Exception: return {}
        return {}

def call_llm_safely(fn, *args, **kwargs):
    import time
    for i in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            time.sleep(0.6*(2**i) + random.random()*0.3)
    return None

def analyze_text_with_llm(text:str, voice_cues_for_prompt=None)->dict:
    if not openai_client or not text.strip():
        return analyze_text_simulation(text)
    cues_text=""
    if voice_cues_for_prompt:
        c=voice_cues_for_prompt
        cues_text = f"(보조지표) 각성:{int(c.get('arousal',0))}, 긴장:{int(c.get('tension',0))}, 안정:{int(c.get('stability',0))}, 품질:{float(c.get('quality',0)):.2f}"
    user_prompt = (
        "다음 일기를 분석해 JSON으로만 응답하세요. 스키마:\n"
        '{"emotions":["감정1","감정2"],"stress_level":0,"energy_level":0,"mood_score":0,'
        '"summary":"요약","keywords":["키워드"],"tone":"긍정적/중립적/부정적","confidence":0.0}\n'
        f"일기: {text}\n{cues_text}"
    )
    def _call():
        return openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=500,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":TEXT_ANALYZER_SYSTEM},
                {"role":"user","content":user_prompt}
            ]
        )
    resp = call_llm_safely(_call)
    if not resp or not resp.choices or not resp.choices[0].message.content:
        return analyze_text_simulation(text)
    data = safe_json_parse(resp.choices[0].message.content)
    if not data: return analyze_text_simulation(text)
    data.setdefault("emotions",["중립"]); data.setdefault("stress_level",30)
    data.setdefault("energy_level",50); data.setdefault("mood_score",0)
    data.setdefault("summary","일반적인 상태입니다."); data.setdefault("keywords",[])
    data.setdefault("tone","중립적"); data.setdefault("confidence",0.7)
    # Boundaries
    data["stress_level"]=int(np.clip(data["stress_level"],0,100))
    data["energy_level"]=int(np.clip(data["energy_level"],0,100))
    data["mood_score"]=int(np.clip(data["mood_score"],-70,70))
    data["emotions"]=data["emotions"][:2]
    return data

def analyze_text_simulation(text:str)->dict:
    t=text.lower()
    pos_kw=["좋","행복","뿌듯","기쁨","즐겁","평온","만족","감사","성공","좋아"]
    neg_kw=["힘들","불안","걱정","짜증","화","우울","슬픔","스트레스","피곤","어려"]
    pos=sum(k in t for k in pos_kw); neg=sum(k in t for k in neg_kw)
    if pos>neg:
        tone="긍정적"; stress=max(10,40-8*pos); energy=min(85,50+10*pos); emos=["기쁨"]
    elif neg>pos:
        tone="부정적"; stress=min(85,40+10*neg); energy=max(20,55-8*neg); emos=["슬픔"]
    else:
        tone="중립적"; stress=30; energy=50; emos=["중립"]
    mood=int(np.clip(energy-stress,-70,70))
    return {"emotions":emos,"stress_level":int(stress),"energy_level":int(energy),"mood_score":mood,
            "summary":f"{tone} 상태로 보입니다.","keywords":[],"tone":tone,"confidence":0.55}

# =============================
# Fusion (voice as auxiliary)
# =============================
def combine_text_and_voice(text_analysis, voice_analysis=None):
    if not voice_analysis or "voice_cues" not in voice_analysis:
        return text_analysis
    cues = voice_analysis["voice_cues"]; quality=float(cues.get("quality",0.5))
    base_alpha = 0.25*quality
    # 품질 낮으면 보정 0
    if quality < 0.3: base_alpha = 0.0
    tone=text_analysis.get("tone","중립적")
    if tone=="긍정적": base_alpha*=0.6
    elif tone=="부정적": base_alpha*=0.9
    MAX_DS,MAX_DE,MAX_DM = 12,12,10
    stress=text_analysis.get("stress_level",30)
    energy=text_analysis.get("energy_level",50)
    mood=text_analysis.get("mood_score",0)
    delta_energy = base_alpha * ((cues["arousal"]-50)/50.0)*12
    delta_stress = base_alpha * (((cues["tension"]-50)/50.0)*12 - ((cues["stability"]-50)/50.0)*6)
    delta_mood   = base_alpha * ((cues["stability"]-50)/50.0)*8 - base_alpha*((cues["tension"]-50)/50.0)*6
    delta_stress=float(np.clip(delta_stress,-MAX_DS,MAX_DS))
    delta_energy=float(np.clip(delta_energy,-MAX_DE,MAX_DE))
    delta_mood=float(np.clip(delta_mood,-MAX_DM,MAX_DM))
    combined=dict(text_analysis)
    combined["stress_level"]=int(np.clip(stress+delta_stress,0,100))
    combined["energy_level"]=int(np.clip(energy+delta_energy,0,100))
    combined["mood_score"]=int(np.clip(mood+delta_mood,-70,70))
    combined["confidence"]=float(np.clip(text_analysis.get("confidence",0.7)+0.12*quality,0,1))
    combined["voice_analysis"]=voice_analysis
    return combined

# =============================
# ASR: Preprocess + Prompting + Postprocess
# =============================
def build_initial_prompt_from_history(entries, limit=10):
    kws=[]
    for e in entries[-limit:]:
        a=e.get("analysis",{})
        kws += a.get("keywords",[])
    uniq=[k for k in dict.fromkeys(kws) if 1<len(k)<=15]
    return ("키워드: " + ", ".join(uniq[:15])) if uniq else ""

def preprocess_audio_for_asr(audio_bytes:bytes, target_sr=16000)->bytes:
    # Minimal-safe preprocessing pipeline; works even if some deps missing
    if not librosa or not sf:
        return audio_bytes
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
    if y.size==0: return audio_bytes
    # pre-emphasis
    try:
        y = librosa.effects.preemphasis(y, coef=0.97)
    except Exception:
        pass
    # Simple VAD using webrtcvad if available
    if webrtcvad:
        try:
            vad = webrtcvad.Vad(2)
            frame_ms=30; frame_len=int(target_sr*frame_ms/1000)
            pcm=(np.clip(y,-1.0,1.0)*32767).astype(np.int16).tobytes()
            frames=[pcm[i:i+frame_len*2] for i in range(0,len(pcm),frame_len*2)]
            voiced=[]
            for f in frames:
                if len(f)<frame_len*2: continue
                if vad.is_speech(f, target_sr): voiced.append(f)
            if voiced: pcm=b"".join(voiced)
            y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)/32768.0
        except Exception:
            pass
    # RMS normalize (safe)
    rms = float(np.sqrt(np.mean(y**2))+1e-8)
    gain = float(np.clip(0.08 / rms, 0.5, 4.0))
    y = np.clip(y*gain, -1.0, 1.0)
    buf=io.BytesIO()
    sf.write(buf, y, target_sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()

def postprocess_korean_text(text:str)->str:
    s=text.strip().replace("..",".").replace("  "," ")
    # simple de-dup punctuation
    while ".." in s: s=s.replace("..",".")
    return s

def is_low_quality_for_asr(vf:dict)->bool:
    return (vf.get("duration_sec",0)<2.0) or (vf.get("hnr",15)<8) or (vf.get("energy_mean",0)<0.03)

def transcribe_audio(audio_bytes:bytes)->str|None:
    if not openai_client: return None
    try:
        processed = preprocess_audio_for_asr(audio_bytes, target_sr=16000)
        tmp=None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t:
            t.write(processed); tmp=t.name
        init_prompt = build_initial_prompt_from_history(st.session_state.diary_entries)
        def _call(temp=0):
            with open(tmp,"rb") as fh:
                return openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=fh,
                    language="ko",
                    temperature=temp,
                    prompt=init_prompt or None
                )
        out = call_llm_safely(_call, 0)
        if (not out or not getattr(out,"text",None)) and not is_low_quality_for_asr(VoiceFeatureExtractor().extract(audio_bytes)):
            # second try with slight temp
            out = call_llm_safely(_call, 0.2)
        if tmp and os.path.exists(tmp):
            try: os.unlink(tmp)
            except Exception: pass
        if not out or not getattr(out,"text",None): return None
        return postprocess_korean_text(out.text)
    except Exception as e:
        print("ASR error:", e)
        return None

# =============================
# Coaching (2차 LLM with RAG)
# =============================
COACH_SYSTEM = (
    "너는 한국어 웰빙 코치다. 다음 규칙을 따른다.\n"
    "- 의료적 진단/약물/위험판단/자해 조언 금지. 필요한 경우 전문가 상담 권고 문구만.\n"
    "- 제안은 생활 습관/호흡/운동/시간관리/환경조정의 범위에서만.\n"
    "- 제공된 kb_context(지식문서) 근거에 **반드시** 기반해서 답한다. 근거가 없으면 '근거 없음'으로 명시.\n"
    "- 행동 추천은 최대 4개, 각 1~2문장, 구체적 수치(분/회/시간)를 포함.\n"
    "- JSON으로만 응답한다. 스키마: "
    '{"state":"상태","summary":"요약","positives":["긍정요소"],'
    '"recommendations":["행동"],"motivation":"격려","citations":[{"source":"파일","page":0}]}'
)

def build_coach_payload(text, combined, kb_ctx):
    cues = combined.get("voice_analysis",{}).get("voice_cues",{})
    return {
        "text": text,
        "signal": {
            "stress": combined.get("stress_level",0),
            "energy": combined.get("energy_level",0),
            "mood": combined.get("mood_score",0),
            "voice": {
                "arousal": int(cues.get("arousal",50)),
                "tension": int(cues.get("tension",50)),
                "stability": int(cues.get("stability",50)),
                "quality": float(cues.get("quality",0.5)),
            },
        },
        "kb_context": kb_ctx,
        "constraints": {"max_items":4,"no_medical":True,"language":"ko-KR"}
    }

def coach_with_rag(text, combined, kb_ctx)->dict:
    if not openai_client:
        return assess_mental_state(text, combined)  # fallback
    payload = build_coach_payload(text, combined, kb_ctx)
    def _call():
        return openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=650,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":COACH_SYSTEM},
                {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
            ]
        )
    resp = call_llm_safely(_call)
    if not resp or not resp.choices or not resp.choices[0].message.content:
        return assess_mental_state(text, combined)
    data = safe_json_parse(resp.choices[0].message.content)
    if not data:
        return assess_mental_state(text, combined)
    data.setdefault("state","중립"); data.setdefault("summary","오늘의 상태를 차분히 정리했어요.")
    data.setdefault("positives",[]); data.setdefault("recommendations",[])
    data.setdefault("motivation","작은 걸음이 큰 변화를 만듭니다."); data.setdefault("citations",[])
    data["recommendations"]=data.get("recommendations",[])[:4]
    data["positives"]=data.get("positives",[])[:4]
    return data

# =============================
# Fallback Coach (rule-based)
# =============================
def extract_positive_events(text:str)->list[str]:
    t=text.lower()
    keys=[("좋았","오늘 좋았던 점"),("행복","행복한 순간"),("고마","감사한 일"),
          ("즐겁","즐거웠던 활동"),("평온","평온했던 순간"),("성공","성취"),
          ("뿌듯","뿌듯했던 일"),("만족","만족스러운 일"),("친구","친구들과의 시간")]
    tags=[v for k,v in keys if k in t]
    return list(dict.fromkeys(tags))[:4]

def assess_mental_state(text, combined)->dict:
    tone=combined.get("tone","중립적")
    stress=combined.get("stress_level",30)
    energy=combined.get("energy_level",50)
    mood=combined.get("mood_score",0)
    cues=combined.get("voice_analysis",{}).get("voice_cues",{})
    arousal=float(cues.get("arousal",50)); tension=float(cues.get("tension",50))
    stability=float(cues.get("stability",50)); quality=float(cues.get("quality",0.5))
    positives=extract_positive_events(text)
    state="중립"
    if tone=="긍정적" and mood>=15 and stress<40: state="안정/회복"
    if energy<40 and mood<0: state="저활력"
    if stress>=60: state="고스트레스"
    if quality>0.4:
        if tension>65 and stability<45: state="긴장 과다"
        elif arousal>70 and stress>45: state="과흥분/과부하 가능"
        elif arousal<40 and energy<45: state="저각성"
    recs=[]
    if tone=="긍정적" or positives:
        if positives: recs.append("오늘 좋았던 3가지를 3줄로 기록해 보세요.")
        recs.append("좋았던 활동을 내일 10분 더 해보기.")
    if tension>60: recs.append("4-7-8 호흡 3회(4초 들숨,7초 멈춤,8초 날숨).")
    if stability<50: recs.append("목/어깨 이완 스트레칭 2분.")
    if arousal<45 or energy<45: recs.append("햇빛 10분 산책 + 가벼운 워킹 800~1000보.")
    if arousal>65 and stress>50: recs.append("알림 줄이기: 25분 집중+5분 휴식 2회.")
    recs=recs[:4]
    mot="작은 습관이 오늘의 좋은 흐름을 내일로 잇습니다."
    if state in ("고스트레스","긴장 과다"): mot="호흡을 고르고, 천천히. 당신의 속도로 충분합니다."
    elif state in ("저활력","저각성"): mot="작은 한 걸음이 에너지를 깨웁니다. 10분만 움직여볼까요?"
    summary=f"상태: {state} · 스트레스 {stress} · 에너지 {energy} · 각성 {int(arousal)} / 긴장 {int(tension)} / 안정 {int(stability)}"
    return {"state":state,"summary":summary,"positives":positives,"recommendations":recs,"motivation":mot,"voice_cues":{"arousal":arousal,"tension":tension,"stability":stability,"quality":quality}}

# =============================
# Weekly Report (fallback kept)
# =============================
def generate_simple_weekly_report(entries:list[dict])->dict:
    if len(entries)<3:
        return {"overall_trend":"안정적","key_insights":["데이터가 더 필요합니다."],
                "patterns":{"best_days":[],"challenging_days":[],"emotional_patterns":"더 많은 기록이 필요합니다."},
                "recommendations":{"priority_actions":["꾸준한 기록"],"wellness_tips":["하루 10분 성찰"],"goals_for_next_week":["매일 감정 기록"]},
                "encouragement":"좋은 시작입니다! 계속 이어가요."}
    recent=entries[-7:]
    avgS=np.mean([e["analysis"].get("stress_level",0) for e in recent])
    avgE=np.mean([e["analysis"].get("energy_level",0) for e in recent])
    avgM=np.mean([e["analysis"].get("mood_score",0) for e in recent])
    trend="안정적"
    if avgS<40 and avgE>60: trend="개선됨"
    elif avgS>70 or avgE<30: trend="주의필요"
    best=max(recent, key=lambda x:x["analysis"].get("mood_score",0))
    worst=min(recent, key=lambda x:x["analysis"].get("mood_score",0))
    return {
        "overall_trend":trend,
        "key_insights":[f"평균 스트레스: {avgS:.0f}","평균 에너지: {avgE:.0f}",f"평균 기분: {avgM:.0f}"],
        "patterns":{"best_days":[best.get("date","")],"challenging_days":[worst.get("date","")],"emotional_patterns":f"이번 주 전반적으로 {trend} 경향."},
        "recommendations":{"priority_actions":["스트레스 관리 집중","규칙적 운동"] if avgS>50 else ["현재 루틴 유지"],
                           "wellness_tips":["명상 5분","자연 산책","충분한 수면"],
                           "goals_for_next_week":["스트레스 40 이하 유지","매일 감정 기록"]},
        "encouragement":"매일 기록하는 당신, 이미 멋집니다!"
    }

# =============================
# RAG: PDF → Text → Chunk → Vector → Retrieve (No external deps)
# =============================
def default_kb_candidates():
    # Primary path as requested
    paths = ["소리일기/data/심리 건강 관리 정리 파일.pdf",
             "./소리일기/data/심리 건강 관리 정리 파일.pdf",
             "data/심리 건강 관리 정리 파일.pdf",
             "./data/심리 건강 관리 정리 파일.pdf"]
    return [p for p in paths if os.path.exists(p)]

def read_pdf_text(path)->list[dict]:
    """Return list of {'page':i,'text':...}"""
    if not PyPDF2: return []
    out=[]
    try:
        with open(path,"rb") as f:
            r=PyPDF2.PdfReader(f)
            for i in range(len(r.pages)):
                try:
                    t=r.pages[i].extract_text() or ""
                    t=t.replace("\x00","").strip()
                    if t: out.append({"page":i+1,"text":t})
                except Exception: pass
    except Exception as e:
        print("PDF read error:", e)
    return out

def normalize_text(s:str)->str:
    s=s.replace("\u200b"," ").replace("\xa0"," ").replace("  "," ")
    s=s.replace("\t"," ")
    return s.strip()

def chunk_text(text, chunk_chars=1100, overlap=180):
    text = normalize_text(text)
    chunks=[]
    i=0; n=len(text)
    while i<n:
        j=min(n, i+chunk_chars)
        # try cut at last period for nicer split
        cut=text.rfind(".", i, j)
        if cut==-1 or cut<i+chunk_chars*0.6: cut=j
        chunks.append(text[i:cut].strip())
        i = max(cut - overlap, 0) if cut<n else n
    return [c for c in chunks if c]

def hash_str(s:str)->str: return hashlib.md5(s.encode("utf-8")).hexdigest()

def build_vocab(chunks:list[str])->dict:
    vocab={}
    for c in chunks:
        for w in tokenize(c):
            vocab.setdefault(w,0)
    for i,k in enumerate(vocab.keys()):
        vocab[k]=i
    return vocab

def tokenize(s:str)->list[str]:
    s=s.lower()
    for ch in ",.!?:;()[]{}\"'<>/\n\r\t":
        s=s.replace(ch," ")
    tokens=[t for t in s.split(" ") if t and not t.isnumeric() and len(t)<=30]
    return tokens

def tfidf_matrix(chunks:list[str]):
    # Minimal TF-IDF without sklearn
    docs_tokens=[tokenize(c) for c in chunks]
    vocab={}
    for toks in docs_tokens:
        for t in set(toks): vocab[t]=vocab.get(t,0)+1
    N=len(chunks)
    idf={t: np.log((N+1)/(df+1))+1.0 for t,df in vocab.items()}
    rows=[]
    for toks in docs_tokens:
        tf={}
        for t in toks: tf[t]=tf.get(t,0)+1
        denom=max(1,sum(tf.values()))
        vec={}
        for t,cnt in tf.items():
            vec[t]= (cnt/denom) * idf.get(t,0.0)
        rows.append(vec)
    # map to dense with limited top features to save memory
    # keep top 2048 tokens overall
    top_terms=sorted(idf.items(), key=lambda x:x[1], reverse=True)[:2048]
    term_index={t:i for i,(t,_) in enumerate(top_terms)}
    X=np.zeros((len(chunks), len(term_index)), dtype=np.float32)
    for i,vec in enumerate(rows):
        for t,v in vec.items():
            j=term_index.get(t)
            if j is not None: X[i,j]=v
    # row-normalize
    norms=np.linalg.norm(X, axis=1, keepdims=True)+1e-8
    X=X/norms
    return X, term_index

@st.cache_resource(show_spinner=False)
def build_kb_index(paths:list[str]):
    """Return (index_matrix, metas) or (None,None) on failure."""
    metas=[]
    all_chunks=[]
    for p in paths:
        pages=read_pdf_text(p)
        for pg in pages:
            for ch in chunk_text(pg["text"]):
                all_chunks.append(ch)
                metas.append({"source":os.path.basename(p),"page":pg["page"],"chunk":ch})
    if not all_chunks:
        return None, None
    X, term_index = tfidf_matrix(all_chunks)
    return (X, metas)

def retrieve_kb(query:str, kb_index, kb_meta, top_k=4):
    if kb_index is None or kb_meta is None or kb_index.shape[0]==0:
        return []
    q_tokens=tokenize(query)
    if not q_tokens:
        return []
    # quick & dirty query vector by same idf as doc? We don't have idf here; use bag-of-words approximate
    # We approximate by averaging doc vectors which contain query tokens – fallback: random small noise
    # Better: build query vector using the same term_index we used – but not stored.
    # As an approximation, compute cosine between each chunk text and query using Jaccard over tokens + length weight.
    # Simpler & stable: cosine on character 3-gram vectors (lightweight)
    def char_ngrams(s, n=3):
        s=s.lower()
        s=" ".join(s.split())
        return [s[i:i+n] for i in range(0,max(0,len(s)-n+1))]
    q_ngrams=char_ngrams(query,3)
    if not q_ngrams: return []
    # Chunk scores as Jaccard of n-grams (fast enough for few thousand)
    scores=[]
    q_set=set(q_ngrams)
    for i,m in enumerate(kb_meta):
        c_set=set(char_ngrams(m["chunk"],3))
        inter=len(q_set & c_set); union=len(q_set | c_set)+1e-6
        s=inter/union
        scores.append((s,i))
    scores.sort(reverse=True)
    out=[]
    for _,i in scores[:top_k]:
        m=kb_meta[i]
        out.append({"chunk":m["chunk"],"source":m["source"],"page":m["page"]})
    return out

def ensure_kb_ready():
    if st.session_state.kb_ready: return
    candidates = default_kb_candidates()
    # Also allow upload
    up = st.session_state.get("kb_uploaded_bytes")
    if up:
        tmp_path = os.path.join(tempfile.gettempdir(), f"kb_{hashlib.md5(up).hexdigest()}.pdf")
        if not os.path.exists(tmp_path):
            with open(tmp_path,"wb") as f: f.write(up)
        candidates = [tmp_path] + candidates
    if not candidates:
        st.session_state.kb_index, st.session_state.kb_meta = None, None
        st.session_state.kb_ready = True
        return
    idx, meta = build_kb_index(candidates)
    st.session_state.kb_index, st.session_state.kb_meta = idx, meta
    st.session_state.kb_ready=True

# =============================
# Altair Chart helper (x label horizontal)
# =============================
def bar_chart_no_tilt(df, x_col, y_col, title=None):
    try:
        import altair as alt
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{x_col}:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y_col}:Q")
        ).properties(title=title).interactive()
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(df.set_index(x_col)[[y_col]])

# =============================
# UI Blocks
# =============================
def onboarding():
    if not st.session_state.onboarding_completed:
        with st.expander("🌟 처음 사용자 가이드", expanded=True):
            st.markdown("""
            - 🎙️ **음성으로 말하기**가 가장 쉬워요. 2~3분 권장, 조용한 곳에서 자연스럽게.
            - ✍️ 텍스트로 적어도 좋아요. 솔직하고 구체적으로!
            - 분석은 **텍스트 기준**, 목소리는 **보조 지표**예요.
            """)
            c1,c2=st.columns(2)
            if c1.button("🎯 바로 시작하기"):
                st.session_state.onboarding_completed=True; st.rerun()

def emotion_color(emotions:list[str])->str:
    if not emotions: return "#e9ecef"
    cmap={"기쁨":"#28a745","행복":"#28a745","평온":"#17a2b8","만족":"#6f42c1",
          "슬픔":"#6c757d","불안":"#ffc107","걱정":"#ffc107","분노":"#dc3545",
          "짜증":"#fd7e14","스트레스":"#dc3545","피로":"#6c757d","설렘":"#e83e8c","중립":"#e9ecef"}
    for e in emotions:
        if e in cmap: return cmap[e]
    return "#e9ecef"

def emotion_emoji(emotions:list[str])->str:
    if not emotions: return "😐"
    em={"기쁨":"😊","행복":"😊","평온":"😌","만족":"🙂","슬픔":"😢","불안":"😰","걱정":"😟","분노":"😠",
        "짜증":"😤","스트레스":"😵","피로":"😴","설렘":"😍","중립":"😐"}
    for e in emotions:
        if e in em: return em[e]
    return "😐"

# =============================
# Main Sidebar
# =============================
if not st.session_state.show_disclaimer:
    with st.sidebar:
        st.markdown("### 🔧 시스템 상태")
        st.markdown(f"- {'✅' if openai_client else '⚠️'} OpenAI API")
        st.markdown(f"- {'✅' if librosa else '⚠️'} 음성 분석(Librosa)")
        st.markdown(f"- {'✅' if parselmouth else 'ℹ️'} 고급 음성학(Praat)")
        st.markdown(f"- {'✅' if PyPDF2 else '⚠️'} PDF 파서")
        if not openai_client:
            with st.expander("🔑 OpenAI API 키 입력"):
                api_key=st.text_input("OpenAI API 키", type="password")
                if st.button("저장"):
                    if api_key.startswith("sk-"):
                        st.session_state.openai_api_key=api_key
                        st.success("저장됨. 상단 Rerun으로 새로고침해주세요.")
                    else:
                        st.error("키 형식이 올바르지 않습니다.")
        st.markdown("---")
        page=st.selectbox("페이지",["🎙️ 오늘의 이야기","💖 마음 분석","📈 감정 여정","📅 감정 캘린더","🎯 나의 목표","🎵 목소리 보조지표","📚 나의 이야기들","📚 RAG 지식베이스"])
        if st.session_state.diary_entries:
            st.markdown("### 📊 현재 상태")
            latest=st.session_state.diary_entries[-1]; a=latest.get("analysis",{})
            st.metric("기록 수", f"{len(st.session_state.diary_entries)}개")
            st.metric("최근 스트레스", f"{a.get('stress_level',0)}%")
            st.metric("최근 에너지", f"{a.get('energy_level',0)}%")
            if len(st.session_state.diary_entries)>=7 and st.button("📋 주간 리포트 생성"):
                st.session_state.weekly_report = generate_simple_weekly_report(st.session_state.diary_entries)
                st.session_state.show_weekly_report = True
        st.markdown("---")
        st.markdown("### 📁 데이터/KB")
        up_pdf = st.file_uploader("KB PDF 업로드(선택)", type=["pdf"])
        if up_pdf:
            st.session_state.kb_uploaded_bytes = up_pdf.read()
            st.session_state.kb_ready=False
            st.success("KB 업로드됨. 인덱스를 재구축합니다.")
        if st.button("🔍 KB 인덱스 구축/갱신"):
            st.session_state.kb_ready=False
            ensure_kb_ready()
            if st.session_state.kb_index is not None:
                st.success("KB 인덱스 준비 완료!")
            else:
                st.info("KB 문서를 찾지 못했습니다. 기본 경로를 확인하거나 업로드하세요.")
        st.markdown("---")
        st.markdown("### ℹ️ 앱 정보")
        st.markdown("**버전:** v2.1 (RAG/ASR 고도화)")
        st.markdown("**시간대:** 한국 표준시 (KST)")

# =============================
# Pages
# =============================
def page_today():
    st.header("오늘 하루는 어떠셨나요?")
    onboarding()
    extractor=VoiceFeatureExtractor()
    audio_val = st.audio_input("🎤 마음을 편하게 말해보세요", help="녹음 후 업로드 (2~3분 권장)")
    text_input = st.text_area("✏️ 글로 표현해도 좋아요", placeholder="오늘의 이야기를 적어주세요...", height=120)

    if st.button("💝 분석하고 저장", type="primary"):
        diary_text = text_input.strip()
        voice_analysis=None; audio_b64=None

        if audio_val is not None:
            audio_bytes=audio_val.read()
            audio_b64=base64.b64encode(audio_bytes).decode()
            with st.spinner("🎵 목소리 신호 계산 중..."):
                vf=extractor.extract(audio_bytes)
                update_baseline(vf)
                voice_analysis=analyze_voice_as_cues(vf, st.session_state.prosody_baseline)
            if openai_client and not diary_text:
                with st.spinner("🤖 음성 → 텍스트 전사 중..."):
                    tx = transcribe_audio(audio_bytes)
                    if tx:
                        diary_text=tx
                        st.info(f"🎤 들은 이야기: {tx}")
                    else:
                        st.warning("전사가 어려웠어요. 텍스트로 간단히 적어주세요!")

        if not diary_text:
            st.warning("텍스트를 입력하거나 음성을 녹음해 주세요.")
            return

        cues_for_prompt = voice_analysis["voice_cues"] if voice_analysis else None
        with st.spinner("🤖 텍스트 기반 감정 분석 중..."):
            t_res = analyze_text_with_llm(diary_text, cues_for_prompt)

        final = combine_text_and_voice(t_res, voice_analysis)

        # RAG 컨텍스트 준비
        ensure_kb_ready()
        kb_ctx=[]
        if st.session_state.kb_index is not None:
            q = f"스트레스 {final.get('stress_level',0)} 에너지 {final.get('energy_level',0)} 기분 {final.get('mood_score',0)} {diary_text[:200]}"
            kb_ctx = retrieve_kb(q, st.session_state.kb_index, st.session_state.kb_meta, top_k=4)

        with st.spinner("🧠 2차 코칭 생성 중..."):
            coach_card = coach_with_rag(diary_text, final, kb_ctx) if kb_ctx else assess_mental_state(diary_text, final)

        entry={
            "id": len(st.session_state.diary_entries)+1,
            "date": today_key(),
            "time": current_time(),
            "text": diary_text,
            "analysis": final,
            "audio_data": audio_b64,
            "mental_state": coach_card
        }
        st.session_state.diary_entries.append(entry)
        st.success("🎉 소중한 이야기가 저장되었습니다!")

        # 결과 표시
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("💖 감정 (텍스트 기반)")
            st.write(", ".join(final.get("emotions",[])))
            st.caption("감정 라벨은 텍스트만으로 판정합니다.")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 마음 상태")
            sc="metric-negative" if final['stress_level']>60 else ("metric-positive" if final['stress_level']<30 else "metric-neutral")
            ec="metric-positive" if final['energy_level']>60 else ("metric-negative" if final['energy_level']<40 else "metric-neutral")
            st.markdown(f"**스트레스:** <span class='{sc}'>{final['stress_level']}%</span>", unsafe_allow_html=True)
            st.markdown(f"**활력:** <span class='{ec}'>{final['energy_level']}%</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🎯 컨디션")
            mc="metric-positive" if final['mood_score']>10 else ("metric-negative" if final['mood_score']<-10 else "metric-neutral")
            st.markdown(f"**마음 점수:** <span class='{mc}'>{final['mood_score']}</span>", unsafe_allow_html=True)
            st.metric("분석 신뢰도", f"{final.get('confidence',0.6):.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        if voice_analysis:
            st.markdown("### 🎵 목소리 신호 (보조 지표)")
            cues = final["voice_analysis"]["voice_cues"]
            qtxt="높음" if cues["quality"]>0.7 else ("보통" if cues["quality"]>0.4 else "낮음")
            d1,d2,d3,d4=st.columns(4)
            d1.metric("각성도", f"{int(cues['arousal'])}/100")
            d2.metric("긴장도", f"{int(cues['tension'])}/100")
            d3.metric("안정도", f"{int(cues['stability'])}/100")
            d4.metric("녹음 품질", qtxt)
            st.caption("※ 목소리 신호는 보조 지표입니다. 감정 라벨은 텍스트 기반입니다.")

        st.markdown("### 🧠 오늘의 마음 코치")
        card_class = "success-card" if coach_card.get("state")=="안정/회복" else ("warning-card" if "스트레스" in coach_card.get("state","") else "card")
        st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
        st.write(f"**상태:** {coach_card.get('state','중립')}")
        st.write(coach_card.get("summary","오늘의 상태를 차분히 정리했어요."))
        if coach_card.get("positives"):
            st.write("**🌟 오늘의 밝은 포인트**")
            for p in coach_card["positives"]: st.write(f"• {p}")
        st.write("**💡 추천 행동**")
        for i, rec in enumerate(coach_card.get("recommendations",[]),1):
            st.write(f"{i}. {rec}")
        if coach_card.get("citations"):
            st.caption("📚 근거")
            for c in coach_card["citations"]:
                st.caption(f"- {c.get('source','문서')} p.{c.get('page','?')}")
        st.info(f"💪 {coach_card.get('motivation','오늘도 잘 해내셨어요.')}")
        st.markdown("</div>", unsafe_allow_html=True)

def page_dashboard():
    st.header("마음 분석 대시보드")
    if not st.session_state.diary_entries:
        st.info("기록이 아직 없어요. 첫 이야기를 들려주세요! 📝"); return
    st.subheader("📊 전체 통계")
    recent = st.session_state.diary_entries[-30:]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("총 기록 수", f"{len(st.session_state.diary_entries)}개")
    avgS=np.mean([e["analysis"].get("stress_level",0) for e in recent]); c2.metric("평균 스트레스", f"{avgS:.0f}%")
    avgE=np.mean([e["analysis"].get("energy_level",0) for e in recent]); c3.metric("평균 에너지", f"{avgE:.0f}%")
    avgM=np.mean([e["analysis"].get("mood_score",0) for e in recent]); c4.metric("평균 기분", f"{avgM:.0f}")

    st.subheader("😊 감정 분포 (최근 30개)")
    ec={}
    for e in recent:
        for em in e["analysis"].get("emotions",[]): ec[em]=ec.get(em,0)+1
    if ec:
        df=pd.DataFrame(list(ec.items()), columns=["감정","횟수"])
        bar_chart_no_tilt(df, "감정", "횟수", title="감정 분포")

    st.subheader("📋 상세 기록")
    df=pd.DataFrame([{
        "날짜":e["date"],"시간":e["time"],"감정":", ".join(e["analysis"].get("emotions",[])),
        "스트레스":e["analysis"].get("stress_level",0),
        "에너지":e["analysis"].get("energy_level",0),
        "기분":e["analysis"].get("mood_score",0),
        "톤":e["analysis"].get("tone","중립적"),
        "신뢰도":f"{e['analysis'].get('confidence',0.6):.2f}"
    } for e in st.session_state.diary_entries])
    c1,c2=st.columns(2)
    with c1: date_filter=st.date_input("날짜 필터 (이후)", value=None)
    with c2: emotion_filter=st.selectbox("감정 필터", ["전체"]+list(ec.keys()))
    fdf=df.copy()
    if date_filter: fdf=fdf[pd.to_datetime(fdf["날짜"])>=pd.to_datetime(date_filter)]
    if emotion_filter!="전체": fdf=fdf[fdf["감정"].str.contains(emotion_filter)]
    st.dataframe(fdf, use_container_width=True, hide_index=True,
                 column_config={"스트레스":st.column_config.ProgressColumn("스트레스",max_value=100),
                                "에너지":st.column_config.ProgressColumn("에너지",max_value=100)})

def page_journey():
    st.header("시간에 따른 변화")
    if not st.session_state.diary_entries:
        st.info("기록이 쌓이면 추세를 보여드릴게요. 📈"); return
    c1,c2=st.columns(2)
    with c1: period=st.selectbox("기간 선택",["전체","최근 30일","최근 14일","최근 7일"])
    entries=st.session_state.diary_entries
    if period=="최근 30일": entries=entries[-30:]
    elif period=="최근 14일": entries=entries[-14:]
    elif period=="최근 7일": entries=entries[-7:]
    if len(entries)<2:
        st.warning("추세 분석을 위해 최소 2개 기록이 필요합니다."); return
    df=pd.DataFrame([{
        "날짜시간": f"{e['date']} {e['time']}",
        "날짜": e['date'],
        "스트레스": e["analysis"].get("stress_level",0),
        "에너지": e["analysis"].get("energy_level",0),
        "기분": e["analysis"].get("mood_score",0)+70
    } for e in entries])
    with c2: metric=st.selectbox("지표 선택",["전체","스트레스","에너지","기분"])
    if metric=="전체":
        st.line_chart(df.set_index("날짜시간")[["스트레스","에너지","기분"]])
    else:
        st.line_chart(df.set_index("날짜시간")[[metric]])
        if metric=="기분": st.caption("※ 시각화를 위해 +70 조정 (실제 범위 -70~70)")
    st.subheader("📊 추세 분석")
    stress_trend=np.polyfit(range(len(entries)), [e["analysis"].get("stress_level",0) for e in entries], 1)[0]
    energy_trend=np.polyfit(range(len(entries)), [e["analysis"].get("energy_level",0) for e in entries], 1)[0]
    mood_trend=np.polyfit(range(len(entries)), [e["analysis"].get("mood_score",0) for e in entries], 1)[0]
    c1,c2,c3=st.columns(3)
    c1.metric("스트레스 추세", "📉 감소" if stress_trend<-0.1 else ("📈 증가" if stress_trend>0.1 else "➡️ 안정"), delta=f"{stress_trend:.2f}")
    c2.metric("에너지 추세", "📈 증가" if energy_trend>0.1 else ("📉 감소" if energy_trend<-0.1 else "➡️ 안정"), delta=f"{energy_trend:.2f}")
    c3.metric("기분 추세", "📈 개선" if mood_trend>0.1 else ("📉 하락" if mood_trend<-0.1 else "➡️ 안정"), delta=f"{mood_trend:.2f}")
    st.subheader("🔍 인사이트")
    insights=[]
    if stress_trend<-0.5: insights.append("✨ 스트레스가 꾸준히 감소 중! 현재 방식을 유지해보세요.")
    elif stress_trend>0.5: insights.append("⚠️ 스트레스 증가 추세. 휴식과 관리가 필요합니다.")
    if energy_trend>0.5: insights.append("🔋 에너지 레벨 상승 중! 좋은 습관을 이어가요.")
    elif energy_trend<-0.5: insights.append("😴 에너지 하락. 수면/운동/영양 루틴 점검을.")
    if mood_trend>0.5: insights.append("😊 기분이 전반적으로 좋아지고 있어요!")
    elif mood_trend<-0.5: insights.append("💙 기분 하락. 자기돌봄 시간을 확보해요.")
    if not insights: insights.append("📊 전반적으로 안정적인 상태입니다.")
    for s in insights: st.info(s)

def page_calendar():
    st.header("📅 감정 캘린더")
    if not st.session_state.diary_entries:
        st.info("기록이 쌓이면 캘린더로 패턴을 볼 수 있어요!"); return
    today=kst_now()
    months=set([e["date"][:7] for e in st.session_state.diary_entries])
    months.add(today.strftime("%Y-%m"))
    sorted_months=sorted(list(months), reverse=True)
    col1,col2=st.columns([1,3])
    with col1:
        sel=st.selectbox("월 선택", sorted_months, index=0,
                         format_func=lambda x:f"{x.split('-')[0]}년 {int(x.split('-')[1])}월")
        year,month = map(int, sel.split('-'))
    with col2:
        st.markdown(f"### {year}년 {month}월")
    month_entries={}
    for e in st.session_state.diary_entries:
        if e["date"].startswith(sel):
            day=int(e["date"].split("-")[2])
            month_entries.setdefault(day,[]).append(e)
    try:
        cal=calendar.monthcalendar(year, month)
    except Exception:
        st.error("캘린더 생성 오류"); return
    weekdays=["월","화","수","목","금","토","일"]
    cols=st.columns(7)
    for i,d in enumerate(weekdays): cols[i].markdown(f"<div style='text-align:center;font-weight:bold;padding:8px'>{d}</div>", unsafe_allow_html=True)
    for week_idx,week in enumerate(cal):
        cols=st.columns(7)
        for day_idx,day in enumerate(week):
            if day==0:
                cols[day_idx].markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)
            else:
                entries=month_entries.get(day,[])
                is_today=(day==today.day and month==today.month and year==today.year)
                if entries:
                    latest=entries[-1]; emotions=latest.get("analysis",{}).get("emotions",[])
                    emoji=emotion_emoji(emotions); color=emotion_color(emotions)
                    key=f"cal_{year}_{month}_{day}_{week_idx}_{day_idx}"
                    clicked=cols[day_idx].button(f"{emoji}\n{day}", key=key, help=f"{', '.join(emotions)} ({len(entries)}개 기록)", use_container_width=True)
                    border="border:2px solid #667eea;" if is_today else "border:1px solid #ddd;"
                    cols[day_idx].markdown(f"<div style='background:{color};opacity:0.3;{border}border-radius:8px;height:10px;margin-top:2px;'></div>", unsafe_allow_html=True)
                    if clicked:
                        st.session_state[f"show_day_{year}_{month}_{day}"]=True
                else:
                    border="border:2px solid #667eea;" if is_today else "border:1px solid #ddd;"
                    cols[day_idx].markdown(f"<div style='{border}border-radius:8px;padding:20px;margin:2px;text-align:center;background:#fafafa;color:#999'>{day}</div>", unsafe_allow_html=True)
    for d in range(1,32):
        sk=f"show_day_{year}_{month}_{d}"
        if st.session_state.get(sk, False):
            entries=month_entries.get(d,[])
            if entries:
                st.markdown(f"### {year}년 {month}월 {d}일 기록")
                for i,e in enumerate(entries):
                    emotions=", ".join(e.get("analysis",{}).get("emotions",[]))
                    with st.expander(f"📝 {e.get('time','')} - {emotions}", expanded=(i==0)):
                        st.write(e.get("text",""))
                        a=e.get("analysis",{})
                        c1,c2,c3=st.columns(3)
                        c1.metric("스트레스", f"{a.get('stress_level',0)}%")
                        c2.metric("에너지", f"{a.get('energy_level',0)}%")
                        c3.metric("기분", f"{a.get('mood_score',0)}")
                if st.button("닫기", key=f"close_{year}_{month}_{d}"):
                    st.session_state[sk]=False; st.rerun()
            break

def page_goals():
    st.header("🎯 나의 목표 설정 & 추적")
    with st.expander("➕ 새로운 목표 추가하기"):
        c1,c2=st.columns(2)
        with c1:
            gtype=st.selectbox("목표 유형",["stress","energy","mood","consistency"],
                format_func=lambda x:{"stress":"스트레스 낮추기","energy":"에너지 높이기","mood":"기분 개선","consistency":"주간 기록 횟수"}[x])
        with c2:
            if gtype=="consistency":
                target=st.slider("주간 목표 기록 횟수",1,7,5); desc=f"일주일에 {target}번 이상 기록"
            elif gtype=="stress":
                target=st.slider("목표 스트레스 (이하)",10,50,30); desc=f"스트레스 {target} 이하 유지"
            elif gtype=="energy":
                target=st.slider("목표 에너지 (이상)",50,90,70); desc=f"에너지 {target} 이상 유지"
            else:
                target=st.slider("목표 기분 (이상)",0,50,20); desc=f"기분 {target} 이상 유지"
        custom=st.text_input("목표 설명 (선택)", value=desc)
        if st.button("목표 추가"):
            st.session_state.user_goals.append({"id":len(st.session_state.user_goals)+1,"type":gtype,"target":target,"description":custom,"created_date":today_key(),"active":True})
            st.success("목표 추가 완료!"); st.rerun()
    active=[g for g in st.session_state.user_goals if g.get("active",True)]
    if not active:
        st.info("설정된 목표가 없습니다."); return
    st.subheader("📊 목표 진행 상황")
    for g in active:
        info=check_goal_progress(g); prog=info["progress"]; cur=info["current_value"]; status=info["status"]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1,c2,c3=st.columns([3,1,1])
        with c1:
            st.write(f"**{g['description']}**"); st.progress(prog/100); st.caption(f"진행률: {prog:.1f}% | 현재값: {cur:.1f}")
        with c2:
            st.success(status) if status=="달성!" else st.info(status)
        with c3:
            if st.button("🗑️", key=f"del_goal_{g['id']}"):
                g["active"]=False; st.success("삭제되었습니다."); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def check_goal_progress(goal:dict)->dict:
    rec=st.session_state.diary_entries[-7:]
    if not rec: return {"progress":0,"current_value":0,"status":"진행중"}
    tp=goal["type"]; target=goal["target"]
    if tp=="consistency":
        cur=len(rec); prog=min(100, (cur/target)*100)
    else:
        vals=[]
        for e in rec:
            a=e.get("analysis",{})
            if tp=="stress": vals.append(a.get("stress_level",0))
            elif tp=="energy": vals.append(a.get("energy_level",0))
            elif tp=="mood": vals.append(a.get("mood_score",0))
        cur=np.mean(vals) if vals else 0
        if tp=="stress":
            prog = 100 if cur<=target else max(0, min(100, (target/cur)*100))
        else:
            prog = min(100, (cur/target)*100) if target>0 else 0
    status = "달성!" if prog>=100 else "진행중"
    return {"progress":prog,"current_value":cur,"status":status}

def page_voice():
    st.header("목소리 신호 상세 분석")
    entries=[e for e in st.session_state.diary_entries if e.get("analysis",{}).get("voice_analysis")]
    if not entries:
        st.info("음성으로 기록된 항목이 아직 없습니다. 🎤 첫 음성 기록을 남겨보세요!"); return
    sel=st.selectbox("분석할 기록 선택", entries, index=len(entries)-1,
                     format_func=lambda x:f"{x['date']} {x['time']} - {', '.join(x['analysis'].get('emotions',[]))}")
    voice=sel["analysis"]["voice_analysis"]; vf=voice["voice_features"]; cues=voice["voice_cues"]
    st.subheader("🎯 음성 보조지표")
    qtxt="높음" if cues["quality"]>0.7 else ("보통" if cues["quality"]>0.4 else "낮음")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("각성도", f"{int(cues['arousal'])}/100")
    c2.metric("긴장도", f"{int(cues['tension'])}/100")
    c3.metric("안정도", f"{int(cues['stability'])}/100")
    c4.metric("녹음 품질", qtxt)
    st.subheader("🔬 기초 음성 특성")
    d1,d2=st.columns(2)
    with d1:
        s1,s2=st.columns(2); s1.metric("피치 평균", f"{vf.get('pitch_mean',0):.1f} Hz"); s2.metric("피치 변동", f"{vf.get('pitch_variation',0):.3f}")
        s3,s4=st.columns(2); s3.metric("음성 에너지", f"{vf.get('energy_mean',0):.3f}"); s4.metric("최대 에너지", f"{vf.get('energy_max',0):.3f}")
    with d2:
        s5,s6=st.columns(2); s5.metric("말하기 속도", f"{vf.get('tempo',0):.0f} BPM"); s6.metric("영교차율", f"{vf.get('zcr_mean',0):.3f}")
        s7,s8=st.columns(2); s7.metric("HNR(명료도)", f"{vf.get('hnr',0):.1f} dB"); s8.metric("Jitter(안정)", f"{vf.get('jitter',0):.4f}")
    if st.session_state.prosody_baseline:
        st.subheader("📈 개인 베이스라인")
        base=st.session_state.prosody_baseline; st.info(f"{base.get('_count',0)}개 기록 기반 베이스라인")
        if st.button("베이스라인 초기화"):
            st.session_state.prosody_baseline={}; st.success("초기화 완료"); st.rerun()
    st.caption("※ 보조지표이며, 감정 라벨은 텍스트로 판단합니다.")

def page_archive():
    st.header("나의 이야기 아카이브")
    if not st.session_state.diary_entries:
        st.info("아직 기록이 없어요. 첫 이야기를 들려주세요! ✨"); return
    if len(st.session_state.diary_entries)>=7:
        c1,c2=st.columns([1,3])
        with c1:
            if st.button("📋 주간 리포트 생성", type="primary"):
                st.session_state.weekly_report = generate_simple_weekly_report(st.session_state.diary_entries)
                st.session_state.show_weekly_report=True
    if st.session_state.get("show_weekly_report",False) and st.session_state.weekly_report:
        r=st.session_state.weekly_report
        st.markdown("### 📊 주간 웰빙 리포트")
        icon={"개선됨":"🟢","안정적":"🟡","주의필요":"🔴"}.get(r.get("overall_trend","안정적"),"🟡")
        st.markdown(f"**전체 추세:** {icon} {r.get('overall_trend','안정적')}")
        if r.get("key_insights"):
            st.markdown("**🔍 주요 발견사항**"); [st.write(f"• {x}") for x in r["key_insights"]]
        pat=r.get("patterns",{})
        if pat:
            c1,c2=st.columns(2)
            with c1:
                if pat.get("best_days"):
                    st.markdown("**🌟 좋았던 날들**"); [st.write(f"• {d}") for d in pat["best_days"]]
            with c2:
                if pat.get("challenging_days"):
                    st.markdown("**💪 도전적이었던 날들**"); [st.write(f"• {d}") for d in pat["challenging_days"]]
            if pat.get("emotional_patterns"):
                st.markdown("**📈 감정 패턴**"); st.write(pat["emotional_patterns"])
        rec=r.get("recommendations",{})
        if rec:
            st.markdown("### 💡 다음 주를 위한 추천")
            if rec.get("priority_actions"):
                st.markdown("**🎯 우선순위 행동**"); [st.write(f"{i+1}. {x}") for i,x in enumerate(rec["priority_actions"])]
            if rec.get("wellness_tips"):
                st.markdown("**🌱 웰빙 팁**"); [st.write(f"• {x}") for x in rec["wellness_tips"]]
            if rec.get("goals_for_next_week"):
                st.markdown("**🎯 다음 주 목표**"); [st.write(f"• {x}") for x in rec["goals_for_next_week"]]
        st.success(f"💪 {r.get('encouragement','잘하고 있어요!')}")
        if st.button("리포트 닫기"):
            st.session_state.show_weekly_report=False; st.rerun()
        st.markdown("---")
    st.subheader("🔍 기록 탐색")
    c1,c2,c3=st.columns(3)
    with c1: stext=st.text_input("🔍 텍스트 검색", placeholder="키워드로 검색…")
    with c2:
        all_em=set([em for e in st.session_state.diary_entries for em in e.get("analysis",{}).get("emotions",[])])
        efilter=st.selectbox("😊 감정 필터", ["전체"]+list(all_em))
    with c3: dfilter=st.date_input("📅 날짜 이후", value=None)
    ents=st.session_state.diary_entries
    if stext: ents=[e for e in ents if stext.lower() in e.get("text","").lower()]
    if efilter!="전체": ents=[e for e in ents if efilter in e.get("analysis",{}).get("emotions",[])]
    if dfilter: ents=[e for e in ents if e.get("date","")>=dfilter.strftime("%Y-%m-%d")]
    st.write(f"**총 {len(ents)}개** (전체 {len(st.session_state.diary_entries)}개 중)")
    for i,e in enumerate(reversed(ents[-20:])):
        a=e.get("analysis",{}); emos=a.get("emotions",[]); state=e.get("mental_state",{}).get("state","")
        card="success-card" if state=="안정/회복" else ("warning-card" if any(k in state for k in ["스트레스","긴장","과부하"]) else "card")
        emoji=emotion_emoji(emos)
        with st.expander(f"{emoji} {e['date']} {e['time']} · {', '.join(emos)} · {state}", expanded=(i==0)):
            st.markdown(f"<div class='{card}'>", unsafe_allow_html=True)
            st.markdown("**📝 기록 내용**"); st.write(e["text"])
            c1,c2,c3=st.columns(3)
            s=a.get("stress_level",0); en=a.get("energy_level",0); m=a.get("mood_score",0)
            c1.write(f"**스트레스:** {'🔴' if s>60 else ('🟡' if s>30 else '🟢')} {s}%")
            c2.write(f"**에너지:** {'🟢' if en>60 else ('🟡' if en>40 else '🔴')} {en}%")
            c3.write(f"**기분:** {'🟢' if m>10 else ('🟡' if m>-10 else '🔴')} {m}")
            ms=e.get("mental_state",{})
            if ms.get("summary"): st.markdown("**🧠 코치 요약**"); st.info(ms["summary"])
            if a.get("voice_analysis"):
                vc=a["voice_analysis"]["voice_cues"]; st.markdown("**🎵 음성 보조지표**")
                v1,v2,v3=st.columns(3); v1.write(f"각성:{int(vc.get('arousal',0))}"); v2.write(f"긴장:{int(vc.get('tension',0))}"); v3.write(f"안정:{int(vc.get('stability',0))}")
            st.markdown("</div>", unsafe_allow_html=True)

def page_kb():
    st.header("📚 RAG 지식베이스")
    st.write("행동 추천의 근거가 되는 문서를 색인하고 검색합니다.")
    ensure_kb_ready()
    if st.session_state.kb_index is None:
        st.info("KB가 아직 준비되지 않았습니다. 좌측 사이드바에서 PDF 업로드 또는 인덱스 구축을 눌러주세요.")
        return
    q=st.text_input("🔍 KB 검색어", placeholder="예) 스트레스 관리 호흡법, 수면 루틴, 긴장 완화")
    if st.button("검색") and q.strip():
        ctx=retrieve_kb(q, st.session_state.kb_index, st.session_state.kb_meta, top_k=5)
        if not ctx: st.info("결과가 없습니다.")
        else:
            for c in ctx:
                with st.expander(f"📄 {c['source']} · p.{c['page']}"):
                    st.write(c["chunk"][:1500]+"...")

# =============================
# Export/Reset Sidebar (bottom)
# =============================
def export_sidebar():
    with st.sidebar:
        if st.session_state.diary_entries:
            st.markdown("---"); st.markdown("### 📁 데이터 관리")
            if st.button("📊 CSV 내보내기"):
                rows=[]
                for e in st.session_state.diary_entries:
                    a=e["analysis"]; row={"날짜":e["date"],"시간":e["time"],"텍스트":e["text"],
                        "감정":", ".join(a.get("emotions",[])),"스트레스":a.get("stress_level",0),
                        "에너지":a.get("energy_level",0),"기분":a.get("mood_score",0),
                        "톤":a.get("tone","중립적"),"신뢰도":a.get("confidence",0.6)}
                    ms=e.get("mental_state")
                    if ms: row.update({"상태":ms.get("state",""),"코치요약":ms.get("summary",""),
                                       "추천사항":" | ".join(ms.get("recommendations",[]))})
                    v=a.get("voice_analysis")
                    if v:
                        vc=v["voice_cues"]; vf=v["voice_features"]
                        row.update({"각성도":vc.get("arousal",""),"긴장도":vc.get("tension",""),
                                    "안정도":vc.get("stability",""),"음질":vc.get("quality",""),
                                    "피치평균":vf.get("pitch_mean",""),"음성에너지":vf.get("energy_mean",""),
                                    "말속도":vf.get("tempo",""),"HNR":vf.get("hnr","")})
                    rows.append(row)
                df=pd.DataFrame(rows); csv=df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("📥 다운로드", csv, file_name=f"voice_diary_{kst_now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
            if st.button("📋 JSON 내보내기"):
                export={"exported_at":kst_now().isoformat(),"total_entries":len(st.session_state.diary_entries),
                        "entries":st.session_state.diary_entries,"goals":st.session_state.user_goals,
                        "baseline":st.session_state.prosody_baseline}
                js=json.dumps(export, ensure_ascii=False, indent=2)
                st.download_button("📥 전체 데이터 다운로드", js, file_name=f"voice_diary_full_{kst_now().strftime('%Y%m%d_%H%M')}.json", mime="application/json")
            st.markdown("---")
            if st.button("🗑️ 모든 기록 삭제", type="secondary"):
                if st.button("⚠️ 정말 삭제하시겠습니까?", type="secondary"):
                    st.session_state.diary_entries=[]; st.session_state.user_goals=[]; st.session_state.prosody_baseline={}
                    st.success("모든 기록이 삭제되었습니다."); st.rerun()

# =============================
# Footer
# =============================
def footer():
    if not st.session_state.show_disclaimer:
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align:center;color:#666;font-size:0.9rem;padding:1rem;'>
            Made with ❤️ | 감정 라벨은 <strong>텍스트 우선</strong> · 목소리는 <strong>보조</strong><br>
            마지막 업데이트: {kst_now().strftime('%Y-%m-%d %H:%M KST')} |
            기록 수: {len(st.session_state.diary_entries)}개 |
            목표 수: {len([g for g in st.session_state.user_goals if g.get('active', True)])}개
        </div>
        """, unsafe_allow_html=True)

# =============================
# Run
# =============================
header()
show_disclaimer()
if not st.session_state.show_disclaimer:
    # Main Router
    extractor=VoiceFeatureExtractor()
    if 'page' not in locals():
        page="🎙️ 오늘의 이야기"
    if page=="🎙️ 오늘의 이야기": page_today()
    elif page=="💖 마음 분석": page_dashboard()
    elif page=="📈 감정 여정": page_journey()
    elif page=="📅 감정 캘린더": page_calendar()
    elif page=="🎯 나의 목표": page_goals()
    elif page=="🎵 목소리 보조지표": page_voice()
    elif page=="📚 나의 이야기들": page_archive()
    elif page=="📚 RAG 지식베이스": page_kb()
    export_sidebar()
footer()
