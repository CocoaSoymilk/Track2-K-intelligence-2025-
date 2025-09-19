import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import base64
import io
import numpy as np
from typing import Dict, List, Tuple, Optional
import tempfile
import warnings
warnings.filterwarnings('ignore')

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

try:
    import librosa
    import scipy.stats
    from scipy import signal
    AUDIO_ANALYSIS_AVAILABLE = True
except ImportError:
    st.warning("음성 분석 패키지가 설치되지 않았습니다. 기본 분석만 제공됩니다.")
    AUDIO_ANALYSIS_AVAILABLE = False
    librosa = None
    scipy = None

try:
    import parselmouth
    from parselmouth.praat import call
    PRAAT_AVAILABLE = True
except ImportError:
    st.info("Praat 분석 패키지가 없어 기본 음성 분석을 사용합니다.")
    PRAAT_AVAILABLE = False
    parselmouth = None

# 페이지 설정
st.set_page_config(
    page_title="소리로 쓰는 하루 - 고도화",
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
    
    .voice-analysis-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(255, 152, 0, 0.1);
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
    
    .prosodic-meter {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
        }
        
        .emotion-card, .metric-container, .feedback-box, .voice-analysis-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'diary_entries' not in st.session_state:
    st.session_state.diary_entries = []

class VoiceFeatureExtractor:
    """음성 피처 추출 클래스"""
    
    def __init__(self):
        self.sample_rate = 22050
        
    def extract_prosodic_features(self, audio_bytes: bytes) -> Dict:
        """음성에서 prosodic features 추출"""
        try:
            if not AUDIO_ANALYSIS_AVAILABLE:
                return self._get_default_features()
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # librosa로 오디오 로드
                y, sr = librosa.load(tmp_file_path, sr=self.sample_rate)
                
                # 기본 피처들 추출
                features = {}
                
                # 1. Pitch (F0) 분석
                pitch_features = self._extract_pitch_features(y, sr)
                features.update(pitch_features)
                
                # 2. 에너지/강도 분석
                energy_features = self._extract_energy_features(y, sr)
                features.update(energy_features)
                
                # 3. 말하기 속도 분석
                tempo_features = self._extract_tempo_features(y, sr)
                features.update(tempo_features)
                
                # 4. 스펙트럼 특성 분석
                spectral_features = self._extract_spectral_features(y, sr)
                features.update(spectral_features)
                
                # 5. Praat 기반 분석 (가능한 경우)
                if PRAAT_AVAILABLE:
                    praat_features = self._extract_praat_features(tmp_file_path)
                    features.update(praat_features)
                
                return features
                
            finally:
                # 임시 파일 삭제
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            st.warning(f"음성 피처 추출 중 오류: {str(e)}")
            return self._get_default_features()
    
    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> Dict:
        """피치 관련 특성 추출"""
        # 피치 추출 (librosa의 piptrack 사용)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1, fmin=50, fmax=400)
        
        # 유효한 피치 값만 추출
        valid_pitches = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                valid_pitches.append(pitch)
        
        if len(valid_pitches) == 0:
            return {
                'pitch_mean': 150.0,  # 기본값
                'pitch_std': 20.0,
                'pitch_range': 50.0,
                'pitch_variation': 0.13
            }
        
        pitch_array = np.array(valid_pitches)
        
        return {
            'pitch_mean': float(np.mean(pitch_array)),
            'pitch_std': float(np.std(pitch_array)),
            'pitch_range': float(np.max(pitch_array) - np.min(pitch_array)),
            'pitch_variation': float(np.std(pitch_array) / np.mean(pitch_array)) if np.mean(pitch_array) > 0 else 0.1
        }
    
    def _extract_energy_features(self, y: np.ndarray, sr: int) -> Dict:
        """에너지/강도 관련 특성 추출"""
        # RMS 에너지
        rms_energy = librosa.feature.rms(y=y)[0]
        
        # 스펙트럼 롤오프 (에너지 분포의 85% 지점)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        return {
            'energy_mean': float(np.mean(rms_energy)),
            'energy_std': float(np.std(rms_energy)),
            'energy_max': float(np.max(rms_energy)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff))
        }
    
    def _extract_tempo_features(self, y: np.ndarray, sr: int) -> Dict:
        """템포/리듬 관련 특성 추출"""
        # 템포 추정
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Zero Crossing Rate (음성 활동성의 지표)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'tempo': float(tempo),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr))
        }
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """스펙트럼 특성 추출"""
        # Spectral Centroid (음색의 밝기)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # MFCC (음성 특성)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'mfcc_mean': float(np.mean(mfccs[1:5])),  # 1-4번째 MFCC 계수들의 평균
            'mfcc_std': float(np.mean(np.std(mfccs[1:5], axis=1)))
        }
    
    def _extract_praat_features(self, audio_path: str) -> Dict:
        """Praat을 통한 고급 음성학적 분석"""
        try:
            sound = parselmouth.Sound(audio_path)
            
            # Pitch 객체 생성
            pitch = sound.to_pitch()
            
            # Jitter와 Shimmer (음성 안정성 지표)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Harmonics-to-Noise Ratio
            harmonicity = sound.to_harmonicity()
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            return {
                'jitter': float(jitter) if not np.isnan(jitter) else 0.01,
                'shimmer': float(shimmer) if not np.isnan(shimmer) else 0.05,
                'hnr': float(hnr) if not np.isnan(hnr) else 15.0
            }
        except Exception as e:
            return {
                'jitter': 0.01,
                'shimmer': 0.05,
                'hnr': 15.0
            }
    
    def _get_default_features(self) -> Dict:
        """기본 피처 값들 (분석 불가능한 경우)"""
        return {
            'pitch_mean': 150.0,
            'pitch_std': 20.0,
            'pitch_range': 50.0,
            'pitch_variation': 0.13,
            'energy_mean': 0.1,
            'energy_std': 0.02,
            'energy_max': 0.3,
            'spectral_rolloff_mean': 3000.0,
            'tempo': 120.0,
            'zcr_mean': 0.1,
            'zcr_std': 0.05,
            'spectral_centroid_mean': 2000.0,
            'spectral_centroid_std': 500.0,
            'spectral_bandwidth_mean': 1500.0,
            'mfcc_mean': 0.0,
            'mfcc_std': 1.0,
            'jitter': 0.01,
            'shimmer': 0.05,
            'hnr': 15.0
        }

class EmotionAnalyzer:
    """음성 피처 기반 감정 분석기"""
    
    def __init__(self):
        # 감정별 음성 피처 기준값 (선행 연구 기반)
        self.emotion_profiles = {
            '기쁨': {
                'pitch_mean': (180, 220),  # 높은 피치
                'pitch_variation': (0.15, 0.25),  # 높은 변동성
                'energy_mean': (0.15, 0.35),  # 높은 에너지
                'tempo': (130, 160),  # 빠른 템포
                'spectral_centroid_mean': (2200, 3000),  # 밝은 음색
                'jitter': (0.005, 0.015),  # 낮은 jitter (안정적)
            },
            '슬픔': {
                'pitch_mean': (100, 140),  # 낮은 피치
                'pitch_variation': (0.08, 0.15),  # 낮은 변동성
                'energy_mean': (0.05, 0.15),  # 낮은 에너지
                'tempo': (80, 110),  # 느린 템포
                'spectral_centroid_mean': (1500, 2200),  # 어두운 음색
                'jitter': (0.008, 0.020),  # 약간 높은 jitter
            },
            '분노': {
                'pitch_mean': (160, 200),  # 높은 피치
                'pitch_variation': (0.20, 0.35),  # 매우 높은 변동성
                'energy_mean': (0.20, 0.40),  # 매우 높은 에너지
                'tempo': (140, 180),  # 매우 빠른 템포
                'spectral_centroid_mean': (2500, 3500),  # 매우 밝은/거친 음색
                'jitter': (0.015, 0.030),  # 높은 jitter (불안정)
            },
            '불안': {
                'pitch_mean': (150, 190),  # 약간 높은 피치
                'pitch_variation': (0.18, 0.30),  # 높은 변동성
                'energy_mean': (0.10, 0.25),  # 중간 에너지
                'tempo': (110, 140),  # 약간 빠른 템포
                'spectral_centroid_mean': (2000, 2800),  # 약간 밝은 음색
                'jitter': (0.012, 0.025),  # 높은 jitter
            },
            '평온': {
                'pitch_mean': (130, 160),  # 중간 피치
                'pitch_variation': (0.10, 0.18),  # 낮은 변동성
                'energy_mean': (0.08, 0.20),  # 중간 에너지
                'tempo': (100, 130),  # 중간 템포
                'spectral_centroid_mean': (1800, 2400),  # 부드러운 음색
                'jitter': (0.005, 0.012),  # 낮은 jitter (안정적)
            }
        }
    
    def analyze_emotion_from_voice(self, voice_features: Dict) -> Dict:
        """음성 피처로부터 감정 분석"""
        emotion_scores = {}
        
        for emotion, profile in self.emotion_profiles.items():
            score = self._calculate_emotion_score(voice_features, profile)
            emotion_scores[emotion] = score
        
        # 점수 정규화
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
        
        # 상위 감정들 선택
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        detected_emotions = [emotion for emotion, score in sorted_emotions[:3] if score > 0.15]
        
        if not detected_emotions:
            detected_emotions = ['중립']
        
        # 감정 기반 스트레스/에너지/기분 점수 계산
        stress_level = self._calculate_stress_level(voice_features, emotion_scores)
        energy_level = self._calculate_energy_level(voice_features, emotion_scores)
        mood_score = self._calculate_mood_score(voice_features, emotion_scores)
        
        return {
            'detected_emotions': detected_emotions,
            'emotion_scores': emotion_scores,
            'voice_stress_level': stress_level,
            'voice_energy_level': energy_level,
            'voice_mood_score': mood_score,
            'voice_features': voice_features
        }
    
    def _calculate_emotion_score(self, features: Dict, profile: Dict) -> float:
        """특정 감정에 대한 점수 계산"""
        score = 0.0
        feature_count = 0
        
        for feature_name, (min_val, max_val) in profile.items():
            if feature_name in features:
                feature_value = features[feature_name]
                
                # 범위 내에 있으면 1, 범위를 벗어나면 거리에 따라 감소
                if min_val <= feature_value <= max_val:
                    score += 1.0
                else:
                    # 범위에서 벗어난 정도에 따라 점수 감소
                    range_center = (min_val + max_val) / 2
                    range_width = max_val - min_val
                    distance = abs(feature_value - range_center)
                    normalized_distance = distance / (range_width / 2)
                    score += max(0, 1 - normalized_distance * 0.5)
                
                feature_count += 1
        
        return score / feature_count if feature_count > 0 else 0.0
    
    def _calculate_stress_level(self, features: Dict, emotion_scores: Dict) -> int:
        """음성 기반 스트레스 레벨 계산 (0-100)"""
        base_stress = 30
        
        # 감정별 스트레스 가중치
        stress_weights = {
            '분노': 35,
            '불안': 30,
            '슬픔': 20,
            '기쁨': -15,
            '평온': -20
        }
        
        for emotion, weight in stress_weights.items():
            if emotion in emotion_scores:
                base_stress += weight * emotion_scores[emotion]
        
        # 음성 피처 기반 조정
        if features.get('jitter', 0) > 0.02:  # 높은 jitter = 높은 스트레스
            base_stress += 15
        
        if features.get('pitch_variation', 0) > 0.25:  # 높은 피치 변동성
            base_stress += 10
        
        return max(0, min(100, int(base_stress)))
    
    def _calculate_energy_level(self, features: Dict, emotion_scores: Dict) -> int:
        """음성 기반 에너지 레벨 계산 (0-100)"""
        base_energy = 50
        
        # 감정별 에너지 가중치
        energy_weights = {
            '기쁨': 25,
            '분노': 30,
            '불안': 10,
            '슬픔': -25,
            '평온': -5
        }
        
        for emotion, weight in energy_weights.items():
            if emotion in emotion_scores:
                base_energy += weight * emotion_scores[emotion]
        
        # 음성 피처 기반 조정
        energy_mean = features.get('energy_mean', 0.1)
        if energy_mean > 0.2:  # 높은 에너지
            base_energy += 15
        elif energy_mean < 0.08:  # 낮은 에너지
            base_energy -= 15
        
        tempo = features.get('tempo', 120)
        if tempo > 140:  # 빠른 템포
            base_energy += 10
        elif tempo < 100:  # 느린 템포
            base_energy -= 10
        
        return max(0, min(100, int(base_energy)))
    
    def _calculate_mood_score(self, features: Dict, emotion_scores: Dict) -> int:
        """음성 기반 기분 점수 계산 (-70 to +70)"""
        base_mood = 0
        
        # 감정별 기분 가중치
        mood_weights = {
            '기쁨': 40,
            '평온': 20,
            '슬픔': -35,
            '분노': -25,
            '불안': -20
        }
        
        for emotion, weight in mood_weights.items():
            if emotion in emotion_scores:
                base_mood += weight * emotion_scores[emotion]
        
        # 음성 품질 기반 조정 (좋은 음성 품질 = 더 긍정적)
        hnr = features.get('hnr', 15.0)
        if hnr > 20:  # 좋은 음성 품질
            base_mood += 5
        elif hnr < 10:  # 나쁜 음성 품질
            base_mood -= 5
        
        return max(-70, min(70, int(base_mood)))

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

def analyze_emotion_with_gpt(text: str, voice_analysis: Optional[Dict] = None) -> Dict:
    """GPT-4를 사용하여 텍스트와 음성 분석을 종합한 감정 분석"""
    if not openai_client:
        return analyze_emotion_simulation(text, voice_analysis)
    
    try:
        # 음성 분석 정보를 포함한 프롬프트 구성
        voice_context = ""
        if voice_analysis:
            voice_context = f"""
            
            추가로, 음성 분석 결과도 참고해주세요:
            - 음성으로 감지된 감정: {', '.join(voice_analysis.get('detected_emotions', []))}
            - 음성 기반 스트레스: {voice_analysis.get('voice_stress_level', 30)}%
            - 음성 기반 에너지: {voice_analysis.get('voice_energy_level', 50)}%
            - 음성 기반 기분: {voice_analysis.get('voice_mood_score', 0)}
            """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 "소리로 쓰는 하루" 서비스의 따뜻하고 공감적인 AI 마음 분석가입니다. 
                    사용자가 음성이나 글로 들려준 하루 이야기를 분석하여 정확한 JSON 형식으로 응답해주세요.
                    
                    텍스트 분석과 음성 분석 결과를 종합하여 더 정확한 감정 분석을 제공해주세요.
                    {voice_context}
                    
                    응답 형식:
                    {{
                        "emotions": ["기쁨", "슬픔", "분노", "불안", "평온", "중립" 중 해당하는 것들의 배열],
                        "stress_level": 스트레스 수치 (0-100의 정수),
                        "energy_level": 에너지 수치 (0-100의 정수),
                        "mood_score": 전체적인 마음 점수 (-70부터 +70 사이의 정수),
                        "summary": "따뜻하고 공감적인 톤으로 한두 문장 요약",
                        "keywords": ["핵심 키워드들"],
                        "tone": "긍정적" 또는 "중립적" 또는 "부정적",
                        "confidence": 분석 신뢰도 (0.0-1.0, 음성 분석이 있으면 더 높게)
                    }}
                    
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
                'tone': '중립적',
                'confidence': 0.7
            }
            
            for field, default_value in required_fields.items():
                if field not in result:
                    result[field] = default_value
            
            # 음성 분석이 있으면 가중 평균으로 결합
            if voice_analysis:
                result = combine_text_and_voice_analysis(result, voice_analysis)
            
            return result
            
        except json.JSONDecodeError:
            st.warning("GPT 응답 파싱 중 오류 발생. 시뮬레이션 모드로 전환합니다.")
            return analyze_emotion_simulation(text, voice_analysis)
        
    except Exception as e:
        st.error(f"GPT 분석 오류: {str(e)}")
        return analyze_emotion_simulation(text, voice_analysis)

def combine_text_and_voice_analysis(text_analysis: Dict, voice_analysis: Dict) -> Dict:
    """텍스트 분석과 음성 분석을 결합하여 최종 분석 결과 생성"""
    
    # 가중치 설정 (텍스트 60%, 음성 40%)
    text_weight = 0.6
    voice_weight = 0.4
    
    # 감정 결합 (두 분석에서 공통으로 나온 감정을 우선)
    text_emotions = set(text_analysis.get('emotions', []))
    voice_emotions = set(voice_analysis.get('detected_emotions', []))
    
    combined_emotions = list(text_emotions.union(voice_emotions))
    if not combined_emotions:
        combined_emotions = ['중립']
    
    # 수치 결합 (가중 평균)
    combined_stress = int(
        text_analysis.get('stress_level', 30) * text_weight + 
        voice_analysis.get('voice_stress_level', 30) * voice_weight
    )
    
    combined_energy = int(
        text_analysis.get('energy_level', 50) * text_weight + 
        voice_analysis.get('voice_energy_level', 50) * voice_weight
    )
    
    combined_mood = int(
        text_analysis.get('mood_score', 0) * text_weight + 
        voice_analysis.get('voice_mood_score', 0) * voice_weight
    )
    
    # 신뢰도 향상 (음성 분석이 있으면 더 높은 신뢰도)
    confidence = min(1.0, text_analysis.get('confidence', 0.7) + 0.2)
    
    return {
        'emotions': combined_emotions[:3],  # 최대 3개 감정
        'stress_level': max(0, min(100, combined_stress)),
        'energy_level': max(0, min(100, combined_energy)),
        'mood_score': max(-70, min(70, combined_mood)),
        'summary': text_analysis.get('summary', '종합적인 분석이 완료되었습니다.'),
        'keywords': text_analysis.get('keywords', []),
        'tone': text_analysis.get('tone', '중립적'),
        'confidence': confidence,
        'voice_analysis': voice_analysis  # 음성 분석 결과도 저장
    }

def analyze_emotion_simulation(text: str, voice_analysis: Optional[Dict] = None) -> Dict:
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
    
    result = {
        'emotions': detected_emotions if detected_emotions else ['중립'],
        'stress_level': stress_level,
        'energy_level': energy_level,
        'mood_score': mood_score,
        'summary': f"{tone} 상태로, 주요 감정은 {', '.join(detected_emotions[:2]) if detected_emotions else '중립'}입니다.",
        'keywords': keywords[:5],
        'tone': tone,
        'confidence': 0.5
    }
    
    # 음성 분석이 있으면 결합
    if voice_analysis:
        result = combine_text_and_voice_analysis(result, voice_analysis)
    
    return result

def generate_personalized_feedback(entries: List[Dict]) -> str:
    """개인화된 피드백 생성"""
    if not entries:
        return "첫 번째 음성 일기를 작성해보세요!"
    
    recent_entries = entries[-7:]  # 최근 7일
    
    if not openai_client:
        return generate_basic_feedback(recent_entries)
    
    try:
        # 최근 데이터 요약 (음성 분석 포함)
        summary_data = []
        for entry in recent_entries:
            entry_summary = {
                'date': entry['date'],
                'emotions': entry['analysis']['emotions'],
                'stress': entry['analysis']['stress_level'],
                'energy': entry['analysis']['energy_level'],
                'tone': entry['analysis'].get('tone', '중립적'),
                'confidence': entry['analysis'].get('confidence', 0.5)
            }
            
            # 음성 분석이 있으면 추가
            if 'voice_analysis' in entry['analysis']:
                entry_summary['voice_emotions'] = entry['analysis']['voice_analysis'].get('detected_emotions', [])
                entry_summary['voice_confidence'] = 'high' if entry['analysis'].get('confidence', 0.5) > 0.8 else 'medium'
            
            summary_data.append(entry_summary)
        
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
                    음성 분석이 포함된 경우 더 정확한 분석이 가능했다는 점을 언급해주세요."""
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
        return "첫 번째 음성 일기를 작성해보세요!"
    
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
    
    # 음성 분석 포함 여부 체크
    voice_analyzed_count = sum(1 for entry in entries if 'voice_analysis' in entry['analysis'])
    voice_feedback = f" 특히 음성 분석을 통해 더 정확한 분석이 가능했습니다." if voice_analyzed_count > 0 else ""
    
    if avg_stress > 65:
        return f"최근 스트레스 지수가 {avg_stress:.0f}%로 높은 편이에요. 깊은 호흡이나 짧은 산책으로 마음을 달래보세요.{voice_feedback} 작은 휴식도 큰 도움이 됩니다!"
    elif avg_energy < 35:
        return f"최근 에너지가 {avg_energy:.0f}%로 낮아 보여요. 충분한 수면과 좋아하는 활동으로 에너지를 충전해보세요.{voice_feedback} 당신을 위한 시간을 가져보세요!"
    elif most_frequent == "기쁨":
        return f"최근 긍정적인 감정이 많이 보이네요!{voice_feedback} 이 좋은 에너지를 유지하며 새로운 목표에 도전해보는 건 어떨까요?"
    else:
        return f"전체적으로 안정적인 상태를 보이고 있어요.{voice_feedback} 꾸준히 자신의 감정을 기록하는 습관이 정말 훌륭합니다! 계속 응원할게요!"

# 전역 객체 초기화
voice_extractor = VoiceFeatureExtractor()
emotion_analyzer = EmotionAnalyzer()

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🎙️ 소리로 쓰는 하루 - 고도화</h1>
    <p>목소리로 담는 오늘, AI가 읽어주는 마음</p>
    <small style="opacity: 0.8;">음성 피처 분석으로 더 정확한 감정 분석을 제공합니다</small>
</div>
""", unsafe_allow_html=True)

# 기능 상태 표시
with st.sidebar:
    st.markdown("### 🔧 시스템 상태")
    
    status_indicators = []
    if OPENAI_AVAILABLE and openai_client:
        status_indicators.append("✅ OpenAI API")
    else:
        status_indicators.append("⚠️ OpenAI API 설정 필요")
    
    if AUDIO_ANALYSIS_AVAILABLE:
        status_indicators.append("✅ 음성 분석 (Librosa)")
    else:
        status_indicators.append("⚠️ 기본 음성 분석")
    
    if PRAAT_AVAILABLE:
        status_indicators.append("✅ 고급 음성학 분석 (Praat)")
    else:
        status_indicators.append("ℹ️ 표준 음성학 분석")
    
    if PLOTLY_AVAILABLE:
        status_indicators.append("✅ 시각화")
    else:
        status_indicators.append("⚠️ 기본 차트만 가능")
    
    for indicator in status_indicators:
        st.markdown(f"- {indicator}")

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
                    st.success("API 키가 저장되었습니다!")
                    st.rerun()
                else:
                    st.error("올바른 API 키 형식이 아닙니다.")
        
        st.info("💡 API 키 없이도 음성 피처 기반 감정 분석을 체험할 수 있어요.")

# 사이드바 네비게이션
with st.sidebar:
    st.title("🌟 오늘의 마음")
    
    # 오늘 일기 작성 여부 확인
    today = datetime.now().strftime("%Y-%m-%d")
    today_entries = [entry for entry in st.session_state.diary_entries if entry['date'] == today]
    
    if today_entries:
        st.success(f"✅ 오늘 {len(today_entries)}번의 마음을 기록했어요")
        # 음성 분석 포함 여부
        voice_count = sum(1 for entry in today_entries if 'voice_analysis' in entry.get('analysis', {}))
        if voice_count > 0:
            st.info(f"🎵 {voice_count}개 항목에서 음성 분석 완료")
    else:
        st.info("💭 오늘의 이야기를 들려주세요")
    
    page = st.selectbox(
        "페이지 선택",
        ["🎙️ 오늘의 이야기", "💖 마음 분석", "📈 감정 여정", "🎵 음성 분석", "💡 마음 케어", "📚 나의 이야기들"],
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
            
            # 분석 신뢰도
            confidence = latest_entry['analysis'].get('confidence', 0.5)
            confidence_text = "높음" if confidence > 0.8 else "보통" if confidence > 0.6 else "기본"
            st.metric("분석 신뢰도", confidence_text)

# 페이지별 콘텐츠
if page == "🎙️ 오늘의 이야기":
    st.header("오늘 하루는 어떠셨나요?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **💝 마음을 나누는 시간:**
        - 1분만 투자해보세요, 당신의 이야기가 소중해요
        - 오늘 있었던 일, 느낀 감정을 자유롭게 말해보세요
        - 음성으로 말하면 목소리의 높낮이, 속도, 에너지까지 분석해드려요
        - 특별한 일이 없어도 괜찮아요, 평범한 하루도 의미 있어요
        """)
    
    with col2:
        # 오늘 작성한 일기 수
        if today_entries:
            st.info(f"🌟 오늘 {len(today_entries)}번째 이야기")
        else:
            st.info("🌱 오늘 첫 번째 이야기")
        
        # 음성 분석 기능 안내
        if AUDIO_ANALYSIS_AVAILABLE:
            st.success("🎵 음성 피처 분석 지원")
        else:
            st.warning("📝 텍스트 분석 위주")
    
    # 음성 녹음 섹션
    st.markdown("### 🎙️ 목소리로 들려주세요")
    
    with st.container():
        st.markdown('<div class="recording-container">', unsafe_allow_html=True)
        
        # Streamlit 내장 음성 입력 사용
        audio_value = st.audio_input(
            "🎤 마음을 편하게 말해보세요",
            help="마이크 버튼을 눌러 녹음을 시작하세요. 음성의 높낮이, 속도, 에너지까지 분석해드려요"
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
        voice_analysis = None
        
        # 음성 데이터 처리
        if audio_value is not None:
            audio_bytes = audio_value.read()
            audio_data = base64.b64encode(audio_bytes).decode()
            
            with st.spinner("🎵 음성의 피치, 에너지, 템포를 분석하는 중..."):
                # 음성 피처 추출
                voice_features = voice_extractor.extract_prosodic_features(audio_bytes)
                
                # 음성 기반 감정 분석
                voice_analysis = emotion_analyzer.analyze_emotion_from_voice(voice_features)
                
                st.success("✅ 음성 피처 분석 완료!")
                
                # 음성 분석 결과 미리보기
                if voice_analysis:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("음성 감정", ', '.join(voice_analysis['detected_emotions'][:2]))
                    with col2:
                        st.metric("음성 에너지", f"{voice_analysis['voice_energy_level']}%")
                    with col3:
                        st.metric("음성 스트레스", f"{voice_analysis['voice_stress_level']}%")
            
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
            with st.spinner("🤖 텍스트와 음성을 종합하여 마음을 분석하는 중..."):
                analysis = analyze_emotion_with_gpt(diary_text, voice_analysis)
            
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
            
            # 종합 분석 결과
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
                confidence_emoji = "🎯" if analysis.get('confidence', 0.5) > 0.8 else "📍" if analysis.get('confidence', 0.5) > 0.6 else "📌"
                st.markdown(f"""
                <div class="emotion-card">
                    <h4>🎯 오늘의 컨디션</h4>
                    <p>마음 점수: <strong>{analysis['mood_score']}</strong></p>
                    <p>분석 신뢰도: <strong>{confidence_emoji} {analysis.get('confidence', 0.5):.1f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # 음성 분석이 포함된 경우 추가 정보
            if voice_analysis:
                st.markdown("### 🎵 음성 분석 상세 결과")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="voice-analysis-card">
                        <h4>🎤 음성 특성</h4>
                        <p>피치 평균: <strong>{voice_analysis['voice_features'].get('pitch_mean', 0):.1f} Hz</strong></p>
                        <p>에너지: <strong>{voice_analysis['voice_features'].get('energy_mean', 0):.3f}</strong></p>
                        <p>말하기 속도: <strong>{voice_analysis['voice_features'].get('tempo', 0):.0f} BPM</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="voice-analysis-card">
                        <h4>🎯 음성 감정 점수</h4>
                        <p>음성 스트레스: <strong>{voice_analysis['voice_stress_level']}%</strong></p>
                        <p>음성 활력: <strong>{voice_analysis['voice_energy_level']}%</strong></p>
                        <p>음성 기분: <strong>{voice_analysis['voice_mood_score']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # AI 요약
            if 'summary' in analysis:
                confidence_text = "높은 신뢰도" if analysis.get('confidence', 0.5) > 0.8 else "보통 신뢰도" if analysis.get('confidence', 0.5) > 0.6 else "기본 분석"
                st.markdown(f"""
                <div class="feedback-box">
                    <h4>🤖 AI가 전해드리는 말 ({confidence_text})</h4>
                    <p>{analysis['summary']}</p>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ 목소리나 글로 마음을 들려주세요!")

elif page == "🎵 음성 분석":
    st.header("음성 피처 분석 대시보드")
    
    if not st.session_state.diary_entries:
        st.info("📝 음성으로 일기를 작성하면 상세한 음성 분석을 볼 수 있어요!")
    else:
        # 음성 분석이 포함된 항목들만 필터링
        voice_entries = [entry for entry in st.session_state.diary_entries 
                        if 'voice_analysis' in entry.get('analysis', {})]
        
        if not voice_entries:
            st.warning("🎤 음성으로 기록된 일기가 없어요. 목소리로 이야기를 들려주세요!")
        else:
            st.success(f"🎵 {len(voice_entries)}개의 음성 분석 데이터가 있어요!")
            
            # 최근 음성 분석 결과
            latest_voice = voice_entries[-1]
            voice_analysis = latest_voice['analysis']['voice_analysis']
            
            st.markdown("### 🎯 최근 음성 분석 결과")
            
            # 음성 피처 상세 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="prosodic-meter">
                    <h4>🎼 피치 (Hz)</h4>
                    <h2 style="color: #667eea;">{voice_analysis['voice_features'].get('pitch_mean', 0):.1f}</h2>
                    <small>변동성: {voice_analysis['voice_features'].get('pitch_variation', 0):.2f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="prosodic-meter">
                    <h4>⚡ 에너지</h4>
                    <h2 style="color: #51cf66;">{voice_analysis['voice_features'].get('energy_mean', 0):.3f}</h2>
                    <small>최대: {voice_analysis['voice_features'].get('energy_max', 0):.3f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="prosodic-meter">
                    <h4>🏃 말하기 속도</h4>
                    <h2 style="color: #ffd43b;">{voice_analysis['voice_features'].get('tempo', 0):.0f}</h2>
                    <small>BPM</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                jitter = voice_analysis['voice_features'].get('jitter', 0)
                stability = "안정적" if jitter < 0.015 else "보통" if jitter < 0.025 else "불안정"
                st.markdown(f"""
                <div class="prosodic-meter">
                    <h4>🎚️ 음성 안정성</h4>
                    <h2 style="color: #ff6b6b;">{stability}</h2>
                    <small>Jitter: {jitter:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # 감정별 음성 특성 분석
            st.markdown("### 📊 감정별 음성 패턴 분석")
            
            if PLOTLY_AVAILABLE and len(voice_entries) > 1:
                # 음성 피처 시계열 분석
                voice_df = pd.DataFrame([
                    {
                        'date': entry['date'],
                        'emotions': ', '.join(entry['analysis']['emotions'][:2]),
                        'pitch_mean': entry['analysis']['voice_analysis']['voice_features'].get('pitch_mean', 150),
                        'energy_mean': entry['analysis']['voice_analysis']['voice_features'].get('energy_mean', 0.1),
                        'tempo': entry['analysis']['voice_analysis']['voice_features'].get('tempo', 120),
                        'pitch_variation': entry['analysis']['voice_analysis']['voice_features'].get('pitch_variation', 0.1),
                        'jitter': entry['analysis']['voice_analysis']['voice_features'].get('jitter', 0.01)
                    }
                    for entry in voice_entries
                ])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 피치 변화 그래프
                    fig_pitch = px.line(voice_df, x='date', y='pitch_mean', 
                                       title='피치 변화 추이',
                                       color='emotions',
                                       hover_data=['pitch_variation'])
                    fig_pitch.update_layout(height=300)
                    st.plotly_chart(fig_pitch, use_container_width=True)
                
                with col2:
                    # 에너지와 템포 관계
                    fig_energy = px.scatter(voice_df, x='energy_mean', y='tempo',
                                          size='pitch_variation',
                                          color='emotions',
                                          title='에너지 vs 말하기 속도',
                                          hover_data=['date'])
                    fig_energy.update_layout(height=300)
                    st.plotly_chart(fig_energy, use_container_width=True)
                
                # 음성 안정성 분석
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_stability = px.bar(voice_df, x='date', y='jitter',
                                          color='emotions',
                                          title='음성 안정성 (Jitter)')
                    fig_stability.update_layout(height=300)
                    st.plotly_chart(fig_stability, use_container_width=True)
                
                with col2:
                    # 감정별 음성 특성 레이더 차트
                    emotion_voice_stats = voice_df.groupby('emotions').agg({
                        'pitch_mean': 'mean',
                        'energy_mean': 'mean', 
                        'tempo': 'mean',
                        'pitch_variation': 'mean'
                    }).reset_index()
                    
                    if len(emotion_voice_stats) > 0:
                        fig_radar = go.Figure()
                        
                        for _, row in emotion_voice_stats.iterrows():
                            fig_radar.add_trace(go.Scatterpolar(
                                r=[
                                    row['pitch_mean']/200,  # 정규화
                                    row['energy_mean']*5,   # 스케일 조정
                                    row['tempo']/150,       # 정규화
                                    row['pitch_variation']*5  # 스케일 조정
                                ],
                                theta=['피치', '에너지', '템포', '변동성'],
                                fill='toself',
                                name=row['emotions']
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=True,
                            title="감정별 음성 특성 패턴",
                            height=300
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
            
            else:
                st.info("📈 더 많은 음성 데이터가 쌓이면 상세한 패턴 분석을 제공해드려요!")
            
            # 음성 품질 지표
            st.markdown("### 🎙️ 음성 품질 및 특성 분석")
            
            # 최근 5개 항목의 음성 품질 분석
            recent_voice = voice_entries[-5:]
            
            quality_metrics = []
            for entry in recent_voice:
                features = entry['analysis']['voice_analysis']['voice_features']
                
                # 음성 품질 점수 계산
                hnr = features.get('hnr', 15.0)
                jitter = features.get('jitter', 0.01)
                shimmer = features.get('shimmer', 0.05)
                
                quality_score = min(100, max(0, 
                    (hnr - 5) * 4 +  # HNR 기여도
                    (1 - min(jitter * 50, 1)) * 30 +  # Jitter 기여도 (낮을수록 좋음)
                    (1 - min(shimmer * 20, 1)) * 30    # Shimmer 기여도 (낮을수록 좋음)
                ))
                
                quality_metrics.append({
                    'date': entry['date'],
                    'quality_score': quality_score,
                    'hnr': hnr,
                    'jitter': jitter,
                    'shimmer': shimmer,
                    'emotions': ', '.join(entry['analysis']['emotions'])
                })
            
            quality_df = pd.DataFrame(quality_metrics)
            avg_quality = quality_df['quality_score'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                quality_emoji = "🌟" if avg_quality > 80 else "👍" if avg_quality > 60 else "📢"
                st.metric("평균 음성 품질", f"{quality_emoji} {avg_quality:.0f}점")
            
            with col2:
                avg_hnr = quality_df['hnr'].mean()
                st.metric("음성 명료도 (HNR)", f"{avg_hnr:.1f} dB")
            
            with col3:
                avg_stability = 1 - quality_df['jitter'].mean()
                st.metric("음성 안정성", f"{avg_stability:.1%}")
            
            # 음성 품질 개선 제안
            if avg_quality < 70:
                st.markdown("""
                <div class="feedback-box">
                    <h4>🎤 음성 품질 개선 팁</h4>
                    <p>• 조용한 환경에서 녹음해보세요<br>
                    • 마이크와 적당한 거리를 유지하세요<br>
                    • 천천히, 명확하게 말해보세요<br>
                    • 깊게 숨을 들이마신 후 말하시면 더 안정적인 음성이 됩니다</p>
                </div>
                """, unsafe_allow_html=True)

# 기존 페이지들도 동일하게 유지하되, 음성 분석 결과가 포함된 경우 추가 정보 표시
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
        
        with col3:
            analysis_filter = st.selectbox(
                "분석 유형",
                ["전체", "음성 분석 포함", "텍스트만"]
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
        
        if analysis_filter == "음성 분석 포함":
            filtered_entries = [e for e in filtered_entries if 'voice_analysis' in e.get('analysis', {})]
        elif analysis_filter == "텍스트만":
            filtered_entries = [e for e in filtered_entries if 'voice_analysis' not in e.get('analysis', {})]
        
        if not filtered_entries:
            st.warning("선택한 필터에 해당하는 일기가 없습니다.")
        else:
            # 오늘의 요약 (오늘 일기가 있는 경우)
            today_entries = [e for e in filtered_entries if e['date'] == datetime.now().strftime("%Y-%m-%d")]
            if today_entries and date_filter in ["전체", "오늘", "최근 7일", "최근 30일"]:
                st.markdown("### 📅 오늘의 감정 요약")
                latest = today_entries[-1]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("주요 감정", ', '.join(latest['analysis']['emotions'][:2]))
                with col2:
                    st.metric("스트레스", f"{latest['analysis']['stress_level']}%")
                with col3:
                    st.metric("활력", f"{latest['analysis']['energy_level']}%")
                with col4:
                    mood_emoji = "😊" if latest['analysis']['mood_score'] > 10 else "😐" if latest['analysis']['mood_score'] > -10 else "😔"
                    st.metric("기분", f"{mood_emoji} {latest['analysis']['mood_score']}")
                with col5:
                    confidence = latest['analysis'].get('confidence', 0.5)
                    confidence_emoji = "🎯" if confidence > 0.8 else "📍" if confidence > 0.6 else "📌"
                    st.metric("신뢰도", f"{confidence_emoji} {confidence:.1f}")
            
            # 일기 목록
            st.markdown(f"### 📝 일기 목록 ({len(filtered_entries)}개)")
            
            # 페이지네이션
            items_per_page = 5
            total_pages = (len(filtered_entries) - 1) // items_per_page + 1 if filtered_entries else 1
            current_page = st.select_slider("페이지", range(1, total_pages + 1), value=1)
            
            start_idx = (current_page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            current_entries = list(reversed(filtered_entries))[start_idx:end_idx]
            
            for i, entry in enumerate(current_entries):
                has_voice = 'voice_analysis' in entry.get('analysis', {})
                voice_indicator = "🎵" if has_voice else "📝"
                confidence = entry['analysis'].get('confidence', 0.5)
                confidence_indicator = "🎯" if confidence > 0.8 else "📍" if confidence > 0.6 else "📌"
                
                with st.expander(
                    f"{voice_indicator} 📅 {entry['date']} {entry['time']} - {', '.join(entry['analysis']['emotions'])} "
                    f"({'😊' if entry['analysis']['mood_score'] > 10 else '😐' if entry['analysis']['mood_score'] > -10 else '😔'}) {confidence_indicator}"
                ):
                    # 일기 내용
                    st.markdown(f"**📝 내용:** {entry['text']}")
                    
                    # 음성 파일 재생
                    if entry.get('audio_data'):
                        st.markdown("**🎵 녹음된 음성:**")
                        audio_bytes = base64.b64decode(entry['audio_data'])
                        st.audio(audio_bytes)
                    
                    # 분석 결과
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("스트레스", f"{entry['analysis']['stress_level']}%")
                    with col2:
                        st.metric("활력", f"{entry['analysis']['energy_level']}%")
                    with col3:
                        st.metric("기분 점수", f"{entry['analysis']['mood_score']}")
                    with col4:
                        st.metric("분석 신뢰도", f"{confidence:.2f}")
                    
                    # 음성 분석이 있는 경우 추가 정보
                    if has_voice:
                        st.markdown("**🎤 음성 분석 결과:**")
                        voice_analysis = entry['analysis']['voice_analysis']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("음성 피치", f"{voice_analysis['voice_features'].get('pitch_mean', 0):.1f} Hz")
                        with col2:
                            st.metric("음성 에너지", f"{voice_analysis['voice_energy_level']}%")
                        with col3:
                            st.metric("말하기 속도", f"{voice_analysis['voice_features'].get('tempo', 0):.0f} BPM")
                    
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
        
        col1, col2 = st.columns(2)
        with col1:
            selected_period = st.selectbox("📅 분석 기간", list(period_options.keys()), index=1)
        with col2:
            analysis_type = st.selectbox("분석 유형", ["종합 분석", "음성 분석만", "텍스트 분석만"])
        
        entries_to_analyze = st.session_state.diary_entries
        if period_options[selected_period]:
            entries_to_analyze = st.session_state.diary_entries[-period_options[selected_period]:]
        
        # 분석 유형 필터링
        if analysis_type == "음성 분석만":
            entries_to_analyze = [e for e in entries_to_analyze if 'voice_analysis' in e.get('analysis', {})]
        elif analysis_type == "텍스트 분석만":
            entries_to_analyze = [e for e in entries_to_analyze if 'voice_analysis' not in e.get('analysis', {})]
        
        if not entries_to_analyze:
            st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            return
        
        # 데이터 준비
        df_data = []
        for entry in entries_to_analyze:
            row = {
                'date': entry['date'],
                'time': entry['time'],
                'datetime': f"{entry['date']} {entry['time']}",
                'stress': entry['analysis']['stress_level'],
                'energy': entry['analysis']['energy_level'],
                'mood': entry['analysis']['mood_score'],
                'emotions': ', '.join(entry['analysis']['emotions'][:2]),
                'tone': entry['analysis'].get('tone', '중립적'),
                'confidence': entry['analysis'].get('confidence', 0.5),
                'has_voice': 'voice_analysis' in entry.get('analysis', {})
            }
            
            # 음성 분석 데이터 추가
            if 'voice_analysis' in entry.get('analysis', {}):
                voice_analysis = entry['analysis']['voice_analysis']
                row.update({
                    'voice_stress': voice_analysis.get('voice_stress_level', 30),
                    'voice_energy': voice_analysis.get('voice_energy_level', 50),
                    'voice_mood': voice_analysis.get('voice_mood_score', 0),
                    'pitch_mean': voice_analysis['voice_features'].get('pitch_mean', 150),
                    'energy_mean': voice_analysis['voice_features'].get('energy_mean', 0.1),
                    'tempo': voice_analysis['voice_features'].get('tempo', 120)
                })
            else:
                row.update({
                    'voice_stress': None,
                    'voice_energy': None,
                    'voice_mood': None,
                    'pitch_mean': None,
                    'energy_mean': None,
                    'tempo': None
                })
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 일별 평균 계산
        daily_avg = df.groupby('date').agg({
            'stress': 'mean',
            'energy': 'mean',
            'mood': 'mean',
            'confidence': 'mean'
        }).reset_index()
        
        # 메인 그래프들
        col1, col2 = st.columns(2)
        
        with col1:
            # 시간별 감정 변화 (신뢰도 고려)
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
                
                # 신뢰도가 높은 데이터 포인트 표시
                high_confidence = df[df['confidence'] > 0.8]
                if len(high_confidence) > 0:
                    fig.add_trace(go.Scatter(
                        x=high_confidence['date'],
                        y=high_confidence['stress'],
                        mode='markers',
                        name='높은 신뢰도',
                        marker=dict(color='#667eea', size=8, symbol='star'),
                        hovertemplate='%{x}<br>높은 신뢰도 분석<extra></extra>'
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
            # 감정 분포 (분석 유형별)
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
                        title=f"감정별 빈도 ({analysis_type})",
                        color_discrete_sequence=colors
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    # Plotly 없이 바차트 형태로 표시
                    st.bar_chart(emotion_counts)
        
        # 음성 분석이 포함된 경우 추가 그래프
        voice_data = df[df['has_voice'] == True]
        if len(voice_data) > 0 and PLOTLY_AVAILABLE:
            st.subheader("🎵 음성 분석 트렌드")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 피치 변화
                fig_pitch = px.line(voice_data, x='date', y='pitch_mean',
                                   color='emotions',
                                   title='피치 변화 추이')
                fig_pitch.update_layout(height=300)
                st.plotly_chart(fig_pitch, use_container_width=True)
            
            with col2:
                # 에너지 vs 템포
                fig_energy_tempo = px.scatter(voice_data, x='energy_mean', y='tempo',
                                             color='emotions', size='confidence',
                                             title='음성 에너지 vs 템포')
                fig_energy_tempo.update_layout(height=300)
                st.plotly_chart(fig_energy_tempo, use_container_width=True)
            
            with col3:
                # 텍스트 vs 음성 분석 비교
                comparison_data = voice_data[['date', 'stress', 'voice_stress', 'energy', 'voice_energy']].melt(
                    id_vars=['date'], 
                    value_vars=['stress', 'voice_stress', 'energy', 'voice_energy']
                )
                comparison_data['analysis_type'] = comparison_data['variable'].apply(
                    lambda x: '음성 분석' if 'voice' in x else '텍스트 분석'
                )
                comparison_data['metric'] = comparison_data['variable'].apply(
                    lambda x: '스트레스' if 'stress' in x else '에너지'
                )
                
                fig_comparison = px.line(comparison_data, x='date', y='value',
                                       color='analysis_type', facet_col='metric',
                                       title='텍스트 vs 음성 분석 비교')
                fig_comparison.update_layout(height=300)
                st.plotly_chart(fig_comparison, use_container_width=True)
        
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
                # 신뢰도 분포
                fig_confidence = px.histogram(
                    df,
                    x='confidence',
                    nbins=10,
                    title="분석 신뢰도 분포",
                    color_discrete_sequence=['#667eea']
                )
                fig_confidence.update_layout(height=300)
                st.plotly_chart(fig_confidence, use_container_width=True)
        
        else:
            # Plotly 없이 간단한 차트로 대체
            st.info("더 자세한 그래프를 보려면 plotly 패키지 설치가 필요합니다.")
            
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
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        avg_stress = df['stress'].mean()
        avg_energy = df['energy'].mean()
        avg_mood = df['mood'].mean()
        avg_confidence = df['confidence'].mean()
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
            confidence_emoji = "🎯" if avg_confidence > 0.8 else "📍" if avg_confidence > 0.6 else "📌"
            st.metric(
                "평균 신뢰도",
                f"{confidence_emoji} {avg_confidence:.2f}"
            )
        
        with col5:
            st.metric("분석 기간", f"{total_entries}개 일기")

elif page == "💡 마음 케어":
    st.header("당신만을 위한 마음 케어")
    
    if not st.session_state.diary_entries:
        st.info("이야기를 기록하면 AI가 당신만의 맞춤 케어를 추천해드려요!")
    else:
        # AI 피드백
        with st.spinner("AI가 당신만의 마음 케어 방법을 찾고 있어요..."):
            feedback = generate_personalized_feedback(st.session_state.diary_entries)
        
        st.markdown(f"""
        <div class="feedback-box">
            <h3>AI 마음 케어 코치의 메시지</h3>
            <p style="font-size: 1.1em; line-height: 1.6;">{feedback}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 개인 통계 카드
        st.subheader("나의 마음 여정 리포트")
        
        recent_entries = st.session_state.diary_entries[-30:]
        if recent_entries:
            avg_stress = sum(entry['analysis']['stress_level'] for entry in recent_entries) / len(recent_entries)
            avg_energy = sum(entry['analysis']['energy_level'] for entry in recent_entries) / len(recent_entries)
            avg_mood = sum(entry['analysis']['mood_score'] for entry in recent_entries) / len(recent_entries)
            avg_confidence = sum(entry['analysis'].get('confidence', 0.5) for entry in recent_entries) / len(recent_entries)
            
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
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #ff6b6b;">평균 스트레스</h3>
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
                    <h3 style="color: #51cf66;">평균 활력</h3>
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
                    <h3 style="color: #339af0;">평균 기분</h3>
                    <h2>{avg_mood:.1f}</h2>
                    <p style="color: {'green' if mood_trend > 0 else 'red' if mood_trend < 0 else 'gray'};">
                        {'↗️' if mood_trend > 3 else '↘️' if mood_trend < -3 else '→'} 
                        {abs(mood_trend):.1f} {'개선' if mood_trend > 0 else '하락' if mood_trend < 0 else '안정'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                confidence_emoji = "🎯" if avg_confidence > 0.8 else "📍" if avg_confidence > 0.6 else "📌"
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #667eea;">분석 품질</h3>
                    <h2>{confidence_emoji} {avg_confidence:.2f}</h2>
                    <p style="color: #666;">
                        {'높은 신뢰도' if avg_confidence > 0.8 else '보통 신뢰도' if avg_confidence > 0.6 else '기본 분석'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # 맞춤 웰빙 가이드
        st.subheader("맞춤 웰빙 가이드")
        
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
                    st.success("훌륭해요! 자신을 위한 시간을 가져주셔서 감사합니다.")
                    st.balloons()
        
        # 추가 웰빙 리소스
        st.subheader("추가 웰빙 리소스")
        
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
        st.info("아직 기록된 이야기가 없어요.")
    else:
        # 검색 및 필터
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_query = st.text_input("🔍 이야기 내용 검색", placeholder="찾고 싶은 기억을 검색해보세요")
        
        with col2:
            sort_order = st.selectbox("정렬 순서", ["최신순", "오래된순", "기분 좋은순", "힘들었던순", "신뢰도순"])
        
        with col3:
            voice_filter = st.selectbox("분석 유형", ["전체", "음성 분석 포함", "텍스트만"])
        
        # 데이터 필터링 및 정렬
        filtered_entries = st.session_state.diary_entries.copy()
        
        if search_query:
            filtered_entries = [
                entry for entry in filtered_entries
                if search_query.lower() in entry['text'].lower()
            ]
        
        if voice_filter == "음성 분석 포함":
            filtered_entries = [e for e in filtered_entries if 'voice_analysis' in e.get('analysis', {})]
        elif voice_filter == "텍스트만":
            filtered_entries = [e for e in filtered_entries if 'voice_analysis' not in e.get('analysis', {})]
        
        # 정렬
        if sort_order == "최신순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['timestamp'], reverse=True)
        elif sort_order == "오래된순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['timestamp'])
        elif sort_order == "기분 좋은순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['analysis']['mood_score'], reverse=True)
        elif sort_order == "힘들었던순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['analysis']['mood_score'])
        elif sort_order == "신뢰도순":
            filtered_entries = sorted(filtered_entries, key=lambda x: x['analysis'].get('confidence', 0.5), reverse=True)
        
        if filtered_entries:
            st.write(f"총 {len(filtered_entries)}개의 소중한 이야기를 찾았어요.")
            
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
                    avg_confidence = sum(entry['analysis'].get('confidence', 0.5) for entry in entries) / len(entries)
                    voice_count = sum(1 for entry in entries if 'voice_analysis' in entry.get('analysis', {}))
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("이 달의 평균 마음", f"{avg_mood:.1f}")
                    with col2:
                        st.metric("이 달의 평균 스트레스", f"{avg_stress:.1f}%")
                    with col3:
                        st.metric("이 달의 평균 활력", f"{avg_energy:.1f}%")
                    with col4:
                        confidence_emoji = "🎯" if avg_confidence > 0.8 else "📍" if avg_confidence > 0.6 else "📌"
                        st.metric("평균 신뢰도", f"{confidence_emoji} {avg_confidence:.2f}")
                    with col5:
                        st.metric("음성 분석", f"🎵 {voice_count}개")
                    
                    st.markdown("---")
                    
                    # 해당 월 일기들
                    for entry in entries:
                        mood_emoji = "😊" if entry['analysis']['mood_score'] > 10 else "😐" if entry['analysis']['mood_score'] > -10 else "😔"
                        has_voice = 'voice_analysis' in entry.get('analysis', {})
                        voice_indicator = "🎵" if has_voice else "📝"
                        confidence = entry['analysis'].get('confidence', 0.5)
                        confidence_emoji = "🎯" if confidence > 0.8 else "📍" if confidence > 0.6 else "📌"
                        
                        with st.container():
                            st.markdown(f"""
                            **{voice_indicator} 📅 {entry['date']} {entry['time']} {mood_emoji}**  
                            **마음:** {', '.join(entry['analysis']['emotions'])} {confidence_emoji}  
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
                                    st.info(f"**AI가 읽어드린 마음:** {entry['analysis']['summary']}")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("스트레스", f"{entry['analysis']['stress_level']}%")
                                with col2:
                                    st.metric("활력", f"{entry['analysis']['energy_level']}%")
                                with col3:
                                    st.metric("마음 점수", f"{entry['analysis']['mood_score']}")
                                with col4:
                                    st.metric("분석 신뢰도", f"{confidence:.2f}")
                                
                                # 음성 분석 추가 정보
                                if has_voice:
                                    st.markdown("**🎤 음성 분석 상세:**")
                                    voice_analysis = entry['analysis']['voice_analysis']
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("피치", f"{voice_analysis['voice_features'].get('pitch_mean', 0):.1f} Hz")
                                    with col2:
                                        st.metric("말하기 속도", f"{voice_analysis['voice_features'].get('tempo', 0):.0f} BPM")
                                    with col3:
                                        st.metric("음성 에너지", f"{voice_analysis['voice_features'].get('energy_mean', 0):.3f}")
                                
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
            export_data = []
            for entry in st.session_state.diary_entries:
                row = {
                    'date': entry['date'],
                    'time': entry['time'],
                    'text': entry['text'],
                    'emotions': ', '.join(entry['analysis']['emotions']),
                    'stress_level': entry['analysis']['stress_level'],
                    'energy_level': entry['analysis']['energy_level'],
                    'mood_score': entry['analysis']['mood_score'],
                    'summary': entry['analysis'].get('summary', ''),
                    'keywords': ', '.join(entry['analysis'].get('keywords', [])),
                    'tone': entry['analysis'].get('tone', '중립적'),
                    'confidence': entry['analysis'].get('confidence', 0.5),
                    'has_voice_analysis': 'voice_analysis' in entry.get('analysis', {})
                }
                
                # 음성 분석 데이터 추가
                if 'voice_analysis' in entry.get('analysis', {}):
                    voice_analysis = entry['analysis']['voice_analysis']
                    row.update({
                        'voice_emotions': ', '.join(voice_analysis.get('detected_emotions', [])),
                        'voice_stress': voice_analysis.get('voice_stress_level', ''),
                        'voice_energy': voice_analysis.get('voice_energy_level', ''),
                        'voice_mood': voice_analysis.get('voice_mood_score', ''),
                        'pitch_mean': voice_analysis['voice_features'].get('pitch_mean', ''),
                        'energy_mean': voice_analysis['voice_features'].get('energy_mean', ''),
                        'tempo': voice_analysis['voice_features'].get('tempo', ''),
                        'jitter': voice_analysis['voice_features'].get('jitter', ''),
                        'hnr': voice_analysis['voice_features'].get('hnr', '')
                    })
                
                export_data.append(row)
            
            df_export = pd.DataFrame(export_data)
            
            csv = df_export.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📁 마음 리포트 다운로드",
                data=csv,
                file_name=f"소리로_쓰는_하루_고도화_리포트_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        
        # 백업 저장
        if st.button("💾 전체 이야기 백업"):
            backup_data = {
                'service_name': '소리로 쓰는 하루 - 고도화',
                'version': '2.0',
                'entries': st.session_state.diary_entries,
                'export_date': datetime.now().isoformat(),
                'total_count': len(st.session_state.diary_entries),
                'features': {
                    'voice_analysis': AUDIO_ANALYSIS_AVAILABLE,
                    'praat_analysis': PRAAT_AVAILABLE,
                    'openai_integration': OPENAI_AVAILABLE and openai_client is not None
                }
            }
            backup_json = json.dumps(backup_data, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="📦 백업 파일 다운로드",
                data=backup_json,
                file_name=f"소리로_쓰는_하루_고도화_백업_{datetime.now().strftime('%Y%m%d')}.json",
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
    <p><strong>소리로 쓰는 하루 - 고도화</strong> - 목소리로 담는 오늘, AI가 읽어주는 마음</p>
    <p>음성 피처 분석으로 더 정확하고 풍부한 감정 분석을 제공합니다 ✨</p>
    <small style="color: #999;">Made with ❤️ using Streamlit, OpenAI, Librosa & Praat</small>
</div>
""", unsafe_allow_html=True)
