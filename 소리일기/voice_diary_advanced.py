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
        try:
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
        except:
            return {
                'pitch_mean': 150.0,
                'pitch_std': 20.0,
                'pitch_range': 50.0,
                'pitch_variation': 0.13
            }
    
    def _extract_energy_features(self, y: np.ndarray, sr: int) -> Dict:
        """에너지/강도 관련 특성 추출"""
        try:
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
        except:
            return {
                'energy_mean': 0.1,
                'energy_std': 0.02,
                'energy_max': 0.3,
                'spectral_rolloff_mean': 3000.0
            }
    
    def _extract_tempo_features(self, y: np.ndarray, sr: int) -> Dict:
        """템포/리듬 관련 특성 추출"""
        try:
            # 템포 추정
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Zero Crossing Rate (음성 활동성의 지표)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            return {
                'tempo': float(tempo),
                'zcr_mean': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr))
            }
        except:
            return {
                'tempo': 120.0,
                'zcr_mean': 0.1,
                'zcr_std': 0.05
            }
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """스펙트럼 특성 추출"""
        try:
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
        except:
            return {
                'spectral_centroid_mean': 2000.0,
                'spectral_centroid_std': 500.0,
                'spectral_bandwidth_mean': 1500.0,
                'mfcc_mean': 0.0,
                'mfcc_std': 1.0
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
        """음성 기반 스트레스 레벨 계산 (0-100) - 보조적 역할"""
        base_stress = 20  # 기본값을 낮춤
        
        # 감정별 스트레스 가중치 (완화)
        stress_weights = {
            '분노': 25,      # 35 -> 25
            '불안': 20,      # 30 -> 20  
            '슬픔': 15,      # 20 -> 15
            '기쁨': -20,     # -15 -> -20 (긍정 보상 증가)
            '평온': -25      # -20 -> -25 (평온 보상 증가)
        }
        
        for emotion, weight in stress_weights.items():
            if emotion in emotion_scores:
                base_stress += weight * emotion_scores[emotion] * 0.6  # 음성 영향력 감소
        
        # 음성 피처 기반 조정 (완화)
        if features.get('jitter', 0) > 0.025:  # 임계값 상향 (0.02 -> 0.025)
            base_stress += 8  # 페널티 감소 (15 -> 8)
        
        if features.get('pitch_variation', 0) > 0.30:  # 임계값 상향 (0.25 -> 0.30)
            base_stress += 5  # 페널티 감소 (10 -> 5)
        
        return max(0, min(100, int(base_stress)))
    
    def _calculate_energy_level(self, features: Dict, emotion_scores: Dict) -> int:
        """음성 기반 에너지 레벨 계산 (0-100) - 보조적 역할"""
        base_energy = 55  # 기본값을 약간 상향
        
        # 감정별 에너지 가중치 (긍정 감정 보강)
        energy_weights = {
            '기쁨': 30,      # 25 -> 30 (긍정 보상 증가)
            '분노': 20,      # 30 -> 20 (분노는 에너지보다 스트레스)
            '불안': 5,       # 10 -> 5
            '슬픔': -20,     # -25 -> -20 (완화)
            '평온': 0        # -5 -> 0 (평온은 중립적)
        }
        
        for emotion, weight in energy_weights.items():
            if emotion in emotion_scores:
                base_energy += weight * emotion_scores[emotion] * 0.7  # 음성 영향력 적당히 제한
        
        # 음성 피처 기반 조정 (완화)
        energy_mean = features.get('energy_mean', 0.1)
        if energy_mean > 0.25:  # 임계값 상향 (0.2 -> 0.25)
            base_energy += 10  # 보상 감소 (15 -> 10)
        elif energy_mean < 0.06:  # 임계값 하향 (0.08 -> 0.06)
            base_energy -= 10  # 페널티 감소 (15 -> 10)
        
        tempo = features.get('tempo', 120)
        if tempo > 150:  # 임계값 상향 (140 -> 150)
            base_energy += 8  # 보상 감소 (10 -> 8)
        elif tempo < 90:  # 임계값 하향 (100 -> 90)
            base_energy -= 8  # 페널티 감소 (10 -> 8)
        
        return max(0, min(100, int(base_energy)))
    
    def _calculate_mood_score(self, features: Dict, emotion_scores: Dict) -> int:
        """음성 기반 기분 점수 계산 (-70 to +70) - 보조적 역할"""
        base_mood = 5  # 약간 긍정적 기본값
        
        # 감정별 기분 가중치 (긍정 감정 강화)
        mood_weights = {
            '기쁨': 45,      # 40 -> 45 (긍정 보상 증가)
            '평온': 25,      # 20 -> 25 (평온 보상 증가)
            '슬픔': -25,     # -35 -> -25 (페널티 감소)
            '분노': -20,     # -25 -> -20 (페널티 감소)
            '불안': -15      # -20 -> -15 (페널티 감소)
        }
        
        for emotion, weight in mood_weights.items():
            if emotion in emotion_scores:
                base_mood += weight * emotion_scores[emotion] * 0.5  # 음성 영향력 크게 제한
        
        # 음성 품질 기반 조정 (미미하게)
        hnr = features.get('hnr', 15.0)
        if hnr > 22:  # 임계값 상향 (20 -> 22)
            base_mood += 3  # 보상 감소 (5 -> 3)
        elif hnr < 8:  # 임계값 하향 (10 -> 8)
            base_mood -= 3  # 페널티 감소 (5 -> 3)
        
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
    """텍스트 분석과 음성 분석을 결합하여 최종 분석 결과 생성 - 텍스트 우선"""
    
    # 가중치 설정 (텍스트 우선: 텍스트 80%, 음성 20%)
    text_weight = 0.8
    voice_weight = 0.2
    
    # 감정 결합 (텍스트 감정을 우선하되, 음성은 보조적으로)
    text_emotions = set(text_analysis.get('emotions', []))
    voice_emotions = set(voice_analysis.get('detected_emotions', []))
    
    # 텍스트 감정을 기본으로 하고, 음성은 보조적으로만 추가
    combined_emotions = list(text_emotions)
    for voice_emotion in voice_emotions:
        if voice_emotion not in combined_emotions and len(combined_emotions) < 3:
            combined_emotions.append(voice_emotion)
    
    if not combined_emotions:
        combined_emotions = ['중립']
    
    # 텍스트 기반 긍정성 체크
    text_positive = any(emotion in ['기쁨', '평온'] for emotion in text_emotions)
    text_negative = any(emotion in ['슬픔', '분노', '불안'] for emotion in text_emotions)
    
    # 수치 결합 (텍스트 우선, 음성은 미세 조정)
    text_stress = text_analysis.get('stress_level', 30)
    voice_stress = voice_analysis.get('voice_stress_level', 30)
    text_energy = text_analysis.get('energy_level', 50)
    voice_energy = voice_analysis.get('voice_energy_level', 50)
    text_mood = text_analysis.get('mood_score', 0)
    voice_mood = voice_analysis.get('voice_mood_score', 0)
    
    # 텍스트가 긍정적이면 음성의 부정적 영향 제한
    if text_positive:
        # 긍정적 텍스트인 경우, 음성 영향력 더 축소
        combined_stress = int(text_stress * 0.9 + voice_stress * 0.1)
        combined_energy = int(text_energy * 0.85 + voice_energy * 0.15)
        combined_mood = int(text_mood * 0.85 + voice_mood * 0.15)
        
        # 추가적으로 긍정 보정
        combined_stress = max(10, combined_stress - 10)  # 스트레스 완화
        combined_energy = min(90, combined_energy + 5)   # 에너지 약간 증가
        combined_mood = min(70, combined_mood + 8)       # 기분 개선
        
    elif text_negative:
        # 부정적 텍스트인 경우, 음성이 완화 역할을 할 수 있도록
        combined_stress = int(text_stress * 0.75 + voice_stress * 0.25)
        combined_energy = int(text_energy * 0.75 + voice_energy * 0.25)
        combined_mood = int(text_mood * 0.75 + voice_mood * 0.25)
        
    else:
        # 중립적인 경우, 표준 가중치
        combined_stress = int(text_stress * text_weight + voice_stress * voice_weight)
        combined_energy = int(text_energy * text_weight + voice_energy * voice_weight)
        combined_mood = int(text_mood * text_weight + voice_mood * voice_weight)
    
    # 전체적인 균형 조정 (웰빙 고려)
    # 스트레스가 너무 높으면 완화
    if combined_stress > 70:
        combined_stress = int(combined_stress * 0.85)
    
    # 신뢰도 향상 (음성 분석이 있으면 텍스트 분석의 신뢰도 보강)
    base_confidence = text_analysis.get('confidence', 0.7)
    confidence = min(1.0, base_confidence + 0.15)  # 적당한 신뢰도 증가
    
    return {
        'emotions': combined_emotions[:3],  # 최대 3개 감정
        'stress_level': max(0, min(100, combined_stress)),
        'energy_level': max(0, min(100, combined_energy)),
        'mood_score': max(-70, min(70, combined_mood)),
        'summary': text_analysis.get('summary', '텍스트 기반 종합 분석이 완료되었습니다.'),
        'keywords': text_analysis.get('keywords', []),
        'tone': text_analysis.get('tone', '중립적'),  # 텍스트 톤 우선
        'confidence': confidence,
        'voice_analysis': voice_analysis  # 음성 분석 결과도 저장 (참고용)
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
    # 오늘의 이야기 페이지 내용
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

elif page == "💖 마음 분석":
    st.header("마음 분석 대시보드")
    
    if not st.session_state.diary_entries:
        st.info("📝 아직 기록된 이야기가 없어요. 첫 번째 이야기를 들려주세요!")
    else:
        st.write("마음 분석 페이지 내용...")

elif page == "📈 감정 여정":
    st.header("마음의 변화를 살펴보세요")
    
    if not st.session_state.diary_entries:
        st.info("📊 이야기를 기록하면 마음의 변화를 아름다운 그래프로 볼 수 있어요!")
    else:
        st.write("감정 여정 페이지 내용...")

elif page == "💡 마음 케어":
    st.header("당신만을 위한 마음 케어")
    
    if not st.session_state.diary_entries:
        st.info("📝 이야기를 기록하면 AI가 당신만의 맞춤 케어를 추천해드려요!")
    else:
        st.write("마음 케어 페이지 내용...")

elif page == "📚 나의 이야기들":
    st.header("소중한 이야기 아카이브")
    
    if not st.session_state.diary_entries:
        st.info("📝 아직 기록된 이야기가 없어요.")
    else:
        st.write("이야기 아카이브 페이지 내용...")

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
