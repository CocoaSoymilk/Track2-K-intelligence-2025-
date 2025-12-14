# 🎙️ 하루 소리 - AI 마음 챙김 플랫폼

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **KT Track2 K-intelligence 2025** 프로젝트  
> AI 기반 음성 일기 및 심리 건강 관리 플랫폼

## 📖 프로젝트 소개

"하루 소리"는 음성 인식과 AI 기술을 활용하여 사용자의 일상을 기록하고 감정을 분석하는 스마트 마음 챙김 플랫폼입니다. 단순한 일기 작성을 넘어, 음성 운율 분석과 AI 코칭을 통해 정서적 안정과 자기 성찰을 돕습니다.

### ✨ 주요 기능

#### 🎤 음성 일기 작성
- **음성 녹음 및 자동 텍스트 변환**: OpenAI Whisper를 활용한 고정밀 음성 인식
- **다중 오디오 파일 업로드**: MP3, WAV, M4A 등 다양한 형식 지원
- **실시간 녹음**: 브라우저에서 직접 음성 녹음 가능

#### 🧠 AI 감정 분석
- **음성 운율 분석**: Praat Parselmouth를 활용한 피치, 강도, 발화 속도 분석
- **감정 키워드 추출**: GPT-4 기반 자연어 처리로 핵심 감정 파악
- **베이스라인 비교**: 개인별 평소 음성 패턴과의 차이 분석

#### 📊 시각화 대시보드
- **감정 트렌드**: 시간대별 감정 변화 추이 그래프
- **단어 클라우드**: 자주 사용하는 감정 표현 시각화
- **주간/월간 리포트**: 정서적 패턴 요약 및 인사이트 제공

#### 💬 AI 코칭
- **개인화된 피드백**: 사용자의 감정 상태에 맞춘 맞춤형 조언
- **코칭 톤 설정**: 따뜻함/간결함/도전적 스타일 선택 가능
- **목표 기반 가이드**: 설정한 목표에 따른 실천 가능한 제안

#### 📚 지식 기반 RAG
- **PDF 업로드**: 심리 건강 관련 자료를 시스템에 추가
- **맥락 기반 응답**: 업로드한 자료를 바탕으로 더욱 정확한 조언 제공

#### 🔐 데이터 보안
- **로컬 저장**: 모든 일기 데이터는 사용자의 로컬 환경에만 저장
- **암호화 지원**: 민감한 정보 보호를 위한 데이터 암호화

## 🚀 시작하기

### 필수 요구사항

- Python 3.8 이상
- OpenAI API Key (GPT-4 및 Whisper 사용)

### 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/CocoaSoymilk/Track2-K-intelligence-2025-.git
cd Track2-K-intelligence-2025-/소리일기
```

2. **의존성 설치**
```bash
pip install -r requirements.txt
```

3. **환경 설정**
- OpenAI API Key 준비
- 첫 실행 시 앱 내에서 API Key 입력

### 실행 방법

```bash
streamlit run voice_diary_advanced.py
```

브라우저에서 자동으로 `http://localhost:8501`이 열립니다.

## 📂 프로젝트 구조

```
Track2-K-intelligence-2025-/
├── 소리일기/
│   ├── voice_diary_advanced.py  # 메인 애플리케이션
│   ├── requirements.txt         # Python 패키지 목록
│   ├── 심리 건강 관리 정리 파일.pdf  # 샘플 지식 베이스
│   └── 삭제/                    # 임시 파일
├── .devcontainer/              # VSCode 개발 컨테이너 설정
└── README.md                   # 프로젝트 문서
```

## 🛠️ 기술 스택

### Frontend
- **Streamlit**: 웹 애플리케이션 프레임워크
- **Altair**: 인터랙티브 차트 및 시각화

### AI & ML
- **OpenAI GPT-4**: 자연어 이해 및 AI 코칭
- **OpenAI Whisper**: 음성-텍스트 변환 (STT)
- **Librosa**: 오디오 신호 처리
- **Praat Parselmouth**: 음성 운율 분석

### Data Processing
- **Pandas**: 데이터 처리 및 분석
- **NumPy**: 수치 연산
- **PyPDF2**: PDF 문서 파싱

### Audio Processing
- **Soundfile**: 오디오 파일 입출력
- **WebRTC VAD**: 음성 활동 감지

## 📊 주요 기능 상세

### 1. 음성 운율 분석
```python
# 피치 (음높이)
- 평균 피치: 감정의 활성화 수준 파악
- 피치 변동성: 감정의 변화 폭 측정

# 강도 (음량)
- 평균 강도: 에너지 수준 측정
- 강도 변동: 감정 표현의 역동성

# 발화 속도
- 말하기 속도: 긴장감 또는 안정감 지표
```

### 2. AI 감정 분석
- **감정 키워드 자동 추출**: "기쁨", "불안", "평온" 등 핵심 감정 단어 탐지
- **감정 강도 측정**: 0-10 스케일로 감정의 세기 평가
- **맥락 이해**: 전체 문장의 의미를 파악하여 정확한 감정 분석

### 3. 개인화 코칭
```python
코칭 톤 옵션:
- 따뜻함: 공감적이고 지지적인 메시지
- 간결함: 핵심만 전달하는 실용적 조언
- 도전적: 동기부여와 성장을 독려하는 메시지
```

## 📈 사용 예시

### 음성 일기 작성 흐름
1. **녹음 또는 파일 업로드**: 오늘의 생각과 감정을 음성으로 기록
2. **자동 텍스트 변환**: AI가 음성을 텍스트로 자동 변환
3. **감정 분석**: 음성 운율과 텍스트를 분석하여 감정 상태 파악
4. **AI 피드백**: 개인화된 조언과 응원 메시지 제공
5. **저장 및 리뷰**: 일기를 저장하고 과거 기록과 비교


