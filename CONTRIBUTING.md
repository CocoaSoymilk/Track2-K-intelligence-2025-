# 기여 가이드

"하루 소리" 프로젝트에 기여해주셔서 감사합니다! 🎉

## 기여 방법

### 1. 이슈 제보

버그를 발견하거나 새로운 기능을 제안하고 싶으시다면:

1. [Issues](https://github.com/CocoaSoymilk/Track2-K-intelligence-2025-/issues) 탭에서 중복된 이슈가 없는지 확인
2. 새로운 이슈 생성
3. 명확한 제목과 상세한 설명 작성
4. 가능하면 재현 방법이나 스크린샷 첨부

### 2. Pull Request

코드를 직접 기여하고 싶으시다면:

1. **Fork**: 저장소를 자신의 계정으로 Fork
2. **Clone**: Fork한 저장소를 로컬로 Clone
   ```bash
   git clone https://github.com/YOUR_USERNAME/Track2-K-intelligence-2025-.git
   cd Track2-K-intelligence-2025-
   ```

3. **Branch**: 새로운 기능 브랜치 생성
   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **Develop**: 코드 작성 및 테스트
   - 코드 스타일 가이드 준수
   - 주석 작성
   - 테스트 코드 추가 (가능한 경우)

5. **Commit**: 의미있는 커밋 메시지 작성
   ```bash
   git commit -m "feat: 새로운 감정 분석 알고리즘 추가"
   ```

6. **Push**: 변경사항을 자신의 Fork로 Push
   ```bash
   git push origin feature/amazing-feature
   ```

7. **PR**: GitHub에서 Pull Request 생성

## 커밋 메시지 컨벤션

우리는 다음과 같은 커밋 메시지 형식을 사용합니다:

- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `docs:` 문서 수정
- `style:` 코드 포맷팅 (기능 변경 없음)
- `refactor:` 코드 리팩터링
- `test:` 테스트 코드 추가
- `chore:` 빌드 작업, 패키지 관리자 설정 등

**예시:**
```
feat: 주간 리포트 이메일 발송 기능 추가
fix: 음성 녹음 시 크래시 문제 해결
docs: README에 설치 방법 상세 설명 추가
```

## 코드 스타일

### Python
- PEP 8 스타일 가이드 준수
- 함수와 클래스에 Docstring 작성
- 변수명은 의미 있고 명확하게

```python
def analyze_emotion(text: str, audio_features: dict) -> dict:
    """
    텍스트와 오디오 특성을 분석하여 감정을 추출합니다.
    
    Args:
        text: 분석할 텍스트
        audio_features: 오디오 특성 딕셔너리
        
    Returns:
        감정 분석 결과 딕셔너리
    """
    pass
```

### Streamlit
- 사용자 경험(UX)을 최우선으로 고려
- 로딩 상태 표시
- 에러 메시지는 친절하고 명확하게

## 테스트

새로운 기능을 추가할 때는 가능한 한 테스트 코드를 작성해주세요.

```bash
# 테스트 실행
pytest tests/
```

## 질문이 있으신가요?

- Issues 탭에서 질문하기
- Discussions 탭에서 토론하기

## 행동 강령

우리는 모든 기여자를 환영하며, 서로 존중하는 분위기를 유지합니다.

- 친절하고 포용적인 언어 사용
- 다른 관점과 경험 존중
- 건설적인 비판 수용
- 커뮤니티 최선의 이익 고려

---

다시 한번 기여해주셔서 감사합니다! ❤️

