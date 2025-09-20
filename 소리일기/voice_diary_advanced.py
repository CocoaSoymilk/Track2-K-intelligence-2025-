# GitHub 리포지토리 구조가 그대로 Streamlit Cloud에 배포됨
# 따라서 GitHub에 PDF 파일이 업로드되어 있어야 함

pdf_path = "소리일기/심리 건강 관리 정리 파일.pdf"

import streamlit as st
import os

def check_file_structure():
    st.subheader("파일 구조 확인")
    
    # 현재 작업 디렉토리 확인
    current_dir = os.getcwd()
    st.write(f"현재 디렉토리: {current_dir}")
    
    # 파일 목록 확인
    files = os.listdir(".")
    st.write("루트 디렉토리 파일들:", files)
    
    # 소리일기 폴더 확인
    if "소리일기" in files:
        diary_files = os.listdir("소리일기")
        st.write("소리일기 폴더 내용:", diary_files)
        
        # PDF 파일 확인
        pdf_path = "소리일기/심리 건강 관리 정리 파일.pdf"
        if os.path.exists(pdf_path):
            st.success(f"✅ PDF 파일 발견: {pdf_path}")
            file_size = os.path.getsize(pdf_path)
            st.write(f"파일 크기: {file_size / 1024:.2f} KB")
            return pdf_path
        else:
            st.error("❌ PDF 파일을 찾을 수 없습니다")
    else:
        st.error("❌ '소리일기' 폴더를 찾을 수 없습니다")
    
    return None

# 사용법
pdf_path = check_file_structure()
