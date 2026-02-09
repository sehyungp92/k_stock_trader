"""
Gemini prompt templates for PCIM signal extraction.

PCIM needs structured JSON output for automated processing.
"""

# =============================================================================
# PUNCTUATION PROMPT - Clean up raw transcript
# =============================================================================
PUNCTUATION_PROMPT = """
Punctuate the following transcript in Korean.
Remove any filler words that do not add any meaning to the sentence in Korean.
The focus should be on accuracy and preserving content, where the goal is to produce a clean, readable version of exactly what was said, ie do not summarise, paraphrase, or omit any meaningful phrases.
It is important that you use the correct words and company names, and spell everything correctly. Use the context of the transcript to help.
Add subheadings where relevant.

TRANSCRIPT:
{transcript}
"""

# =============================================================================
# SIGNAL EXTRACTION PROMPT - Produces structured JSON for automated processing
# =============================================================================
SIGNAL_EXTRACTION_PROMPT = """
당신은 한국 주식 시장 전문 분석가입니다. 제공된 유튜브 전사본을 분석하여 화자가 **명시적으로 추천한 종목**만 추출하십시오.

[추출 원칙]
1. **명시적 매수 추천만 포함**: 화자가 직접적인 투자 의견을 밝힌 종목만 포함
2. **단순 언급 제외**: 시황 설명, 과거 성과 언급, 뉴스 전달, 과거 수익률 언급은 추천이 아님
3. **확신도 측정** (0.0 ~ 1.0):
   - 0.9-1.0: 강력 추천, 반복 언급, 단기적으로 주가가 상승할 것인 명확한 이유들을 언급
   - 0.7-0.9: 명확한 매수 추천, "사야 한다", "좋다"
   - 0.5-0.7: 관심 종목, "괜찮아 보인다", "지켜볼 만하다"
   - 0.5 미만: 약한 언급 (제외)
4. **종목코드 생성**: 한국 상장 종목의 6자리 코드를 직접 제공. Use the context of the transcript to help.

[출력 형식]
반드시 아래 JSON 형식으로만 응답하십시오:

```json
{{
  "video_summary": "영상 핵심 내용 1-2문장 요약",
  "recommendations": [
    {{
      "company_name": "삼성전자",
      "ticker": "005930",
      "conviction_score": 0.85
    }},
    {{
      "company_name": "SK하이닉스",
      "ticker": "000660",
      "conviction_score": 0.72
    }},
    {{
      "company_name": "신규상장사",
      "ticker": null,
      "conviction_score": 0.80
    }}
  ]
}}
```

[주의사항]
- 추천 종목이 없으면 recommendations를 빈 배열로 반환
- conviction_score는 0.0~1.0 사이 소수점
- ticker를 모르면 null (후처리에서 해결)
- 과거 성과 언급은 추천이 아님

[전사본]
{transcript}
"""
