import os
import json
import asyncio
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from groq import Groq
import edge_tts

HOST = "0.0.0.0"
PORT = 5000

# Groq API Key ayarı
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_wIgVXy0Rf1VU7JGo5KxPWGdyb3FYOT9HoR6W6bP6V6CcR...")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """Sen ev robotunun yapay zeka beynisin.
Kullanıcıyla Türkçe, kısa ve doğal konuş.
Matematik sorularını doğru ve adım adım mantığıyla kısa şekilde açıkla.
Genel kültür sorularına net cevaplar ver, sohbet edildiğinde dostça karşılık ver.
Çocukların anlayabileceği sade bir dil kullan.
Her cevabın yanında robotun yüzü için tek bir duygu seç.
Sadece şu JSON'u döndür:
{
  "reply": "kısa Türkçe cevap",
  "emotion": "normal|happy|sad|angry|surprised|thinking|sleepy|scared|love"
}
Başka alan ekleme."""

def ask_groq(text: str) -> dict:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"}
    )
    result = json.loads(completion.choices[0].message.content)
    return {
        "reply": str(result.get("reply", "")),
        "emotion": str(result.get("emotion", "normal"))
    }
  
