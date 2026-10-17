import io
import base64
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from groq import Groq
import edge_tts
from pydub import AudioSegment

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
PORT = int(os.environ.get("PORT", "10000"))
CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile").strip()
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()


def process_voice_input(wav_bytes: bytes) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY Render ortam değişkeni ayarlanmamış.")
    client = Groq(api_key=GROQ_API_KEY)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(wav_bytes)
        f.flush()
        with open(f.name, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=("robot.wav", audio_file.read()),
                model="whisper-large-v3-turbo",
                language="tr",
            )
    heard_text = (transcript.text or "").strip()
    if not heard_text:
        return {"text": "", "reply": "Seni duyamadım, tekrar söyler misin?", "emotion": "worried"}

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": (
                "Türkçe konuşan sevimli bir robot ol. Kısa ve doğal cevap ver. "
                "Yalnızca geçerli JSON döndür: {\"reply\": string, \"emotion\": string}. "
                "emotion yalnızca normal, happy, sad, angry, surprised, sleepy, thinking, scared, love değerlerinden biri olsun. "
                "Duyguyu konuşmanın içeriğine göre seç; aşırı dramatik olma."
            )},
            {"role": "user", "content": heard_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"reply": raw, "emotion": "normal"}
    reply = str(parsed.get("reply", "Üzgünüm, cevap oluşturamadım."))[:1200]
    emotion = str(parsed.get("emotion", "normal")).lower()
    allowed = {"normal", "happy", "sad", "angry", "surprised", "sleepy", "thinking", "scared", "love"}
    if emotion not in allowed:
        emotion = "normal"
    return {"text": heard_text, "reply": reply, "emotion": emotion}


def process_vision_input(jpeg_bytes: bytes) -> dict:
    """Analyze one JPEG using a vision-capable Groq model."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY Render ortam değişkeni ayarlanmamış.")
    if not jpeg_bytes or len(jpeg_bytes) > 1_500_000:
        raise ValueError("Görüntü boş veya izin verilen boyuttan büyük.")
    if not jpeg_bytes.startswith(b"\xff\xd8"):
        raise ValueError("Gelen veri JPEG görüntüsü değil.")
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": (
                "Türkçe konuşan sevimli bir robot olarak yanıt ver. "
                "Fotoğrafta görünen nesneleri, renkleri ve sahneyi kısa ama faydalı şekilde anlat. "
                "Yalnızca fotoğrafta gerçekten seçilebilen şeyleri söyle; emin değilsen belirsiz olduğunu belirt. "
                "Bir insan görünüyorsa insan bulunduğunu ve genel görünüşünü söyleyebilirsin; "
                "kişinin kimliğini fotoğraftan teşhis etme veya kim olduğunu iddia etme. "
                "Fotoğrafta görünmeyen ayrıntıları uydurma. Türkçe cevap ver, 1-3 cümle kullan."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": "Ne görüyorsun? Nesneleri ve sahneyi tarif et."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ]},
        ],
        temperature=0.2,
        max_tokens=180,
    )
    reply = (completion.choices[0].message.content or "").strip()
    if not reply:
        reply = "Görüntüyü aldım ama net bir şey seçemedim."
    return {"reply": reply[:1000], "emotion": "surprised", "model": VISION_MODEL}


async def generate_wav_bytes(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        mp3_path = os.path.join(td, "speech.mp3")
        wav_path = os.path.join(td, "speech.wav")
        communicate = edge_tts.Communicate(text[:1200], "tr-TR-AhmetNeural")
        await communicate.save(mp3_path)
        audio = AudioSegment.from_file(mp3_path, format="mp3")
        audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")
        with open(wav_path, "rb") as f:
            return f.read()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self._send(200, b"Robot Backend is Live!")

    def do_GET(self):
        self._send(200, b"Robot Backend is Live!")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 2_000_000:
            self._send(400, b'{"error":"Gecersiz istek boyutu"}', "application/json")
            return
        data = self.rfile.read(length)
        try:
            if self.path == "/voice":
                result = process_voice_input(data)
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            elif self.path == "/vision":
                result = process_vision_input(data)
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            elif self.path == "/tts":
                payload = json.loads(data.decode("utf-8"))
                text = str(payload.get("text", "")).strip()
                if not text:
                    self._send(400, b'{"error":"text gerekli"}', "application/json")
                    return
                import asyncio
                wav = asyncio.run(generate_wav_bytes(text))
                self._send(200, wav, "audio/wav")
            else:
                self._send(404, b'{"error":"Not found"}', "application/json")
        except Exception as exc:
            # Detailed exception goes to server logs, not to the client.
            print(f"İstek hatası ({self.path}): {type(exc).__name__}: {exc}", flush=True)
            self._send(500, json.dumps({"error":"Sunucu hatası"}).encode(), "application/json")

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


if __name__ == "__main__":
    print(f"Robot backend {PORT} portunda başlıyor", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
