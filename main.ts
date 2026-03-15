import { Hono } from "hono";
import { cors } from "hono/cors";

const app = new Hono();

// Enable CORS for all routes
app.use("*", cors({
  origin: "*",
  allowMethods: ["GET", "POST", "OPTIONS"],
  allowHeaders: ["Content-Type", "Authorization"],
}));

// Health check endpoint
app.get("/health", (c) => {
  return c.json({ status: "ok", timestamp: new Date().toISOString() });
});

/**
 * POST /api/transcribe
 * 
 * Proxy endpoint for Deepgram or Whisper transcription
 * 
 * Request body (multipart/form-data):
 *   - audio: Blob (WAV or MP3 audio file)
 *   - apiKey: string (user's Deepgram or Whisper API key)
 *   - engine: "deepgram" | "whisper"
 *   - language?: string (optional language code, e.g., "zh", "en")
 * 
 * Response:
 *   - language: string (detected or specified language)
 *   - segments: Array<{start: number, end: number, text: string}>
 *   - srtContent: string (formatted SRT content)
 *   - segmentsTraditional?: Array (if Chinese)
 *   - srtContentTraditional?: string (if Chinese)
 */
app.post("/api/transcribe", async (c) => {
  try {
    const formData = await c.req.formData();
    const audioBlob = formData.get("audio") as File;
    const apiKey = formData.get("apiKey") as string;
    const engine = formData.get("engine") as string;
    const language = (formData.get("language") as string) || undefined;

    if (!audioBlob) {
      return c.json({ error: "No audio file provided" }, 400);
    }

    if (!apiKey) {
      return c.json({ error: "API key not provided" }, 400);
    }

    if (!engine || !["deepgram", "whisper"].includes(engine)) {
      return c.json({ error: "Invalid engine specified" }, 400);
    }

    let result;

    if (engine === "deepgram") {
      result = await transcribeWithDeepgram(audioBlob, apiKey, language);
    } else {
      result = await transcribeWithWhisper(audioBlob, apiKey, language);
    }

    return c.json(result);
  } catch (error) {
    console.error("[/api/transcribe] Error:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    return c.json({ error: errorMessage }, 500);
  }
});

/**
 * Transcribe audio using Deepgram API
 */
async function transcribeWithDeepgram(
  audioBlob: File,
  apiKey: string,
  language?: string
): Promise<any> {
  // Determine model and language parameters
  const isChineseRequest = language && language.toLowerCase().startsWith("zh");
  const model = isChineseRequest ? "nova-2" : "nova-3";
  const langParam = isChineseRequest ? "zh-TW" : language || "auto";

  // Build query parameters
  const params = new URLSearchParams({
    model,
    language: langParam,
    utterances: "true",
    punctuate: "true",
    smart_format: "true",
  });

  if (langParam === "auto") {
    params.append("detect_language", "true");
  }

  // Convert Blob to ArrayBuffer
  const arrayBuffer = await audioBlob.arrayBuffer();

  // Call Deepgram API
  const response = await fetch(`https://api.deepgram.com/v1/listen?${params}`, {
    method: "POST",
    headers: {
      "Authorization": `Token ${apiKey}`,
      "Content-Type": audioBlob.type || "audio/wav",
    },
    body: arrayBuffer,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Deepgram API error: ${response.status} ${errorText}`);
  }

  const deepgramResponse = await response.json();

  // Parse response and build segments
  const segments: Array<{ start: number; end: number; text: string }> = [];
  let detectedLanguage = "unknown";

  // Try to get detected language from results
  if (
    deepgramResponse.results?.channels?.[0]?.detected_language
  ) {
    detectedLanguage = deepgramResponse.results.channels[0].detected_language;
  } else if (deepgramResponse.metadata?.detected_language) {
    detectedLanguage = deepgramResponse.metadata.detected_language;
  }

  // Extract utterances (sentence-level segments with timestamps)
  const utterances = deepgramResponse.results?.channels?.[0]?.utterances || [];

  if (utterances.length > 0) {
    for (const utterance of utterances) {
      let text = utterance.transcript || "";
      // Remove spaces between Chinese characters
      text = removeChineseSpaces(text);

      segments.push({
        start: Math.round(utterance.start * 1000),
        end: Math.round(utterance.end * 1000),
        text,
      });
    }
  } else {
    // Fallback: aggregate words into chunks if no utterances available
    const words = deepgramResponse.results?.channels?.[0]?.words || [];
    let currentChunk: { start: number; end: number; words: string[] } | null =
      null;

    for (const word of words) {
      if (!currentChunk) {
        currentChunk = {
          start: Math.round(word.start * 1000),
          end: Math.round(word.end * 1000),
          words: [word.punctuated_word || word.word],
        };
      } else if (word.start - currentChunk.end / 1000 < 5) {
        // Within 5 seconds, add to current chunk
        currentChunk.end = Math.round(word.end * 1000);
        currentChunk.words.push(word.punctuated_word || word.word);
      } else {
        // Start new chunk
        let text = currentChunk.words.join(" ");
        text = removeChineseSpaces(text);
        segments.push({
          start: currentChunk.start,
          end: currentChunk.end,
          text,
        });
        currentChunk = {
          start: Math.round(word.start * 1000),
          end: Math.round(word.end * 1000),
          words: [word.punctuated_word || word.word],
        };
      }
    }

    if (currentChunk) {
      let text = currentChunk.words.join(" ");
      text = removeChineseSpaces(text);
      segments.push({
        start: currentChunk.start,
        end: currentChunk.end,
        text,
      });
    }
  }

  // Build SRT content
  const srtContent = buildSrtContent(segments);

  // Handle Chinese: generate traditional version
  let segmentsTraditional: typeof segments | undefined;
  let srtContentTraditional: string | undefined;

  if (detectedLanguage.toLowerCase().startsWith("zh")) {
    const { convertToTraditional } = await import(
      "https://cdn.jsdelivr.net/npm/opencc-js@1.0.2/dist/opencc.min.js"
    );
    segmentsTraditional = segments.map((seg) => ({
      ...seg,
      text: convertToTraditional(seg.text),
    }));
    srtContentTraditional = buildSrtContent(segmentsTraditional);
  }

  return {
    language: detectedLanguage,
    segments,
    srtContent,
    segmentsTraditional,
    srtContentTraditional,
  };
}

/**
 * Transcribe audio using Whisper API
 */
async function transcribeWithWhisper(
  audioBlob: File,
  apiKey: string,
  language?: string
): Promise<any> {
  const formData = new FormData();
  formData.append("file", audioBlob);
  formData.append("model", "whisper-1");
  formData.append("response_format", "verbose_json");

  if (language && language !== "auto") {
    formData.append("language", language);
  }

  const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Whisper API error: ${response.status} ${errorText}`);
  }

  const whisperResponse = await response.json();

  // Parse Whisper response
  const segments: Array<{ start: number; end: number; text: string }> = [];
  const detectedLanguage = whisperResponse.language || "unknown";

  if (whisperResponse.segments && Array.isArray(whisperResponse.segments)) {
    for (const seg of whisperResponse.segments) {
      segments.push({
        start: Math.round(seg.start * 1000),
        end: Math.round(seg.end * 1000),
        text: seg.text.trim(),
      });
    }
  } else if (whisperResponse.text) {
    // Fallback: single segment
    segments.push({
      start: 0,
      end: Math.round((whisperResponse.duration || 0) * 1000),
      text: whisperResponse.text,
    });
  }

  const srtContent = buildSrtContent(segments);

  // Handle Chinese: generate traditional version
  let segmentsTraditional: typeof segments | undefined;
  let srtContentTraditional: string | undefined;

  if (detectedLanguage.toLowerCase().startsWith("zh")) {
    const { convertToTraditional } = await import(
      "https://cdn.jsdelivr.net/npm/opencc-js@1.0.2/dist/opencc.min.js"
    );
    segmentsTraditional = segments.map((seg) => ({
      ...seg,
      text: convertToTraditional(seg.text),
    }));
    srtContentTraditional = buildSrtContent(segmentsTraditional);
  }

  return {
    language: detectedLanguage,
    segments,
    srtContent,
    segmentsTraditional,
    srtContentTraditional,
  };
}

/**
 * Build SRT format content from segments
 */
function buildSrtContent(
  segments: Array<{ start: number; end: number; text: string }>
): string {
  return segments
    .map((seg, idx) => {
      const startTime = formatSrtTime(seg.start);
      const endTime = formatSrtTime(seg.end);
      return `${idx + 1}\n${startTime} --> ${endTime}\n${seg.text}`;
    })
    .join("\n\n");
}

/**
 * Format milliseconds to SRT time format (HH:MM:SS,mmm)
 */
function formatSrtTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const milliseconds = ms % 1000;

  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")},${String(milliseconds).padStart(3, "0")}`;
}

/**
 * Remove spaces between Chinese characters
 * Keeps spaces between English words and Chinese characters
 */
function removeChineseSpaces(text: string): string {
  // Use Unicode property to detect Chinese characters
  // Pattern: Chinese char followed by space(s) followed by another Chinese char
  return text.replace(/(\p{Script=Han})\s+(?=\p{Script=Han})/gu, "$1");
}

// Start server
const port = parseInt(Deno.env.get("PORT") || "8000");
console.log(`🚀 Server running on http://localhost:${port}`);
Deno.serve({ port }, app.fetch);
