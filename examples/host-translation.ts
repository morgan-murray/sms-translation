// The host's authenticated reverse proxy supplies service credentials.
export type SendLanguage = 'english' | 'russian' | 'mandarin' | 'korean';

export async function translateDraft(
  text: string,
  language: SendLanguage,
  signal: AbortSignal,
): Promise<string> {
  if (language === 'english') return text;
  const response = await fetch('/translation/api/v1/translate', {
    method: 'POST',
    credentials: 'same-origin',
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: 'english', dest: language, corpus: text }),
  });
  if (!response.ok) throw new Error(`Translation failed (${response.status})`);
  const result = await response.json();
  if (typeof result.translation !== 'string' || !result.translation.trim()) {
    throw new Error('Translation service returned no text');
  }
  return result.translation;
}

// In the composer, abort the previous controller and clear the reviewed result
// whenever draft/language changes. Also compare a request-generation counter
// before applying async results (including finally/error handlers). Only enable
// Send when the reviewed result belongs to the current draft and language.
// Run existing SDS encoding/length validation on the returned text before send.
