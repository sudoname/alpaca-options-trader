// GET-only, same-origin client for the Oracle JSON API. Mirrors app.js `api()`:
// any network/parse failure resolves to an ERROR envelope so widgets degrade
// to a badge instead of throwing.
import { type ApiEnvelope } from './types'

export async function apiGet<T extends ApiEnvelope>(path: string): Promise<T> {
  try {
    const res = await fetch('/api/' + path, {
      headers: { Accept: 'application/json' },
    })
    const json = (await res.json()) as T
    return json
  } catch (e) {
    return { verdict: 'ERROR', error: String(e) } as T
  }
}
