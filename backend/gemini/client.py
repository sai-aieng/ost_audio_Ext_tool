"""Gemini REST client; credentials never enter URLs, results, or logs."""
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

BASE = 'https://generativelanguage.googleapis.com'

class GeminiError(RuntimeError):
    pass


def settings():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env', override=False)
    load_dotenv(root.parent / '.env', override=False)
    key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    model = os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')
    if not re.fullmatch(r'[a-zA-Z0-9._-]+', model):
        raise GeminiError('GEMINI_MODEL must be a model ID, without a URL or models/ prefix.')
    return key, model


class Client:
    def __init__(self):
        key, self.model = settings()
        if not key:
            raise GeminiError('Set GOOGLE_API_KEY or GEMINI_API_KEY in the backend environment.')
        self.session = requests.Session()
        self.session.headers['x-goog-api-key'] = key

    def request(self, method, url, **kwargs):
        try:
            response = self.session.request(method, url, timeout=(20, 300), **kwargs)
        except requests.RequestException:
            raise GeminiError('Google API connection failed or timed out. Retry the analysis.') from None
        if not response.ok:
            messages = {400: 'Google rejected the request. Check model and video compatibility.',
                        401: 'Google API key is invalid.', 403: 'Google API access denied. Check the key and project permissions.',
                        404: 'Gemini model or uploaded file is unavailable. Check GEMINI_MODEL.',
                        429: 'Google API quota exceeded. Check quota/billing or retry later.'}
            raise GeminiError(messages.get(response.status_code, f'Google API returned HTTP {response.status_code}. Retry later.'))
        return response

    def upload(self, video):
        mime = {'.mp4': 'video/mp4', '.mov': 'video/quicktime', '.avi': 'video/x-msvideo', '.mkv': 'video/x-matroska'}[video.suffix.lower()]
        response = self.request('POST', BASE + '/upload/v1beta/files',
            headers={'X-Goog-Upload-Protocol': 'resumable', 'X-Goog-Upload-Command': 'start',
                     'X-Goog-Upload-Header-Content-Length': str(video.stat().st_size),
                     'X-Goog-Upload-Header-Content-Type': mime}, json={'file': {'display_name': 'video-analysis'}})
        url = response.headers.get('X-Goog-Upload-URL', '')
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname != 'generativelanguage.googleapis.com':
            raise GeminiError('Google returned an unexpected upload destination.')
        with video.open('rb') as stream:
            uploaded = self.request('POST', url, data=stream, headers={
                'Content-Length': str(video.stat().st_size), 'X-Goog-Upload-Offset': '0',
                'X-Goog-Upload-Command': 'upload, finalize'}).json()['file']
        return uploaded

    def ready(self, uploaded):
        deadline = time.monotonic() + 600
        while uploaded.get('state') == 'PROCESSING':
            if time.monotonic() > deadline:
                raise GeminiError('Google video preparation timed out. Retry with a shorter video.')
            time.sleep(3)
            uploaded = self.request('GET', BASE + '/v1beta/' + uploaded['name']).json()
        if uploaded.get('state') != 'ACTIVE':
            raise GeminiError('Google could not process this video. Try an MP4 export.')
        return uploaded

    def generate(self, parts, schema):
        response = self.request('POST', BASE + '/v1beta/models/' + self.model + ':generateContent', json={
            'contents': [{'role': 'user', 'parts': parts}],
            'generationConfig': {'temperature': 0.1, 'maxOutputTokens': 16384,
                                 'responseMimeType': 'application/json', 'responseSchema': schema}}).json()
        candidates = response.get('candidates', [])
        if not candidates or candidates[0].get('finishReason') != 'STOP':
            raise GeminiError('Gemini did not return a complete analysis (blocked or output limit). Try a shorter video.')
        try:
            return json.loads(''.join(p.get('text', '') for p in candidates[0]['content']['parts'] if not p.get('thought')))
        except (ValueError, KeyError):
            raise GeminiError('Gemini returned an invalid analysis. Please retry.') from None

    def delete(self, uploaded):
        self.request('DELETE', BASE + '/v1beta/' + uploaded['name'])

    def close(self):
        self.session.close()
