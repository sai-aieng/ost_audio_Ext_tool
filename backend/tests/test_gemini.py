"""Gemini contract, timeline, cleanup and API isolation checks (no live API)."""
import json
from pathlib import Path
from uuid import uuid4
import httpx
import pytest
from fastapi import FastAPI
from gemini import service
from gemini.client import Client, GeminiError
from api.routes.gemini import router


def row(start=0, end=5):
    return dict(start_sec=start, end_sec=end, visual_description='Visible scene', purpose_context='Context', on_screen_text=['Hello'])


def test_align_keeps_whisper_verbatim_and_absolute_times():
    transcript = [dict(start_sec=61, end_sec=66, text='Actual words.')]
    rows = service.align_scenes({'scenes':[row(60,65), row(65,70)]},60,70,transcript)
    assert [r['transcribed_dialogue'] for r in rows] == ['Actual words.','Actual words.']
    assert rows[0]['dialogue_segments'] == transcript
    assert rows[0]['start_sec'] == 60


@pytest.mark.parametrize('scenes', [[row(0,5)], [row(60,67),row(65,70)], [row(60,75)]])
def test_rejects_relative_overlapping_or_out_of_bounds_times(scenes):
    with pytest.raises(GeminiError):
        service.align_scenes({'scenes':scenes},60,70,[])


def test_csv_escapes_spreadsheet_formulas(tmp_path):
    item = row()
    item['transcribed_dialogue'] = '=cmd()'
    service.export_csv(tmp_path,[item])
    assert "'=cmd()" in (tmp_path/'analysis.csv').read_text(encoding='utf-8-sig')


def test_provider_error_does_not_expose_response_or_key(monkeypatch):
    client = Client.__new__(Client)
    class Session:
        def request(self, *args, **kwargs):
            class Response:
                ok=False
                status_code=429
                text='secret-key-from-provider'
            return Response()
    client.session=Session()
    with pytest.raises(GeminiError, match='quota') as caught:
        client.request('GET','https://example.test')
    assert 'secret' not in str(caught.value)


def test_failed_job_releases_slot_and_deletes_source(tmp_path, monkeypatch):
    from pipeline import video_ingestion
    video=tmp_path/'input.mp4'
    video.write_bytes(b'bad')
    monkeypatch.setattr(video_ingestion,'validate_video',lambda p: (_ for _ in ()).throw(ValueError('bad video')))
    assert service.SLOT.acquire(blocking=False)
    service.run(str(uuid4()),video,tmp_path,{}, {'include_dialogue':False,'include_faces':False})
    assert json.loads((tmp_path/'status.json').read_text())['status']=='failed'
    assert not video.exists()
    assert service.SLOT.acquire(blocking=False)
    service.SLOT.release()


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
async def test_routes_config_upload_and_download_isolation(tmp_path, monkeypatch):
    from api.routes import gemini
    monkeypatch.setattr(gemini,'settings',lambda: ('private-key','gemini-3-flash-preview'))
    app=FastAPI()
    app.state.base_dir=tmp_path
    app.state.config={'upload':{'output_dir':'output','allowed_extensions':['.mp4'],'max_file_size_mb':1}}
    app.include_router(router,prefix='/api/v1')
    def fake_run(job_id,video,output,config,options):
        assert options=={'include_dialogue':False,'include_faces':True}
        assert video.read_bytes()==b'video'
        service.write_json(output/'status.json',{'status':'completed'})
        service.write_json(output/'analysis.json',{'scenes':[],'faces':[]})
        service.SLOT.release()
    monkeypatch.setattr(service,'run',fake_run)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        configuration=await client.get('/api/v1/gemini/config')
        assert configuration.json()['model']=='gemini-3-flash-preview'
        assert 'private-key' not in configuration.text
        empty=await client.post('/api/v1/gemini/analyze',files={'file':('x.mp4',b'')})
        assert empty.status_code==400
        response=await client.post('/api/v1/gemini/analyze',files={'file':('x.mp4',b'video')},data={'include_dialogue':'false'})
        assert response.status_code==202
        job=response.json()['job_id']
        assert (await client.get('/api/v1/gemini/results/'+job)).status_code==200
        assert (await client.get('/api/v1/gemini/download/'+job+'/env')).status_code==400
        assert (await client.get('/api/v1/gemini/status/not-a-uuid')).status_code==400
