"""Independent Gemini uploads, status, results and downloads."""
import json
from pathlib import Path
from uuid import UUID, uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from api.dependencies import get_base_dir, get_config
from faces.json_io import write_json
from gemini import service
from gemini.client import settings
from utils.file_utils import resolve_configured_path, remove_directory_within

router = APIRouter(prefix='/gemini', tags=['Gemini video analysis'])


def root(config, base_dir):
    return resolve_configured_path(base_dir, config['upload']['output_dir']) / 'gemini'


def directory(job_id, config, base_dir):
    try:
        normalized = str(UUID(job_id))
    except ValueError:
        raise HTTPException(400, 'Invalid analysis ID') from None
    return root(config, base_dir) / normalized


def read_status(output):
    try:
        return json.loads((output / 'status.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise HTTPException(404, 'Analysis not found') from None


@router.get('/config')
def configuration():
    key, model = settings()
    return {'configured': bool(key), 'model': model, 'max_duration_sec': 1200}


@router.post('/analyze', status_code=202)
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                  include_dialogue: bool = Form(True), include_faces: bool = Form(True),
                  config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    submitted = False
    reserved = False
    output = None
    try:
        if not settings()[0]:
            raise HTTPException(503, 'Set GOOGLE_API_KEY or GEMINI_API_KEY in the backend environment.')
        extension = Path(file.filename or '').suffix.lower()
        if extension not in config['upload']['allowed_extensions']:
            raise HTTPException(400, 'Choose an MP4, AVI, MKV, or MOV video.')
        if not service.SLOT.acquire(blocking=False):
            raise HTTPException(429, 'A Gemini analysis is already running. Wait for it to finish.')
        reserved = True
        job_id = str(uuid4())
        output = directory(job_id, config, base_dir)
        output.mkdir(parents=True)
        video = output / ('input' + extension)
        size = 0
        async with aiofiles.open(video, 'wb') as stream:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > int(config['upload']['max_file_size_mb']) * 1024 * 1024:
                    raise HTTPException(413, 'Video exceeds the configured upload limit.')
                await stream.write(chunk)
        if not size:
            raise HTTPException(400, 'The uploaded video is empty.')
        write_json(output / 'status.json', {'job_id': job_id, 'status': 'queued', 'stage': 'Waiting to start', 'elapsed_sec': 0})
        background_tasks.add_task(service.run, job_id, video, output, config,
                                  {'include_dialogue': include_dialogue, 'include_faces': include_faces})
        submitted = True
        return {'job_id': job_id, 'status': 'queued'}
    finally:
        await file.close()
        if not submitted:
            if reserved:
                service.SLOT.release()
            if output is not None:
                remove_directory_within(output, root(config, base_dir))


@router.get('/status/{job_id}')
def status(job_id: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    return read_status(directory(job_id, config, base_dir))


def completed(job_id, config, base_dir):
    output = directory(job_id, config, base_dir)
    if read_status(output)['status'] != 'completed':
        raise HTTPException(409, 'Analysis is not complete.')
    return output


@router.get('/results/{job_id}')
def results(job_id: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    output = completed(job_id, config, base_dir)
    return json.loads((output / 'analysis.json').read_text(encoding='utf-8'))


@router.get('/download/{job_id}/{format}')
def download(job_id: str, format: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    if format not in {'json', 'csv'}:
        raise HTTPException(400, 'Choose JSON or CSV.')
    output = completed(job_id, config, base_dir)
    return FileResponse(output / ('analysis.' + format), filename=f'video-analysis-{job_id}.{format}',
                        media_type='application/json' if format == 'json' else 'text/csv')


@router.get('/images/{job_id}/{face_id}')
def image(job_id: str, face_id: str, config: dict = Depends(get_config), base_dir: Path = Depends(get_base_dir)):
    result = results(job_id, config, base_dir)
    if face_id not in {face['id'] for face in result['faces']}:
        raise HTTPException(404, 'Face photo not found.')
    return FileResponse(directory(job_id, config, base_dir) / (face_id + '.jpg'),
                        media_type='image/jpeg', filename=face_id + '.jpg')
