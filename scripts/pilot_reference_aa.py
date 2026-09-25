"""Local, evidence-backed review and guarded publication of pilot AA references."""
from __future__ import annotations

import argparse
import bisect
import copy
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from pipeline.atomic import atomic_copy, atomic_write_json
from pipeline.utils.video import get_video_info


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data/pilot_study.csv'
LEDGER = ROOT / 'benchmark/pilot_reference_aa.json'
WORK = ROOT / 'outputs/pilot_reference_aa'
EVIDENCE = ROOT / 'outputs/pilot_reference_review'
COLUMN = '判断依据(AI润色版)'
CALIBRATION = ['AA-train-740', 'AA-test-280', 'AA-train-508', 'AA-test-188', 'AA-train-1920']
ROOTS = {'train': 'SoccerNet-AA_clean_train_valid', 'test': 'SoccerNet-AA_clean_test_challenge'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def key(row):
    return tuple(row[field] for field in ('clip', 'tactic', 'interval'))


def parse_csv(raw):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    rows = list(reader)
    assert all(set(row) == set(reader.fieldnames) and None not in row.values() for row in rows)
    return reader.fieldnames, rows


def window(value):
    result = []
    for stamp in value.split('-'):
        minute, second = map(int, stamp.split(':'))
        assert 0 <= second < 60
        result.append(minute * 60 + second)
    assert len(result) == 2 and 0 <= result[0] < result[1]
    return result


def initialize():
    assert shutil.which('ffmpeg') and shutil.which('ffprobe')
    raw = CSV.read_bytes()
    fields, rows = parse_csv(raw)
    assert fields == ['clip', 'tactic', 'interval', '判断依据', COLUMN]
    assert len(rows) == len({key(row) for row in rows}) == 261
    targets = [row for row in rows if row['clip'].startswith('AA-')]
    clips = list(dict.fromkeys(row['clip'] for row in targets))
    assert len(targets) == 156 and len(clips) == 155
    assert all(not row[COLUMN] for row in targets)
    assert sum(bool(row[COLUMN]) for row in rows) == 104
    assert raw.startswith(b'\xef\xbb\xbf') and raw.count(b'\r\n') == raw.count(b'\n') == 262
    if LEDGER.exists():
        assert load(LEDGER)['inventory']['source_sha256'] == sha(raw)
        print('Existing AA baseline retained')
        return
    videos, issues = {}, []
    for clip in clips:
        prefix, split, number = clip.split('-')
        assert prefix == 'AA' and split in ROOTS and number.isdigit()
        path = ROOT / 'data' / ROOTS[split] / split / 'clean' / f'AA-{number}.mp4'
        entry = {'path': str(path.relative_to(ROOT)), 'split': split}
        if not path.is_file():
            entry['ingest_status'] = 'source_video_missing'
            issues.append({'clip': clip, 'reason': entry['ingest_status'], 'expected_path': entry['path']})
        else:
            entry.update(get_video_info(path))
            stream = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=avg_frame_rate,start_time', '-of', 'json', str(path)]))['streams'][0]
            entry.update(avg_frame_rate=stream['avg_frame_rate'], stream_start_time=float(stream.get('start_time', 0)),
                         size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, ingest_status='ready')
            for row in targets:
                if row['clip'] == clip and window(row['interval'])[1] > entry['duration_s'] + 1 / entry['fps']:
                    entry['ingest_status'] = 'interval_out_of_bounds'
                    issues.append({'clip': clip, 'reason': entry['ingest_status'], 'interval': row['interval']})
        videos[clip] = entry
    available = [clip for clip in clips if videos[clip]['ingest_status'] == 'ready']
    calibration = [clip for clip in CALIBRATION if clip in available]
    remaining = [clip for clip in available if clip not in calibration]
    batches = {'A0': calibration}
    batches.update({f'A{offset // 10 + 1:02d}': remaining[offset:offset + 10] for offset in range(0, len(remaining), 10)})
    WORK.mkdir(parents=True, exist_ok=True)
    baseline = WORK / 'baseline.csv'
    if baseline.exists():
        assert baseline.read_bytes() == raw
    else:
        atomic_copy(CSV, baseline)
    protected = ['benchmark/pilot_reference_gsr.json', 'benchmark/pilot_spatial_coarse.csv']
    ledger = {
        'schema_version': 'pilot-reference-aa-draft-v1',
        'inventory': {'source_csv': 'data/pilot_study.csv', 'source_sha256': sha(raw),
                      'baseline_path': str(baseline.relative_to(ROOT)), 'source_fields': fields,
                      'total_rows': len(rows), 'source_rows': targets, 'encoding': 'utf-8-sig', 'newline': '\r\n',
                      'videos': videos, 'batches': batches,
                      'protected_sha256': {path: sha((ROOT / path).read_bytes()) for path in protected}},
        'ingest_issues': issues, 'completed_batches': [], 'records': [],
    }
    atomic_write_json(LEDGER, ledger)
    assert load(LEDGER) == ledger and baseline.read_bytes() == raw
    print(f'PASS preflight: {len(rows)} rows; {len(targets)} AA rows; {len(available)}/{len(clips)} videos available')
    print('Media:', sorted({(entry['fps'], entry['duration_s'], entry['width'], entry['height'])
                              for entry in videos.values() if entry['ingest_status'] == 'ready'}))
    print('Blocked:', issues)
    print('Batches:', {batch: len(members) for batch, members in batches.items()})


def media(clip):
    ledger = load(LEDGER)
    video = ledger['inventory']['videos'][clip]
    assert video['ingest_status'] == 'ready'
    path = ROOT / video['path']
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (video['size_bytes'], video['mtime_ns'])
    directory = EVIDENCE / clip
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'review_frames.json'
    if manifest_path.exists():
        manifest = load(manifest_path)
    else:
        frames = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)]))['frames']
        pts = [float(frame['best_effort_timestamp_time']) - video['stream_start_time'] for frame in frames]
        assert pts and pts == sorted(pts) and abs(pts[0]) <= 1 / video['fps']
        manifest = {'video_path': video['path'], 'time_origin': 'video_start', 'frame_timestamps_s': pts, 'sheets': {}}
        atomic_write_json(manifest_path, manifest)
    return video, path, manifest_path, manifest


def sheets(clip, prefix, start, end, sample_fps=None, *, crop=None):
    video, path, manifest_path, manifest = media(clip)
    crop_filter = ''
    if crop is not None:
        left, top, width, height = crop
        assert 0 <= left < left + width <= video['width']
        assert 0 <= top < top + height <= video['height']
        crop_filter = f'crop={width}:{height}:{left}:{top},'
    pts = manifest['frame_timestamps_s']
    end = min(end, video['duration_s'])
    if sample_fps is None:
        indices = list(range(bisect.bisect_left(pts, start), bisect.bisect_left(pts, end)))
        columns, rows = 5, 5
    else:
        requested = [start + index / sample_fps for index in range(math.ceil((end - start) * sample_fps))]
        indices = sorted({bisect.bisect_left(pts, stamp) for stamp in requested if stamp < end})
        indices = [index for index in indices if index < len(pts) and pts[index] < end]
        columns, rows = (3, 5) if sample_fps < 1 else (4, 5)
    assert indices
    capacity = columns * rows
    names = [f'{prefix}-{index + 1:02d}.jpg' for index in range(math.ceil(len(indices) / capacity))]
    directory = manifest_path.parent
    if not all(name in manifest['sheets'] and (directory / name).is_file() for name in names):
        assert not any((directory / name).exists() for name in names), 'Partial extraction; inspect before replacing'
        selection = '+'.join(f'eq(n,{index})' for index in indices)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-xerror', '-n',
                        '-i', str(path), '-vf', f"select='{selection}',{crop_filter}scale=640:-2,tile={columns}x{rows}",
                        '-fps_mode', 'vfr', '-q:v', '3', str(directory / f'{prefix}-%02d.jpg')], check=True)
        for index, name in enumerate(names):
            actual = indices[index * capacity:(index + 1) * capacity]
            assert (directory / name).is_file()
            manifest['sheets'][name] = {'kind': 'native' if sample_fps is None else 'sampled',
                                      'sample_fps': sample_fps, 'requested_span_s': [start, end],
                                      'frame_indices': actual, 'timestamps_s': [pts[position] for position in actual],
                                      'layout': [columns, rows], 'frame_width': 640,
                                      'crop_xywh': list(crop) if crop is not None else None,
                                      'sha256': sha((directory / name).read_bytes())}
        atomic_write_json(manifest_path, manifest)
    for name in names:
        assert manifest['sheets'][name].get('crop_xywh') == (list(crop) if crop is not None else None)
    print(clip, prefix, [(name, manifest['sheets'][name]['timestamps_s'][0], manifest['sheets'][name]['timestamps_s'][-1]) for name in names])


def prepare(batch):
    ledger = load(LEDGER)
    for clip in ledger['inventory']['batches'][batch]:
        duration = ledger['inventory']['videos'][clip]['duration_s']
        sheets(clip, 'overview', 0, duration, 0.5)
        for interval in dict.fromkeys(row['interval'] for row in ledger['inventory']['source_rows'] if row['clip'] == clip):
            start, end = window(interval)
            sheets(clip, f'window-{start:02d}-{end:02d}', max(0, start - 2), min(duration, end + 2), 4)
    print('PASS extraction', batch)


def validate(ledger, complete=False):
    inventory = ledger['inventory']
    sources = {key(row): row for row in inventory['source_rows']}
    available = {source_key for source_key in sources if inventory['videos'][source_key[0]]['ingest_status'] == 'ready'}
    keys = [key(record['source']) for record in ledger['records']]
    assert len(keys) == len(set(keys)) and set(keys) <= available
    if complete:
        assert set(keys) == available, sorted(available - set(keys))
    shared = {}
    for record in ledger['records']:
        source, annotation, output = record['source'], record['annotation'], record['output']
        assert source['original_rationale'] == sources[key(source)]['判断依据']
        assert source['video_path'] == inventory['videos'][source['clip']]['path']
        start, end = window(source['interval'])
        assert source['target_window_s'] == output['target_window_s'] == [start, end]
        facts = {item['id'] for item in output['facts']}
        relations = {item['id'] for item in output['relations']}
        assert facts and relations
        for item in output['facts'] + output['relations'] + output['uncertainties']:
            assert start <= item['span_s'][0] < item['span_s'][1] <= end and item['text'].strip()
        for item in output['relations'] + [output['interpretation'], output['commentary']]:
            assert item['text'].strip() and set(item['evidence_ids']) <= facts
        assert set(output['interpretation']['relation_ids']) <= relations
        assert annotation['status'] == 'machine_draft'
        assert annotation['human_reviewer'] is None and annotation['human_reviewed_at'] is None
        assert annotation['viewed_artifacts'] and annotation['native_reviewed_spans_s']
        manifest = load(ROOT / annotation['frame_manifest'])
        for artifact in annotation['viewed_artifacts']:
            path = ROOT / artifact
            assert path.is_file() and sha(path.read_bytes()) == manifest['sheets'][path.name]['sha256']
        assert record['review']['spatial_checks']['basis'].strip()
        shared_key = (source['clip'], source['interval'])
        assert shared.setdefault(shared_key, output['facts']) == output['facts'], shared_key
    for path, checksum in inventory['protected_sha256'].items():
        assert sha((ROOT / path).read_bytes()) == checksum, 'Protected GSR artifact changed: ' + path


def record_batch(batch):
    ledger = load(LEDGER)
    notes = load(WORK / 'drafts' / f'{batch}.json')
    inventory = ledger['inventory']
    sources = {key(row): row for row in inventory['source_rows'] if row['clip'] in inventory['batches'][batch]}
    assert len(notes) == len(sources) and {key(note) for note in notes} == set(sources)
    additions = []
    for note in notes:
        original = sources[key(note)]
        video = inventory['videos'][note['clip']]
        start, end = window(note['interval'])
        directory = EVIDENCE / note['clip']
        manifest = load(directory / 'review_frames.json')
        viewed = note['viewed']
        assert len(viewed) == len(set(viewed)) and set(viewed) <= set(manifest['sheets'])
        required = {name for name in manifest['sheets'] if name.startswith(('overview-', f'window-{start:02d}-{end:02d}-'))}
        assert required <= set(viewed), ('Review all overview/window sheets', sorted(required - set(viewed)))
        native = [manifest['sheets'][name]['requested_span_s'] for name in viewed if manifest['sheets'][name]['kind'] == 'native']
        assert native
        facts = [{'id': f'f{index}', 'span_s': [first, last], 'text': text}
                 for index, (first, last, text) in enumerate(note['facts'], 1)]
        relations = [{'id': f'r{index}', 'span_s': [first, last], 'text': text,
                      'evidence_ids': [f'f{number}' for number in references]}
                     for index, (first, last, text, references) in enumerate(note['relations'], 1)]
        fact_ids = [item['id'] for item in facts]
        additions.append({
            'batch': batch,
            'source': {'csv': 'data/pilot_study.csv', **{field: original[field] for field in ('clip', 'tactic', 'interval')},
                       'original_rationale': original['判断依据'], 'video_path': video['path'], 'video_split': video['split'],
                       'video_duration_s': video['duration_s'], 'video_fps': video['fps'], 'time_origin': 'original_video_start',
                       'target_window_s': [start, end], 'view_window_s': [max(0, start - 2), min(video['duration_s'], end + 2)]},
            'annotation': {'status': 'machine_draft', 'author': 'GitHub Copilot', 'date': datetime.now().date().isoformat(),
                           'method': 'local_sampled_and_selected_native_frame_review', 'overview_fps': 0.5, 'window_fps': 4,
                           'native_reviewed_spans_s': sorted({tuple(span) for span in native}),
                           'viewed_artifacts': [str((directory / name).relative_to(ROOT)) for name in viewed],
                           'frame_manifest': str((directory / 'review_frames.json').relative_to(ROOT)),
                           'human_reviewer': None, 'human_reviewed_at': None,
                           'limitations': ['Sampled overview/window images are not continuous playback; only listed native spans were inspected frame-by-frame.']},
            'output': {'clip': note['clip'], 'target_window_s': [start, end], 'facts': facts, 'relations': relations,
                       'interpretation': {'text': note['text'], 'evidence_ids': fact_ids, 'relation_ids': [item['id'] for item in relations]},
                       'commentary': {'text': note['commentary'], 'evidence_ids': fact_ids},
                       'uncertainties': [{'span_s': [start, end], 'text': text} for text in note['uncertainties']]},
            'review': {'candidate_core_fact_ids': fact_ids,
                       'source_alignment': [{'source_claim': original['判断依据'], 'assessment': 'qualified',
                                             'evidence_ids': fact_ids, 'notes': note['notes']}],
                       'spatial_checks': {'basis': note['spatial'], 'evidence_ids': fact_ids},
                       'human_checks_required': ['Verify expanded facts, regions and label applicability before approving the AI reference.']},
        })
    ledger['records'] = [item for item in ledger['records'] if item['batch'] != batch] + additions
    if batch not in ledger['completed_batches']:
        ledger['completed_batches'].append(batch)
    validate(ledger)
    assert CSV.read_bytes() == (ROOT / inventory['baseline_path']).read_bytes()
    atomic_write_json(LEDGER, ledger)
    validate(load(LEDGER))
    print(f'PASS {batch}: {len(additions)} drafts saved; cumulative {len(ledger["records"])}; CSV unchanged')


def publish():
    ledger = load(LEDGER)
    validate(ledger, complete=True)
    baseline = (ROOT / ledger['inventory']['baseline_path']).read_bytes()
    fields, before = parse_csv(baseline)
    texts = {key(item['source']): item['output']['interpretation']['text'] for item in ledger['records']}
    after = [{**row, COLUMN: texts.get(key(row), row[COLUMN])} for row in before]
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(after)
    payload = buffer.getvalue().encode('utf-8-sig')
    assert parse_csv(payload) == (fields, after)
    for old, new in zip(before, after):
        assert all(old[field] == new[field] for field in fields[:-1])
        assert key(new) in texts or old == new
    assert len(after) == len(before) and sum(old != new for old, new in zip(before, after)) == len(texts)
    if 'publication' in ledger:
        assert CSV.read_bytes() == payload
        print('PASS already published:', len(texts))
        return
    assert CSV.read_bytes() == baseline, 'CSV changed; reconcile before publishing'
    stamp = datetime.now().astimezone()
    backup = ROOT / 'outputs/pilot_reference_backup' / (stamp.strftime('%Y%m%dT%H%M%S%f') + '-aa')
    backup.mkdir(parents=True, exist_ok=False)
    atomic_copy(CSV, backup / CSV.name)
    atomic_copy(LEDGER, backup / LEDGER.name)
    temporary = WORK / 'publish.csv'
    with temporary.open('xb') as handle:
        handle.write(payload)
    assert parse_csv(temporary.read_bytes()) == (fields, after) and CSV.read_bytes() == baseline
    atomic_copy(temporary, CSV)
    assert CSV.read_bytes() == payload
    ledger['publication'] = {'column': COLUMN, 'written_at': stamp.isoformat(timespec='seconds'),
                             'backup_path': str((backup / CSV.name).relative_to(ROOT)),
                             'original_sha256': sha(baseline), 'published_sha256': sha(payload),
                             'total_rows': len(after), 'updated_aa_rows': len(texts),
                             'unprocessed_aa_keys': [list(key(row)) for row in ledger['inventory']['source_rows'] if key(row) not in texts]}
    atomic_write_json(LEDGER, ledger)
    validate(load(LEDGER), complete=True)
    print('PASS publication:', len(texts), 'AA texts;', len(after), 'total rows; GSR and original columns preserved')
    print('Backup:', backup.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'prepare', 'native', 'record', 'check', 'publish'])
    parser.add_argument('values', nargs='*')
    args = parser.parse_args()
    if args.action == 'init':
        initialize()
    elif args.action == 'prepare':
        prepare(args.values[0])
    elif args.action == 'native':
        clip, first, last = args.values
        start, end = float(first), float(last)
        assert 0 <= start < end
        sheets(clip, f'native-{start:05.2f}-{end:05.2f}', start, end)
    elif args.action == 'record':
        record_batch(args.values[0])
    elif args.action == 'publish':
        publish()
    else:
        ledger = load(LEDGER)
        validate(ledger, complete=bool(args.values and args.values[0] == 'complete'))
        print('PASS validation:', len(ledger['records']), 'drafts')


if __name__ == '__main__':
    main()