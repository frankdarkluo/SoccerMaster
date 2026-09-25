"""Validate the diagnostic trial and export a non-overwriting review workbook."""
from __future__ import annotations

import argparse
import io
import os
import re
from collections import Counter
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from scripts.pilot_reference_aa import COLUMN, ROOT, key, load, parse_csv, sha, window


TRIAL = ROOT / 'benchmark/pilot_diagnostic_trial.json'
OUTPUT = ROOT / 'outputs/pilot_diagnostic_review'
VERDICTS = {'supported': '支持', 'contradicted': '矛盾', 'unknown': '未知', 'not_applicable': '不适用'}
KINDS = {'core': '核心判据', 'anchor': '场地锚点', 'relation': '空间关系',
         'diagnostic_probe': '事实探针(非模型输出)', 'rule_probe': '定义检查(不计幻觉)', 'context': '窗外线索'}
STAGES = {'prefix_only': '仅有前缀', 'pattern_observed': '结构可见',
          'completion_observed': '配合完成可见', 'unresolved': '待裁决'}


def prepare():
    trial = load(TRIAL)
    raw = (ROOT / trial['source_csv']).read_bytes()
    assert sha(raw) == trial['source_sha256'], 'Source CSV changed; review the trial before rebuilding.'
    _, rows = parse_csv(raw)
    references = {}
    for filename in ('pilot_reference_aa.json', 'pilot_reference_gsr.json'):
        for record in load(ROOT / 'benchmark' / filename)['records']:
            references[key(record['source'])] = record
    source_rows = {key(row): row for row in rows}
    protocol = (ROOT / trial['definition_source']).read_text(encoding='utf-8')
    definitions = [tuple(cell.strip().strip('`') for cell in line.strip('|').split('|'))
                   for line in protocol.splitlines() if re.match(r'^\| `[a-z-]+` \|', line)]
    assert len(definitions) == 12 and {item[0] for item in definitions} == {row['tactic'] for row in rows}
    predicates = {item[0]: set(re.findall(r'\b[A-Z]{2}[123]\b', item[2])) for item in definitions}
    assert len(trial['records']) == len({key(record) for record in trial['records']}) == 24
    assert Counter(record['tactic'] for record in trial['records']) == Counter({item[0]: 2 for item in definitions})
    assert trial['status'] == 'machine_draft'
    assert trial['human_reviewer'] is None and trial['human_reviewed_at'] is None
    assets = {}
    for record in trial['records']:
        original, reference = source_rows[key(record)], references[key(record)]
        assert original[COLUMN] == reference['output']['interpretation']['text']
        assert (ROOT / reference['source']['video_path']).is_file()
        first, last = window(record['interval'])
        assert record['stage'] in STAGES
        assert {item[0] for item in record['items'] if item[2] == 'core'} == predicates[record['tactic']]
        assert len({item[0] for item in record['items']}) == len(record['items'])
        manifest_path = reference['annotation'].get('frame_manifest')
        sheets = load(ROOT / manifest_path)['sheets'] if manifest_path else {}
        for item in record['items']:
            assert len(item) == len(trial['item_columns'])
            assert item[2] in KINDS and item[6] in VERDICTS
            assert (item[1] == 'Definition') == (item[2] == 'rule_probe')
            assert item[3] in {'target_window', 'context_only'} and item[4][0] < item[4][1]
            if item[3] == 'target_window':
                assert first <= item[4][0] < item[4][1] <= last
            assert item[8]
            for name in item[8]:
                assert Path(name).name == name
                relative = f"outputs/pilot_reference_review/{record['clip']}/{name}"
                assert relative in reference['annotation']['viewed_artifacts']
                checksum = sha((ROOT / relative).read_bytes())
                if name in sheets:
                    assert checksum == sheets[name]['sha256']
                    stamps = sheets[name]['timestamps_s']
                    timing = f'{stamps[0]:.2f}-{stamps[-1]:.2f} s; {len(stamps)} frames'
                else:
                    timing = '见 GSR 原账本的采样步长、view_window_s 和 native 区间'
                assets[relative] = (record['clip'], name, timing, checksum)
    return trial, rows, source_rows, references, definitions, assets


def table(workbook, title, headers, widths):
    sheet = workbook.create_sheet(title)
    sheet.append(headers)
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = 'C2'
    return sheet


def choices(sheet, column, values):
    validation = DataValidation(type='list', formula1='"' + ','.join(values) + '"', allow_blank=True)
    validation.errorTitle = '请选择列表中的值'
    validation.error = '修改具体内容请写入备注列。'
    validation.showErrorMessage = True
    sheet.add_data_validation(validation)
    validation.add(f'{column}2:{column}{sheet.max_row}')


def link(cell, relative):
    cell.hyperlink = os.path.relpath(ROOT / relative, OUTPUT)


def build(data):
    trial, rows, source_rows, references, definitions, assets = data
    workbook = Workbook()
    workbook.remove(workbook.active)
    intro = table(workbook, '审核说明', ['项目', '说明'], [24, 110])
    for name, value in [('状态', trial['status']), ('目的', trial['purpose']), ('输入指纹', trial['source_sha256']),
                        ('操作', trial['review_instructions']), ('实际方法', trial['method']),
                        ('时间口径', trial['time_precision'])]:
        intro.append([name, value])
    for text in trial['limitations']:
        intro.append(['边界', text])
    intro.append(['建议顺序', '先在逐项审核页确认核心判据和未知项，再在样例总览裁决标签/窗口；最后确认十二类定义。'])
    intro.append(['回写', '本文件没有自动回写功能。人工意见不自动覆盖源 CSV 或把机器草稿升级为 GT。'])
    review = table(workbook, '逐项审核',
                   ['clip', '判据', '待核验命题', 'AI判断', '人工意见', '修订判断', '修订或备注', '核对秒段',
                    '首张证据', '原视频', 'AI观察或限制', '维度', '类型', '范围', '原标签', '审核人', '审核日期'],
                   [23, 10, 48, 10, 12, 12, 38, 16, 14, 14, 55, 14, 24, 16, 32, 16, 18])
    overview = table(workbook, '样例总览',
                     ['clip', '原标签', '原窗口', '执行队', '试标目的', 'AI阶段', 'AI候选处理', '人工裁决',
                      '修订标签或窗口', '备注', '本轮补充观察', '现有AI润色', '人工原依据', '原视频', '审核人', '审核日期'],
                     [23, 32, 18, 18, 38, 18, 45, 18, 24, 38, 65, 65, 55, 14, 16, 18])
    notes = ['# 足球关系诊断试标审核稿', '', '状态：machine_draft，未覆盖原 CSV。', '',
             f"输入指纹：`{trial['source_sha256']}`。24 条开发样例，原标签每类两条，不代表每类两个确认正例。", '',
             trial['method'], '', '## 类别覆盖', '', '| 标签 | 原记录 | AI非空 | 试标 |', '| --- | ---: | ---: | ---: |']
    coverage = table(workbook, '覆盖情况', ['标签', '原记录', 'AI非空', '本轮试标', '子型边界'], [36, 14, 14, 14, 65])
    for label, *_ in definitions:
        selected = [row for row in rows if row['tactic'] == label]
        filled = sum(bool(row[COLUMN]) for row in selected)
        coverage.append([label, len(selected), filled, 2, '均为外侧套边，内切套上尚缺' if label == 'overlap-underlap' else '非完整正负例或难度分布'])
        notes.append(f'| {label} | {len(selected)} | {filled} | 2 |')
    notes += ['', '## 审核边界', '', *(f'- {text}' for text in trial['limitations']), '',
              'AI 支持、矛盾、未知均待人工；定义检查不计入事实幻觉。未知不按错误处理；同一事实正式计分前去重。', '']
    for record in trial['records']:
        original, reference = source_rows[key(record)], references[key(record)]
        video = reference['source']['video_path']
        overview.append([record['clip'], record['tactic'], record['interval'], record['reference_team'], record['selection_role'],
                         STAGES[record['stage']], record['decision'], None, None, None, record['summary'],
                         original[COLUMN], original['判断依据'], '打开视频', None, None])
        link(overview.cell(overview.max_row, 14), video)
        notes += [f"## {record['clip']} / {record['tactic']} / {record['interval']}", '', record['summary'], '',
                  f"候选处理：{record['decision']}", '', f"请确认：{record['review_question']}", '',
                  f'[原视频]({os.path.relpath(ROOT / video, OUTPUT)})', '',
                  '| 判据 | 范围/秒段 | 待核验命题 | AI判断 | 观察/限制 | 人工意见 |', '| --- | --- | --- | --- | --- | --- |']
        record_images = set()
        for item in record['items']:
            item_id, dimension, kind, scope, span, claim, verdict, observation, images = item
            interval = f'{span[0]:g}-{span[1]:g}'
            scope_name = '目标窗' if scope == 'target_window' else '仅上下文'
            relative = f"outputs/pilot_reference_review/{record['clip']}/{images[0]}"
            review.append([record['clip'], item_id, claim, VERDICTS[verdict], None, None, None, interval,
                           '打开证据图', '打开视频', observation, dimension, KINDS[kind], scope_name,
                           record['tactic'], None, None])
            link(review.cell(review.max_row, 9), relative)
            link(review.cell(review.max_row, 10), video)
            notes.append(f'| {item_id} ({KINDS[kind]}) | {scope_name} {interval} | {claim} | {VERDICTS[verdict]} | {observation} | |')
            record_images.update(images)
        notes += ['', '证据图：' + '；'.join(f'[{name}](../pilot_reference_review/{record["clip"]}/{name})' for name in sorted(record_images)), '',
                  '- [ ] 同意候选处理', '- [ ] 修改（填写标签、窗口或事实）', '- [ ] 证据不足，暂不确认', '']
    definition_sheet = table(workbook, '十二类定义', ['标签', '候选定义', '核心判据', '边界', '人工意见', '修改建议'], [34, 60, 75, 75, 14, 55])
    for definition in definitions:
        definition_sheet.append([*definition, None, None])
    evidence = table(workbook, '证据索引', ['clip', '图像', '采样范围(非动作边界)', 'SHA256', '查看'], [23, 38, 65, 68, 14])
    for relative, (clip, name, timing, checksum) in sorted(assets.items()):
        evidence.append([clip, name, timing, checksum, '打开图像'])
        link(evidence.cell(evidence.max_row, 5), relative)
    choices(review, 'E', trial['review_options'])
    choices(review, 'F', list(VERDICTS.values()))
    choices(overview, 'H', ['保留', '仅保留前缀', '修改标签', '修改窗口', '修改定义', '证据不足', '多项修改'])
    choices(definition_sheet, 'E', trial['review_options'])
    editable = {'逐项审核': {5, 6, 7, 16, 17}, '样例总览': {8, 9, 10, 15, 16}, '十二类定义': {5, 6}}
    for sheet in workbook:
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.zoomScale = 85
        for row in sheet:
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = 's'
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                cell.font = Font(name='Aptos', size=11, color='176A73' if cell.hyperlink else '222222')
                if cell.row == 1:
                    cell.fill = PatternFill('solid', fgColor='176A73')
                    cell.font = Font(name='Aptos', size=11, bold=True, color='FFFFFF')
                elif cell.column in editable.get(sheet.title, set()):
                    cell.fill = PatternFill('solid', fgColor='E9F4ED')
            sheet.row_dimensions[row[0].row].height = 30 if row[0].row == 1 else (150 if sheet.title in {'样例总览', '十二类定义'} else 78)
    buffer = io.BytesIO()
    workbook.save(buffer)
    reopened = load_workbook(io.BytesIO(buffer.getvalue()))
    assert reopened['逐项审核'].max_row == sum(len(record['items']) for record in trial['records']) + 1
    assert reopened['样例总览'].max_row == 25 and reopened['十二类定义'].max_row == 13
    assert len(reopened['逐项审核'].data_validations.dataValidation) == 2
    for sheet in workbook:
        assert list(reopened[sheet.title].values) == list(sheet.values)
    return buffer.getvalue(), '\n'.join(notes) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate inputs and in-memory export without writing files.')
    args = parser.parse_args()
    data = prepare()
    workbook, markdown = build(data)
    if not args.check:
        paths = [OUTPUT / 'pilot_diagnostic_review.xlsx', OUTPUT / 'pilot_diagnostic_review.md']
        if any(path.exists() for path in paths):
            raise FileExistsError('Review output already exists; refusing to overwrite human edits. Use --check.')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with paths[0].open('xb') as stream:
            stream.write(workbook)
        with paths[1].open('x', encoding='utf-8') as stream:
            stream.write(markdown)
        print('\n'.join(str(path.relative_to(ROOT)) for path in paths))
    assert sha((ROOT / data[0]['source_csv']).read_bytes()) == data[0]['source_sha256']
    print(f'PASS: 24 records / 12 tactics; {len(data[5])} evidence images; XLSX round-trip and source hash verified.')


if __name__ == '__main__':
    main()