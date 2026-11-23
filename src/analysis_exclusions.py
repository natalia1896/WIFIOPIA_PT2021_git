"""Declarative, reproducible scientific exclusions for PT2021 analyses.

Reads analysis_exclusions.rules from the shared project YAML; does NOT change
Step2 selection or manual annotation files. Public function:
    build_exclusion_registry(index, annotations, config)
Both inputs may include the same stem. The Step2/Step4 index wins for metadata.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from common import parse_variant_value

MATCH_FIELDS = frozenset({'mouse_id', 'exp_day', 'state', 'protocol', 'stem'})
META_FIELDS = ('stem', 'mouse_id', 'exp_day', 'state', 'protocol')


def _str(value: Any) -> str:
    """Convert a nullable metadata value to a stripped string.

    Parameters
    ----------
    value : Any
        Scalar metadata or annotation value.

    Returns
    -------
    str
        Empty string for missing values; otherwise stripped text.
    """
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _values(value: Any, *, rule_id: str, field: str) -> list[str]:
    """Normalize the accepted values for one YAML exclusion-rule field.

    Parameters
    ----------
    value : Any
        A string or a list of string-like values.
    rule_id : str
        Exclusion-rule ID used in diagnostic exceptions.
    field : str
        Metadata field being matched.

    Returns
    -------
    list[str]
        Nonempty values to match within the field (OR semantics).

    Raises
    ------
    ValueError
        If the input is neither a string nor a list, or is empty.
    """
    if isinstance(value, str):
        result = [value.strip()]
    elif isinstance(value, list):
        result = [_str(v) for v in value]
    else:
        raise ValueError(f'{rule_id}: {field} must be a string or a list of strings')
    if not result or any(not s for s in result):
        raise ValueError(f'{rule_id}: {field} cannot be empty')
    return result


def _rules(config: dict) -> list[dict]:
    """Validate and normalize declarative scientific exclusion rules.

    Parameters
    ----------
    config : dict
        Project YAML with optional ``analysis_exclusions.rules``.

    Returns
    -------
    list[dict]
        Rules with unique IDs, reasons, and normalized match fields.

    Raises
    ------
    ValueError
        If rules lack required fields or use unsupported match keys.

    Notes
    -----
    Each rule must match ``mouse_id`` or ``stem`` (possibly alongside other
    fields) to prevent accidental exclusions of an entire cohort.
    """
    section = config.get('analysis_exclusions', {}) or {}
    raw_rules = section.get('rules', []) or []
    if not isinstance(raw_rules, list):
        raise ValueError('analysis_exclusions.rules must be a list')
    parsed = []
    used_ids = set()
    for n, rule in enumerate(raw_rules, 1):
        if not isinstance(rule, dict):
            raise ValueError(f'analysis_exclusions.rules[{n}] must be a mapping')
        rid, reason = _str(rule.get('id')), _str(rule.get('reason'))
        match = rule.get('match')
        if not rid or rid in used_ids:
            raise ValueError(f'Empty/duplicate analysis exclusion rule id: {rid!r}')
        if not reason:
            raise ValueError(f'{rid}: reason is mandatory')
        if not isinstance(match, dict) or not match:
            raise ValueError(f'{rid}: match must be a nonempty mapping')
        unknown = set(match) - MATCH_FIELDS
        if unknown:
            raise ValueError(f'{rid}: unsupported match keys: {sorted(unknown)}')
        if not ('mouse_id' in match or 'stem' in match):
            raise ValueError(f'{rid}: specify mouse_id or stem (prevent accidental whole-cohort exclusion)')
        fields = {k: _values(v, rule_id=rid, field=k) for k, v in match.items()}
        parsed.append(dict(id=rid, reason=reason, match=fields))
        used_ids.add(rid)
    return parsed


def _matches(row: pd.Series, fields: dict, config: dict) -> bool:
    # Parse variants exactly as the project shared common.py does; NEVER reselect a repeat.
    """Test whether a metadata row satisfies all fields of one rule.

    Parameters
    ----------
    row : pandas.Series
        Metadata containing ``stem``, ``mouse_id``, day, state and protocol.
    fields : dict
        Match fields with lists of accepted values.
    config : dict
        ``variant_keys.protocol`` controls protocol-base normalization.

    Returns
    -------
    bool
        True if all specified fields match; values within a field use OR.

    Notes
    -----
    Protocol parsing uses ``common.parse_variant_value``. This does not
    choose another recording repeat or modify the Step2 index.
    """
    vcfg = (config.get('variant_keys', {}) or {}).get('protocol', {}) or {}
    variant_protocol = bool(vcfg.get('enabled', True))
    separators = vcfg.get('separators', ['', '-', '_'])
    for field, accepted in fields.items():
        actual = _str(row.get(field))
        if field == 'protocol' and variant_protocol:
            actual = parse_variant_value(actual, separators=separators)[0]
            choices = [parse_variant_value(v, separators=separators)[0] for v in accepted]
        else:
            choices = accepted
        if actual not in choices:
            return False
    return True


def build_exclusion_registry(index: pd.DataFrame,
                             annotations: pd.DataFrame,
                             config: dict) -> dict[str, pd.DataFrame]:
    """Build a reproducible exclusion registry and rule-match audit.

    Parameters
    ----------
    index : pandas.DataFrame
        Full selected Step2 metadata table (including non-REST protocols),
        preferred over duplicate annotation metadata for the same stem.
    annotations : pandas.DataFrame
        Manual annotations, including stems absent from the index.
    config : dict
        Uses ``analysis_exclusions.enabled``, ``rules`` and
        ``fail_on_unmatched_rules`` (default True).

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``records``: per-stem metadata, exclusion flag, matching IDs/reasons;
        ``rule_report``: each rule and all matched stems.

    Raises
    ------
    ValueError
        For invalid rules, duplicate Step2 stems or unmatched enabled rules
        when ``fail_on_unmatched_rules`` is True.

    Notes
    -----
    Match fields are AND across fields and OR across values within one field.
    Rule IDs and reasons are saved in JSON columns. Neither source table
    nor any manual annotation CSV is modified.
    """
    records = []
    seen = set()
    for frame, source in ((index, 'index'), (annotations, 'annotation_only')):
        if frame is None or frame.empty:
            continue
        if 'stem' not in frame:
            raise KeyError(f'{source} is missing stem')
        for _, r in frame.iterrows():
            stem = _str(r['stem'])
            if not stem:
                raise ValueError(f'{source} contains a row with empty stem')
            if stem in seen:
                if source == 'index':
                    raise ValueError(f'Duplicate stem in index: {stem}')
                continue
            seen.add(stem)
            rec = {k: _str(r.get(k)) for k in META_FIELDS}
            rec['source'] = source
            records.append(rec)
    out = pd.DataFrame(records, columns=[*META_FIELDS, 'source'])
    rules = _rules(config)
    enabled = bool((config.get('analysis_exclusions', {}) or {}).get('enabled', True))
    counts = {r['id']: [] for r in rules}
    id_lists, reason_lists = [], []
    for _, r in out.iterrows():
        found = [rule for rule in rules if enabled and _matches(r, rule['match'], config)]
        id_lists.append([v['id'] for v in found])
        reason_lists.append([v['reason'] for v in found])
        for v in found:
            counts[v['id']].append(r['stem'])
    out['analysis_excluded'] = [bool(x) for x in id_lists]
    out['exclusion_rule_ids'] = [json.dumps(x, ensure_ascii=False) for x in id_lists]
    out['exclusion_reasons'] = [json.dumps(x, ensure_ascii=False) for x in reason_lists]
    out['exclusion_reason'] = ['; '.join(x) for x in reason_lists]
    report = pd.DataFrame([
        dict(rule_id=r['id'], reason=r['reason'], match=json.dumps(r['match'],ensure_ascii=False),
             n_matched=len(counts[r['id']]), matched_stems=json.dumps(counts[r['id']],ensure_ascii=False))
        for r in rules
    ], columns=['rule_id','reason','match','n_matched','matched_stems'])
    if enabled and bool((config.get('analysis_exclusions', {}) or {}).get('fail_on_unmatched_rules', True)):
        unmatched = report.loc[report['n_matched'].eq(0), 'rule_id'].tolist()
        if unmatched:
            raise ValueError('Exclusion rules matched no stems: '+repr(unmatched)+
                             '. Check Step2 selected stem metadata, exp_day and variant_keys.')
    return {'records': out, 'rule_report': report}
