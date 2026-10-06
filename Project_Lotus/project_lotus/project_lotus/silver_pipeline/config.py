"""Load reviewed YAML; reject misspelled/unsupported rules instead of skipping them."""
import hashlib
import json
from importlib.resources import files
from pathlib import Path
import yaml

DATASETS = ('account_changes_batch', 'device_lookup_batch', 'audit_trail_services_3')
RULE_OPTIONS = {
    'required': set(), 'allowed': {'values'}, 'by_mno': {'values'},
    'regex': {'pattern'}, 'future': {'grace_seconds'}, 'flag': set(),
    'positive': set(), 'nonnegative': set(), 'date_equal': {'other'},
    'required_when': {'when_column', 'when_values'}, 'long_numeric_id': set(),
}


def load(dataset, directory=None):
    """Load one dataset's YAML rule contract and reject anything malformed.

    A misspelled key or unsupported rule kind fails the run immediately instead
    of being silently ignored, since a rule that is silently dropped would let
    bad data reach Silver without anyone noticing. The returned config carries
    a content hash (`_hash`) so every run report can prove which contract ran.
    """
    if dataset not in DATASETS:
        raise ValueError('Unsupported dataset: ' + dataset)
    root = Path(directory) if directory else files('silver_pipeline').joinpath('configs')
    cfg = yaml.safe_load(root.joinpath(dataset + '.yaml').read_text(encoding='utf-8'))
    required = {'dataset','inputs','notes_fields','output','schema_version','source_name','timezone','null_tokens','rules'}
    allowed = required | {'test_partner_ids','test_provider_ids'}
    if set(cfg) - allowed or required - set(cfg):
        raise ValueError('Unexpected or missing configuration keys')
    if cfg['dataset'] != dataset or cfg['timezone'] != 'America/Toronto':
        raise ValueError('Dataset/timezone contract mismatch; review source timestamp semantics before changing')
    if type(cfg['schema_version']) is not int or cfg['schema_version'] < 1:
        raise ValueError('schema_version must be a positive integer')
    if len({x.lower() for x in cfg['output']}) != len(cfg['output']):
        raise ValueError('Output names must be unique without regard to case')
    if not set(cfg['output'].values()) <= {'string','bigint','int','timestamp','date'}:
        raise ValueError('Unsupported output type')
    seen = set()
    for rule in cfg['rules']:
        kind = rule['kind']
        base = {'id','kind','column','severity'}
        if kind not in RULE_OPTIONS or set(rule) != base | RULE_OPTIONS[kind]:
            raise ValueError('Malformed rule: ' + str(rule))
        if rule['id'] in seen or rule['severity'] not in {'error','warning'}:
            raise ValueError('Duplicate rule ID or unsupported severity')
        seen.add(rule['id'])
        if kind != 'flag' and rule['column'] not in cfg['output']:
            raise ValueError('Unknown output column: ' + rule['column'])
        for option in ('other','when_column'):
            if option in rule and rule[option] not in cfg['output']:
                raise ValueError('Unknown comparison column')
    cfg['_hash'] = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    return cfg


def load_api_types(directory=None):
    """Load the audit API-operation-code to display-name mapping used by the audit transform."""
    root = Path(directory) if directory else files('silver_pipeline').joinpath('configs')
    result = yaml.safe_load(root.joinpath('api_types.yaml').read_text(encoding='utf-8'))
    if not isinstance(result, dict) or not result or not all(isinstance(k,str) and isinstance(v,str) for k,v in result.items()):
        raise ValueError('API mapping must be a nonempty string-to-string map')
    return result
