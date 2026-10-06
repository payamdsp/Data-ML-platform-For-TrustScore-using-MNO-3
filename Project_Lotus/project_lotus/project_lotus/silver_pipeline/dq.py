"""Executable YAML rules, complete null metrics and per-MNO reporting."""
from datetime import date, timedelta
from pyspark.sql import functions as F, Window


def violation(rule, run_date):
    """Translate one YAML rule into the Spark boolean expression that marks a row as bad."""
    c = F.col(rule['column'])
    kind = rule['kind']
    if kind == 'required': return c.isNull()
    if kind == 'flag': return c == F.lit(True)
    if kind == 'allowed': return c.isNotNull() & ~c.isin(rule['values'])
    if kind == 'by_mno':
        bad = F.lit(False)
        for mno, values in rule['values'].items():
            bad = bad | ((F.col('mno') == mno) & c.isNotNull() & ~c.isin(values))
        return bad
    if kind == 'regex': return c.isNotNull() & ~c.cast('string').rlike(rule['pattern'])
    if kind == 'positive': return c <= 0
    if kind == 'nonnegative': return c < 0
    if kind == 'date_equal': return c.isNotNull() & F.col(rule['other']).isNotNull() & ~c.eqNullSafe(F.col(rule['other']))
    if kind == 'required_when': return F.col(rule['when_column']).isin(rule['when_values']) & c.isNull()
    if kind == 'long_numeric_id': return c.rlike('^[0-9]{11,}$')
    if kind == 'future':
        # Batch policy allows the entire Toronto run_date, not just midnight.
        # At the following midnight the row is future, including exact equality.
        tomorrow = date.fromisoformat(run_date) + timedelta(days=1)
        boundary = F.to_timestamp(F.lit(str(tomorrow))) + F.expr(f"INTERVAL {int(rule['grace_seconds'])} SECONDS")
        return c >= boundary
    raise ValueError('Unknown rule: ' + kind)


def annotate(df, cfg, run_date, check_unique=False):
    """Run every applicable YAML rule and attach the resulting issues to each row.

    check_unique is only turned on after exact deduplication: a repeated
    record_id at that point means the same ID arrived with different payloads,
    which is itself a blocking condition rather than a harmless duplicate.
    """
    if not check_unique:
        missing = [r['column'] for r in cfg['rules'] if r['kind']=='flag' and r['column'] not in df.columns]
        if missing:
            raise ValueError('Missing transformation DQ flags: '+str(missing))
    rules = [r for r in cfg['rules'] if r['kind'] != 'flag' or r['column'] in df.columns]
    if check_unique:
        # Only after full-row dedup: repeated IDs now mean different payloads.
        df = df.withColumn('_duplicate_id', (F.count('*').over(Window.partitionBy('record_id')) > 1) & F.col('record_id').isNotNull())
        rules = rules + [dict(id='unique_record_id',kind='flag',column='_duplicate_id',severity='error')]
    expressions = [F.when(F.coalesce(violation(r,run_date),F.lit(False)), F.struct(
        F.lit(r['id']).alias('rule_id'), F.lit(r['severity']).alias('severity'))) for r in rules]
    result = df.withColumn('_issues', F.filter(F.array(*expressions), lambda x: x.isNotNull()))
    return result.withColumn('_has_errors', F.exists('_issues',lambda x: x['severity'] == 'error'))


def metrics(df, cfg):
    """Only bounded aggregate rows reach the driver; source rows never collect."""
    rule_ids = [r['id'] for r in cfg['rules']] + ['unique_record_id']
    aggs = [F.count('*').alias('rows'), F.sum(F.col('_has_errors').cast('long')).alias('error_rows'),
            F.sum((F.size('_issues')>0).cast('long')).alias('issue_rows'),
            F.countDistinct('record_id').alias('distinct_record_ids')]
    aggs += [F.sum(F.col(c).isNull().cast('long')).alias('null__'+c) for c in cfg['output']]
    aggs += [F.sum(F.exists('_issues',lambda x: x['rule_id'] == rid).cast('long')).alias('rule__'+rid) for rid in rule_ids]
    def unpack(row):
        """Reshape one flat aggregate Row (prefixed null__/rule__ columns) into the nested summary shape."""
        data = {k: (0 if v is None else v) for k,v in row.asDict().items()}
        n=data['rows']
        return {**{k:v for k,v in data.items() if not k.startswith(('null__','rule__','_group'))},
                'null_counts': {k[6:]:v for k,v in data.items() if k.startswith('null__')},
                'rule_failures': {k[6:]:v for k,v in data.items() if k.startswith('rule__')},
                'required_completeness': {r['column']: ((n-data['null__'+r['column']])/n if n else None) for r in cfg['rules'] if r['kind']=='required' and r['severity']=='error'}}
    overall=unpack(df.agg(*aggs).first())
    # Fold unknown carrier values into one group to bound metrics cardinality.
    domain=next(r['values'] for r in cfg['rules'] if r['kind']=='allowed' and r['column']=='mno')
    grouped=df.withColumn('_group',F.when(F.col('mno').isNull(),'__null__').when(F.col('mno').isin(domain),F.col('mno')).otherwise('__unknown__'))
    per_mno={r['_group']:unpack(r) for r in grouped.groupBy('_group').agg(*aggs).collect()}
    return {'overall':overall,'per_mno':per_mno,'passed':overall['rows']>0 and overall['error_rows']==0}


def issue_rows(df, cfg):
    """Explode each row's issue list into one row per violated rule.

    One row per violated rule is easier to query than free-text messages.
    source_row_id preserves the link to the exact duplicate source record when
    an operator compares this rule-level artifact to the quarantine table.
    """
    base = df.select(*cfg['output'], *[c for c in ['_raw_payload','_source_file','_source_row_id'] if c in df.columns], F.explode('_issues').alias('_issue'))
    return base.select('*', F.col('_issue.rule_id').alias('rule_id'), F.col('_issue.severity').alias('severity')).drop('_issue')
