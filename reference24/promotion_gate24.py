"""Apply frozen measurement-consistency rules without selecting on endpoint distance."""
import json
from pathlib import Path

def evaluate(diagnostics,rules):
    q=diagnostics;m=q['common_metrics'];base=m['v1'];candidate=m['candidate'];v=q['validation'];before=v['before'];after=v['after']
    failures=[]
    def check(ok,reason):
        if not ok:failures.append(reason)
    n=q['solver']['bundle_points'];rejected=q.get('rejected_training_tracks',0)
    check(n-rejected>=rules['minimum_training_points'],'insufficient_training_points')
    check(after['points']>=rules['minimum_validation_tracks'],'insufficient_reserved_tracks')
    check(after['observations']>=rules['minimum_validation_observations'],'insufficient_reserved_observations')
    check(rejected/max(1,n)<=rules['maximum_training_rejected_track_fraction'],'too_many_rejected_landmark_tracks')
    check(q['training_after']['behind_camera']<=rules['maximum_training_behind_camera'],'invalid_training_projection')
    check(after['median_px']<=before['median_px']*rules['maximum_validation_median_ratio'],'reserved_median_not_improved')
    check(after['p90_px']<=max(before['p90_px']*rules['maximum_validation_p90_ratio'],before['p90_px']+rules['maximum_validation_p90_slack_px']),'reserved_tail_regression')
    check(after['behind_camera']/max(1,after['observations'])<=max(rules['maximum_validation_invalid_fraction'],before['behind_camera']/max(1,before['observations'])+rules['maximum_validation_invalid_fraction_increase']),'invalid_reserved_predictions')
    ratio=candidate['path_m']/base['path_m']
    check(rules['minimum_path_ratio']<=ratio<=rules['maximum_path_ratio'],'large_metric_path_change')
    for metric,prefix in [('gyro_p95_deg','gyro_p95'),('acc_p95','acc_p95'),('jerk_p95','jerk_p95')]:
        floor=rules['maximum_'+prefix+'_floor'+('_deg' if prefix=='gyro_p95' else '')]
        check(candidate[metric] is not None and candidate[metric]<=max(base[metric]*rules['maximum_'+prefix+'_ratio'],floor),prefix+'_regression')
    check(candidate['max_speed']<=rules['maximum_speed_mps'],'speed_outlier')
    check(candidate['max_turn_rate']<=rules['maximum_turn_rate_deg_s'],'rotation_rate_outlier')
    return dict(passed=not failures,failures=failures,path_ratio=ratio,
        validation_median_ratio=after['median_px']/before['median_px'],validation_p90_ratio=after['p90_px']/before['p90_px'],
        approximate_return_consistency_warning=candidate['full_endpoint_m']>.2,
        note='Measurement consistency gate. Endpoint distance is reported but never used to promote a method. No ground truth.')

if __name__=='__main__':
    import sys
    print(json.dumps(evaluate(json.load(open(sys.argv[1])),json.load(open(sys.argv[2]))),indent=2))
