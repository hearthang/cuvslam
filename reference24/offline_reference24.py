"""Generate and audit 47 offline references from the existing nv_slam caches.

Runs the frozen stereo landmark bundle method and applies the same observational
checks to every bag. Unsupported candidates remain available with their reasons;
the exported reference falls back to the separately verified coverage correction.
All original caches and reference_v1 are retained. No endpoint equality is used.
"""
import argparse,concurrent.futures,datetime,fcntl,hashlib,json,os,re,resource,shutil,signal,sqlite3,subprocess,sys,traceback
from pathlib import Path

HELPERS=['export_colmap24.py','run_colmap24.py','triangulate_from_trajectory24.py','export_ba_tracks24.py',
 'refine_stereo_tracks24.py','reference_bundle_robust24.py','reference_bundle_gyrocal24.py','landmark_validity24.py','visual_inertial_ba24.py','return_region24.py',
 'evaluate_reference24.py','imu_constraints24.py','translation_candidate24.py','reference_candidate24.py',
 'diagnose_bundle24.py','promotion_gate24.py','quality_flags24.py','generate_safe24.py']
ROOT_HELPERS=['backend.py','reference.py','evaluate_reference.py','translation_refine.py']

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def save(path,value):
    path=Path(path);tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def run_bounded(command,stdout,timeout,cwd=None):
    """A private session keeps a timed-out multiprocessing child bounded too."""
    process=subprocess.Popen(list(map(str,command)),cwd=cwd,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
    def cleanup():
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=5)
    try:returncode=process.wait(timeout=timeout)
    except BaseException:
        cleanup();raise
    if returncode:
        cleanup();raise subprocess.CalledProcessError(returncode,command)

def install_termination_cleanup():
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)

def source_snapshot(repo,seed,out,gyro_calibration=True,return_radius=None,return_sigma=.1,runtime_signature=None,max_iterations=40,bundle_timeout_seconds=7200):
    sources={name:seed/name for name in HELPERS}
    for name in ROOT_HELPERS:sources[name]=(seed/name) if (seed/name).exists() else repo/name
    sources['offline_reference24.py']=Path(__file__).resolve()
    normalized={}
    for name,path in sources.items():
        text=path.read_text()
        # Prefer the frozen copies to future edits in the repository root.
        normalized[name]=re.sub(r'sys\.path\.insert\(0,\s*str\(Path\.cwd\(\)\)\)', 'sys.path.append(str(Path.cwd()))',text)
    rules=json.load(open(seed/'promotion_rules24.json'))
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in normalized.items()}
    mode=dict(apply_configured_gyro_scale=gyro_calibration,approximate_return_radius_m=return_radius,return_sigma_m=return_sigma if return_radius is not None else None,max_iterations=max_iterations,bundle_timeout_seconds=bundle_timeout_seconds)
    identity=hashlib.sha256(json.dumps(dict(sources=hashes,rules=rules,max_keyframes=120,mode=mode,runtime=runtime_signature),sort_keys=True).encode()).hexdigest()
    if (out/'pipeline.json').exists():
        old=json.load(open(out/'pipeline.json'))
        if old['identity']!=identity:raise ValueError('Output belongs to different code/rules; select a new output folder')
        for name,h in hashes.items():
            if sha(out/'source'/name)!=h:raise ValueError('Frozen source changed: '+name)
    else:
        (out/'source').mkdir(parents=True)
        for name,text in normalized.items():(out/'source'/name).write_text(text)
        save(out/'source/promotion_rules24.json',rules)
        save(out/'source/source_manifest.json',hashes)
        save(out/'pipeline.json',dict(identity=identity,started_utc=stamp(),repo=str(repo),source_sha256=hashes,rules=rules,mode=mode,runtime_signature=runtime_signature,
            method='Fixed-calibration whole-track stereo BA, subpixel KLT, full-frame motion/gyro/gravity/verified loops, invalid-track culling, anchored conditional smoothing',
            prior='No endpoint equality. Return-region prior disabled.' if return_radius is None else f'Explicit approximate return region{ return_radius }m, soft penalty scale{ return_sigma }m, approximately interpreted for imu_link. Endpoint gain is not independent accuracy evidence.',
            input_layout='Existing results/fe5 and results/reference_v1 caches are required; run from the repository root.'))
    return identity

def copy_development_cache(old,dest,calib_hash,inputs):
    if not old or not (old/'vi_bundle_klt.npz').exists():return False
    meta=json.load(open(old/'input.json'))
    if meta.get('calibration_sha256')!=calib_hash or 'half-pixel' not in meta.get('pixel_convention',''):return False
    audit_path=old.parent.parent/'audit.json'
    if audit_path.exists():
        audit=next((row for row in json.load(open(audit_path))['bags'] if row['tag']==meta['tag']),None)
        if audit is None:return False
        by_name={Path(path).name:value for path,value in audit['input_sha256'].items()}
    else:
        summary_path=old/'geometry/summary.json'
        if not summary_path.exists():return False
        initial_hash=json.load(open(summary_path)).get('initial_trajectory_sha256')
        by_name={Path(path).name:value for path,value in meta.get('input_sha256',{}).items()}
        if initial_hash:by_name[meta['tag']+'.txt']=initial_hash
        else:
            quality_path=old.parent.parent/'quality'/f"{meta['tag']}.json"
            if not quality_path.exists():return False
            by_name={Path(path).name:value for path,value in json.load(open(quality_path))['input_sha256'].items()}
    for path,value in inputs.items():
        if Path(path).name in (meta['tag']+'_fe.npz',meta['tag']+'_kf.pkl',meta['tag']+'.txt'):
            if by_name.get(Path(path).name)!=value:return False
    dest.mkdir(parents=True)
    for name in ['images','masks','geometry']:
        shutil.copytree(old/name,dest/name,copy_function=os.link)
    for name in ['input.json','pairs.txt','vi_bundle_input.npz','vi_bundle_klt.npz','klt_refinement.json']:
        shutil.copy2(old/name,dest/name)
    with sqlite3.connect(f'file:{old / "database.db"}?mode=ro',uri=True) as source:
        with sqlite3.connect(dest/'database.db') as target:source.backup(target)
    save(dest/'reuse.json',dict(origin=str(old),origin_input_sha256=sha(old/'input.json'),note='Immutable development geometry reused; full trajectory optimization and diagnostics regenerated with frozen production source.'))
    return True

def preflight_numerical_reuse(production,review,out,max_iterations):
    """Validate the producer before starting any workers; paths are not identity."""
    old=json.load(open(production/'pipeline.json'));new=json.load(open(out/'pipeline.json'))
    state=json.load(open(production/'run_status.json'))
    if state['status'] not in ('finished','finished_with_candidate_failures') or len(state['completed'])!=47 or state['failed'] or state.get('candidate_failures'):
        raise ValueError('Numerical reuse requires a complete47 producer without runtime errors')
    if old['rules']!=new['rules'] or old['runtime_signature']!=new['runtime_signature']:
        raise ValueError('Numerical reuse requires identical rules and numerical runtimes')
    for key in ('apply_configured_gyro_scale','approximate_return_radius_m','return_sigma_m'):
        if old['mode'].get(key)!=new['mode'].get(key):raise ValueError('Reuse mode mismatch: '+key)
    for name,digest in new['source_sha256'].items():
        if name=='offline_reference24.py':continue
        if old['source_sha256'].get(name)!=digest or sha(production/'source'/name)!=digest:
            raise ValueError('Numerical helper changed: '+name)
    if review:
        reviewed=json.load(open(review/'status.json'))
        if max_iterations!=100 or reviewed['max_iterations']!=100:raise ValueError('This reviewed cache contains the fixed100 budget only')
        if reviewed['status'] not in ('finished','needs_review'):raise ValueError('Iteration review must be terminal before numerical reuse')
        if reviewed.get('source_sha256')!=old['source_sha256']:raise ValueError('Iteration review producer source changed')

def reuse_numerical_cache(production,review,out,tag,inputs,d,candidate,max_iterations):
    """Reuse identical numerical work, retaining the actual producer and budget.

    Uncapped40 runs stopped before40 in every GNC/refit stage and are equivalent
    to raising an unused upper bound. Capped40 runs require the actual100 run;
    absence or failure makes the caller compute a fresh candidate instead.
    This never selects a budget from endpoint or reserved-observation values.
    """
    if production is None:return None
    previous=json.load(open(production/'quality'/f'{tag}.json'))
    if previous['input_sha256']!=inputs:raise ValueError('Reuse input changed: '+tag)
    before_path=Path(previous['candidate_diagnostics']);before=json.load(open(before_path))
    budget=before['solver']['config']['max_iterations'];chosen=before;chosen_path=before_path
    stage_list=lambda solver:solver['stages']+solver.get('landmark_cull_stages',[])
    capped=any(stage['iterations']>=budget for stage in stage_list(before['solver']))
    producer_kind='uncapped_lower_budget' if budget<max_iterations else 'equal_budget'
    run=None
    if budget>max_iterations:return None
    if budget<max_iterations and capped:
        if review is None:return None
        reviewed=json.load(open(review/'status.json'));run=reviewed['completed'].get(tag)
        if run is None:return None
        if json.load(open(review/tag/'run.json'))!=run:raise ValueError('Review record changed: '+tag)
        if run['max_iterations']!=max_iterations:raise ValueError('Review budget mismatch: '+tag)
        for path,digest in run['input_sha256'].items():
            if sha(path)!=digest:raise ValueError('Review input changed: '+path)
        chosen_path=Path(run['diagnostics']);chosen=json.load(open(chosen_path));budget=max_iterations;producer_kind='reviewed_equal_budget'
        if run.get('diagnostics_sha256') and sha(chosen_path)!=run['diagnostics_sha256']:raise ValueError('Review diagnostics changed: '+tag)
        incidental={'suffix','bundle_output','max_iterations'}
        path_keys={'fe_dir','bundle_input','initial_trajectory','calib'}
        def objective(config):return {key:str(Path(value).resolve()) if key in path_keys else value for key,value in config.items() if key not in incidental}
        if objective(before['solver']['config'])!=objective(chosen['solver']['config']):raise ValueError('Review objective changed: '+tag)
    if chosen['tag']!=tag or chosen['solver']['config']['max_iterations']!=budget:raise ValueError('Numerical cache identity mismatch: '+tag)
    bundle_hash=sha(d/'vi_bundle_klt.npz')
    if chosen['solver']['bundle_input_sha256']!=bundle_hash:raise ValueError('Numerical cache observations changed: '+tag)
    source_hashes=json.load(open(out/'source/source_manifest.json'))
    for path,digest in chosen['solver']['source_sha256'].items():
        if source_hashes.get(Path(path).name)!=digest:raise ValueError('Numerical producer helper mismatch: '+path)
    original_candidate=Path(chosen['solver']['config']['fe_dir'])/f"{tag}_ref{chosen['solver']['config']['suffix']}.txt"
    original_stats=original_candidate.with_suffix('.json')
    if run and run.get('candidate_sha256') and sha(original_candidate)!=run['candidate_sha256']:raise ValueError('Review candidate changed: '+tag)
    if json.load(open(original_stats))!=chosen['solver']:raise ValueError('Numerical cache solver record changed: '+tag)
    destination=d/'full_bundle_klt_robust'
    if destination.exists():raise ValueError('Do not mix an existing numerical stage with reused work: '+tag)
    # Copies retain the original solver config/source paths as provenance; they
    # are not rewritten to pretend this invocation executed another budget.
    shutil.copytree(chosen_path.parent,destination)
    shutil.copy2(original_candidate,candidate);shutil.copy2(original_stats,candidate.with_suffix('.json'))
    record=dict(kind=producer_kind,producer_identity=json.load(open(production/'pipeline.json'))['identity'],
        producer_diagnostics=str(chosen_path),producer_diagnostics_sha256=sha(chosen_path),producer_config=chosen['solver']['config'],
        actual_max_iterations=budget,requested_max_iterations=max_iterations,
        original_candidate=str(original_candidate),original_candidate_sha256=sha(original_candidate),original_solver_sha256=sha(original_stats),
        bundle_input_sha256=bundle_hash,original_quality_sha256=sha(production/'quality'/f'{tag}.json'),
        copied_stage_sha256={name:sha(chosen_path.parent/name) for name in ('optimized.npz','validation.json')},
        source_sha256=chosen['solver']['source_sha256'],
        remaining_solver_cap=any(stage['iterations']>=budget for stage in stage_list(chosen['solver'])),
        note='Identical observations/objective/helper bytes. Actual producer budget retained; raising an unused upper bound is not a new100 run. Endpoint and heldout values did not choose the budget.')
    if run:record.update(review_run_sha256=sha(review/tag/'run.json'),producer_command=run['command'])
    save(d/'numerical_reuse.json',record);return record

def verified_cached_reuse(d,candidate):
    record=json.load(open(d/'numerical_reuse.json'));bundle_out=d/'full_bundle_klt_robust'
    if sha(candidate)!=record['original_candidate_sha256'] or sha(candidate.with_suffix('.json'))!=record['original_solver_sha256'] or sha(d/'vi_bundle_klt.npz')!=record['bundle_input_sha256'] or any(sha(bundle_out/name)!=value for name,value in record['copied_stage_sha256'].items()):
        raise ValueError('Previously reused numerical artifact changed: '+d.name)
    return record

def bag(job):
    repo,out,py,cp,cache,tag,identity,gyro_calibration,return_radius,return_sigma,max_iterations,numerical_production,iteration_review,bundle_timeout_seconds=job
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    repo=Path(repo);out=Path(out);cache=Path(cache) if cache else None;src=out/'source';d=out/'geometry'/tag
    selected=out/'trajectories'/f'{tag}.txt';quality=out/'quality'/f'{tag}.json';fe=repo/'results/fe5'
    cfg=json.load(open(repo/'results/reference_v1/quality'/f'{tag}.json'))['config']
    inputs={str(p):sha(p) for p in [fe/f'{tag}_fe.npz',fe/f'{tag}_kf.pkl',fe/f"{tag}_be{cfg['be_suffix']}.txt",
        fe/f"{tag}_loops{cfg['be_suffix']}.npz",fe/f"{tag}_extra{cfg['extra_suffix']}.npz",
        repo/'results/reference_v1/trajectories'/f'{tag}.txt',repo/'results/reference_v1/quality'/f'{tag}.json',Path(cfg['calib'])]}
    if selected.exists() and quality.exists():
        previous=json.load(open(quality))
        if previous['input_sha256']!=inputs or previous['trajectory_sha256']!=sha(selected):raise ValueError('Cached input or trajectory changed; select a new output folder')
        return previous
    safe_suffix='_reference24safe_'+identity[:10];candidate_suffix='_reference24bundle_'+identity[:10]
    safe=fe/f'{tag}_ref{safe_suffix}.txt';candidate=fe/f'{tag}_ref{candidate_suffix}.txt'
    state=dict(tag=tag,status='running',started_utc=stamp(),input_sha256=inputs);status=out/'status'/f'{tag}.json'
    if status.exists() and json.load(open(status)).get('input_sha256')!=inputs:raise ValueError('Partial stage inputs changed; select a new output folder')
    def command(cmd,stage,timeout=2100):
        state.update(stage=stage,command=list(map(str,cmd)),updated_utc=stamp());save(status,state)
        with (out/'logs'/f'{tag}.log').open('a') as log:run_bounded(cmd,log,timeout,cwd=repo)
    if not safe.exists() or not safe.with_suffix('.json').exists():command([py,src/'generate_safe24.py',tag,'--suffix',safe_suffix],'coverage_baseline')
    visual_error=None;gate=None;numerical_reuse=None
    try:
        if not d.exists() and cfg['cam1_fscale']==-.00038:copy_development_cache(cache/tag if cache else None,d,inputs[str(Path(cfg['calib']))],inputs)
        if not (d/'input.json').exists():command([py,src/'export_colmap24.py',tag,'--output-root',out/'geometry','--calib',cfg['calib'],'--cam1-fscale',cfg['cam1_fscale']],'export')
        if not (d/'database.db').exists():command([cp,src/'run_colmap24.py',d,'--match-only'],'matching',2400)
        if not (d/'geometry/heldout_tracks.json').exists():command([cp,src/'triangulate_from_trajectory24.py',d],'fixed_pose_triangulation')
        if not (d/'vi_bundle_input.npz').exists():command([cp,src/'export_ba_tracks24.py',d,'geometry'],'reserve_tracks')
        if not (d/'vi_bundle_klt.npz').exists():command([py,src/'refine_stereo_tracks24.py',d],'subpixel_stereo')
        bundle_out=d/'full_bundle_klt_robust'
        reuse_path=d/'numerical_reuse.json'
        if reuse_path.exists() and (bundle_out/'validation.json').exists() and candidate.exists() and candidate.with_suffix('.json').exists():
            numerical_reuse=verified_cached_reuse(d,candidate)
        if not (bundle_out/'validation.json').exists() and numerical_production:
            numerical_reuse=reuse_numerical_cache(Path(numerical_production),Path(iteration_review) if iteration_review else None,out,tag,inputs,d,candidate,max_iterations)
        if not (bundle_out/'validation.json').exists() or not candidate.exists() or not candidate.with_suffix('.json').exists():
            if reuse_path.exists():reuse_path.unlink()
            numerical_reuse=None
            if cfg['imu']:raise ValueError('This frozen method uses coverage-checked gyro; full inertial-state mode is not validated')
            prior_args=[]
            for key,value in cfg.items():
                if key in ('fe_dir','suffix','jobs','only','max_iterations'):continue
                option='--'+key.replace('_','-')
                if isinstance(value,bool):
                    if value:prior_args.append(option)
                else:prior_args.extend([option,*map(str,value if isinstance(value,list) else [value])])
            bundle_script='reference_bundle_gyrocal24.py' if gyro_calibration else 'reference_bundle_robust24.py'
            if return_radius is not None:prior_args.extend(['--return-radius',str(return_radius),'--return-sigma',str(return_sigma)])
            command([py,src/bundle_script,fe,'--only',tag,'--jobs','1',*prior_args,
                '--suffix',candidate_suffix,'--max-iterations',str(max_iterations),'--bundle-input',d/'vi_bundle_klt.npz',
                '--initial-trajectory',repo/'results/reference_v1/trajectories'/f'{tag}.txt',
                '--bundle-output',bundle_out],'joint_bundle',bundle_timeout_seconds)
        # The diagnostics helper expects the development root; production passes
        # an explicit data root to keep all caches and prior experiments separate.
        command([py,src/'diagnose_bundle24.py',tag,'full_bundle_klt_robust',candidate_suffix,'--root',out/'geometry'],'diagnostics',180)
        sys.path.insert(0,str(src));from promotion_gate24 import evaluate
        diagnostics=json.load(open(bundle_out/'diagnostics.json'));gate=evaluate(diagnostics,json.load(open(src/'promotion_rules24.json')))
    except Exception as error:
        visual_error=repr(error);state['visual_exception']=visual_error;save(status,state)
        with (out/'logs'/f'{tag}.log').open('a') as log:traceback.print_exc(file=log)
    chosen=candidate if gate and gate['passed'] else safe
    shutil.copy2(chosen,selected)
    if candidate.exists():shutil.copy2(candidate,out/'candidates'/f'{tag}.txt')
    method='stereo_joint_bundle' if return_radius is None else 'stereo_joint_bundle_with_return_region'
    result=dict(tag=tag,selected_method=method if chosen==candidate else 'imu_coverage_correction',
        requested_configured_gyro_scale=gyro_calibration,applied_configured_gyro_scale=gyro_calibration if chosen==candidate else False,
        candidate_max_iterations=max_iterations,
        candidate_requested_wall_timeout_seconds=bundle_timeout_seconds,
        candidate_actual_max_iterations=numerical_reuse['actual_max_iterations'] if numerical_reuse else max_iterations,
        numerical_reuse=numerical_reuse,
        requested_return_radius_m=return_radius,approximate_return_radius_m=return_radius if chosen==candidate else None,
        return_sigma_m=return_sigma if return_radius is not None and chosen==candidate else None,
        gate=gate,visual_error=visual_error,input_sha256=inputs,trajectory_sha256=sha(selected),
        sensor_quality=str(out/'sensor_quality'/f'{tag}_quality.json'),candidate_diagnostics=str(d/'full_bundle_klt_robust/diagnostics.json'),
        note='No independent ground truth. Endpoint separation and smoothness do not establish absolute accuracy. Handling can impose an uncertain position assumption.',finished_utc=stamp())
    save(quality,result);state.update(status='finished',selected_method=result['selected_method'],finished_utc=stamp());save(status,state)
    return result

def main():
    install_termination_cleanup()
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path.cwd());ap.add_argument('--sources',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--output',type=Path,default=Path('results/reference_v2'));ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--colmap-python',type=Path,default=Path('results/reference_24h_20261010/colmap_env/bin/python'))
    ap.add_argument('--reuse-development',type=Path,default=None)
    ap.add_argument('--reuse-numerical-production',type=Path,default=None,help='Optional complete producer with identical helpers/rules/inputs/runtime; actual producer budget remains explicit')
    ap.add_argument('--reuse-iteration-review',type=Path,default=None,help='Optional terminal100 review for producer cases that hit the40 cap; budget selection uses training-stage caps only')
    ap.add_argument('--only',default='',help='Optional bag substring for a pipeline pilot; default runs all47')
    ap.add_argument('--gyro-calibration',action=argparse.BooleanOptionalAction,default=True,help='Respect the existing gyro_scale_z configuration in gyro-only factors')
    ap.add_argument('--return-radius',type=float,default=None,help='Optional explicitly supplied approximate physical return region; default disabled')
    ap.add_argument('--return-sigma',type=float,default=.1,help='Soft penalty scale outside the optional return region, meters')
    ap.add_argument('--max-iterations',type=int,default=40,help='Joint solver iteration cap per GNC/refit stage; changing compute budget does not change constraints or selection rules')
    ap.add_argument('--bundle-timeout-seconds',type=int,default=7200,help='Wall-time bound per joint solve, including full-frame graph construction; long clips exceeded the original35-minute bound')
    a=ap.parse_args();repo=a.repo.resolve();os.chdir(repo);out=a.output.resolve();seed=a.sources.resolve()
    if not 1<=a.jobs<=2:raise ValueError('Use one or two jobs for the16GB runtime')
    if not 1<=a.max_iterations<=200:raise ValueError('Use a joint solver iteration budget between1 and200')
    if not 60<=a.bundle_timeout_seconds<=14400:raise ValueError('Joint solve wall-time bound must be between60 and14400 seconds')
    if a.return_radius is not None and (a.return_radius<=0 or a.return_sigma<=0):raise ValueError('Return region and its soft penalty scale must be positive; no zero endpoint target')
    out.mkdir(parents=True,exist_ok=True);lock=open(out/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Do not resolve the interpreter symlink: venv discovery depends on invoking
    # the executable at its venv path rather than its underlying base Python.
    cp=str(a.colmap_python.absolute());py=sys.executable
    import importlib.metadata,numpy,scipy,cv2,gtsam
    if not hasattr(gtsam,'GenericProjectionFactorCal3Fisheye'):raise RuntimeError('Missing fisheye GTSAM projection factor')
    probe="import json,sys,pycolmap,numpy;print(json.dumps(dict(python=sys.version.split()[0],prefix=sys.prefix,pycolmap=pycolmap.__version__,numpy=numpy.__version__,cuda=bool(pycolmap.has_cuda))))"
    checked=subprocess.run([cp,'-c',probe],check=True,capture_output=True,text=True,timeout=30)
    colmap_runtime=json.loads(checked.stdout)
    runtime=dict(main=dict(python=sys.version.split()[0],numpy=numpy.__version__,scipy=scipy.__version__,opencv=cv2.__version__,gtsam=importlib.metadata.version('gtsam')),
        colmap={k:v for k,v in colmap_runtime.items() if k!='prefix'})
    identity=source_snapshot(repo,seed,out,a.gyro_calibration,a.return_radius,a.return_sigma,runtime,max_iterations=a.max_iterations,bundle_timeout_seconds=a.bundle_timeout_seconds)
    if a.reuse_iteration_review and not a.reuse_numerical_production:raise ValueError('Iteration reuse requires its complete numerical producer')
    numerical_production=str(a.reuse_numerical_production.resolve()) if a.reuse_numerical_production else None
    iteration_review=str(a.reuse_iteration_review.resolve()) if a.reuse_iteration_review else None
    if numerical_production:
        preflight_numerical_reuse(Path(numerical_production),Path(iteration_review) if iteration_review else None,out,a.max_iterations)
        if a.reuse_development is None:a.reuse_development=Path(numerical_production)/'geometry'
    save(out/'runtime.json',dict(signature=runtime,main_executable=py,colmap_executable=cp,colmap_prefix=colmap_runtime['prefix'],checked_utc=stamp()))
    for name in ['trajectories','candidates','quality','geometry','logs','status','sensor_quality']:(out/name).mkdir(exist_ok=True)
    tags=sorted(p.stem for p in (repo/'results/reference_v1/trajectories').glob('*.txt'))
    if len(tags)!=47:raise ValueError('Expected the47 baseline bags')
    tags=[tag for tag in tags if a.only in tag]
    if not tags:raise ValueError('No matching bags')
    state=dict(pid=os.getpid(),status='running',identity=identity,completed={},failed={},started_utc=stamp())
    def persist():state['updated_utc']=stamp();save(out/'run_status.json',state)
    persist();cache=str(a.reuse_development.resolve()) if a.reuse_development else None
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        jobs={pool.submit(bag,(str(repo),str(out),py,cp,cache,tag,identity,a.gyro_calibration,a.return_radius,a.return_sigma,a.max_iterations,numerical_production,iteration_review,a.bundle_timeout_seconds)):tag for tag in tags}
        for future in concurrent.futures.as_completed(jobs):
            tag=jobs[future]
            try:state['completed'][tag]=future.result()
            except Exception as error:state['failed'][tag]=repr(error);traceback.print_exc()
            persist()
    if state['failed']:state['status']='incomplete';persist();return 1
    state['candidate_failures']={tag:row['visual_error'] for tag,row in state['completed'].items() if row.get('visual_error')}
    with (out/'logs/evaluation.log').open('w') as log:
        subprocess.run([py,str(out/'source/quality_flags24.py'),'--output',str(out/'sensor_quality')],check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run([py,str(out/'source/evaluate_reference24.py'),'--source','v1=results/reference_v1/trajectories/{tag}.txt',
            '--source','coverage=_reference24safe_'+identity[:10],'--source','selected='+str(out/'trajectories/{tag}.txt'),
            '--only',a.only,'--jobs','2','--output',str(out/'evaluation.json')],check=True,stdout=log,stderr=subprocess.STDOUT)
    state['status']=('finished_with_candidate_failures' if state['candidate_failures'] else 'finished') if len(tags)==47 else 'finished_subset';state['finished_utc']=stamp();persist()
    save(out/'manifest.json',dict(identity=identity,trajectories={tag:sha(out/'trajectories'/f'{tag}.txt') for tag in tags},
        evaluation_sha256=sha(out/'evaluation.json'),source_manifest_sha256=sha(out/'source/source_manifest.json'),
        count=len(tags),candidate_failures=state['candidate_failures'],completed_utc=stamp(),note='References generated and audited for the stated count. Selection gates characterize consistency, not absolute accuracy. Review quality flags and rejected candidates.'))
    print('Generated and audited',len(tags),'references in',out,flush=True);return 0

if __name__=='__main__':sys.exit(main())
