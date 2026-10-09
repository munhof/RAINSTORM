"""Generate immutable source manifest and audit full v2 preparation for study 8."""
import hashlib,json,sqlite3,time
from pathlib import Path
from uuid import uuid4
from storm.suite import default_catalog
from storm.experiment_nodes import experiment_registry
from storm.experiments import ExperimentExecutor
from storm.artifacts import FileArtifactStore
from rainstorm_thesis.plugin import register
import argparse
parser=argparse.ArgumentParser(description='Audit and prepare study 8 without opening test assets')
parser.add_argument('--workspace', type=Path, required=True)
parser.add_argument('--roi', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
workspace=args.workspace.resolve()
root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
c=sqlite3.connect(f'file:{workspace}/studio.sqlite3?mode=ro',uri=True);c.row_factory=sqlite3.Row
study=c.execute('select * from storm_studio_study where id=8').fetchone()
row=c.execute('select * from storm_studio_datasetrevision where id=?',(study['dataset_revision_id'],)).fetchone()
config,inventory=json.loads(row['config']),json.loads(row['inventory'])
sessions=[]
for id in json.loads(row['asset_ids']):
 a=c.execute('select * from storm_studio_datasetasset where id=?',(id,)).fetchone()
 if a['role']!='pose':continue
 sid=config.get('asset_sessions',{}).get(str(id),a['session_id']);partition=config['session_partitions'][sid]
 sessions.append(dict(id=sid,partition=partition,path=str(workspace/a['relative_path']),sha256=a['sha256']))
manifest=root/'sessions.json';manifest.write_text(json.dumps(dict(version='1',sessions=sessions),indent=2))
roi=args.roi.resolve();roi_data=json.loads(roi.read_text())
parts=[name[:-2] for name in inventory['feature_names'] if name.endswith('_x')]
recipe=dict(calibration_ref=dict(path=str(roi),sha256=hashlib.sha256(roi.read_bytes()).hexdigest()),bodyparts=parts,px_per_cm_x=roi_data['scale'],px_per_cm_y=roi_data['scale'],flip_y_height=roi_data['frame_shape'][1])
graph=dict(version='1',nodes=[dict(id='source',type='data.pose_sessions',config=dict(manifest_path=str(manifest),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())),dict(id='train_prepare',type='pose.prepare_sessions',config=recipe),dict(id='validation_prepare',type='pose.prepare_sessions',config=recipe),dict(id='sample',type='pose.sample_windows',config=dict(limit=256,seed=156))],edges=[dict(source='source.train',target='train_prepare.context'),dict(source='source.validation',target='validation_prepare.context'),dict(source='train_prepare.context',target='sample.context')])
(root/'graph.json').write_text(json.dumps(graph,indent=2))
evidence=dict(roi_path=str(roi),roi_sha256=hashlib.sha256(roi.read_bytes()).hexdigest(),scale=roi_data['scale'],frame_shape=roi_data['frame_shape'],scientific_equivalence=False,historical_difference='Native historical pose_ego did not explicitly load this ROI; v2 declares calibration and Y convention.')
(root/'calibration.json').write_text(json.dumps(evidence,indent=2))
catalog=default_catalog();register(catalog);catalog.experiment_artifact_root=root/'prepared'
print('Preparing every declared training/validation session; test sources unopened',flush=True)
t=time.monotonic();result=ExperimentExecutor(experiment_registry(catalog)).run(graph)
report=result.manifest();report['calibration']=evidence;report['elapsed_seconds']=time.monotonic()-t
report['partitions']={name:dict(windows=len(value['context'].data),sessions=value['context'].state['preparation_audit']) for name,value in result.outputs.items() if name.endswith('_prepare')}
ref=FileArtifactStore(root/'artifacts').save(kind='prepared_contexts',artifact_id='study8-v2-'+uuid4().hex,value=result.outputs,metadata={'fingerprint':result.fingerprint})
report['outputs_ref']=ref.to_dict();(root/'preparation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(status=result.status,nodes=result.records,seconds=report['elapsed_seconds'])),flush=True)
