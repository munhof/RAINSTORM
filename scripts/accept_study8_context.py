"""Execute the scientific graph with explicit acceptance limits and save evidence."""
import json,time,hashlib,sys
from pathlib import Path
from uuid import uuid4
from storm.suite import default_catalog
from storm.experiment_nodes import experiment_registry
from storm.experiments import ExperimentExecutor
from storm.artifacts import FileArtifactStore
from rainstorm_thesis.plugin import register
import argparse
parser=argparse.ArgumentParser(description='One-epoch CPU acceptance; leaves test untouched')
parser.add_argument('--prepared-root', type=Path, required=True)
parser.add_argument('--artifact-root', type=Path, required=True)
args=parser.parse_args()
root=args.prepared_root.resolve()
preparation=json.loads((root/'preparation.json').read_text())
if preparation['status'] != 'complete':
 raise SystemExit('Every training/validation session must pass preparation before training')
graph=json.loads((root/'graph.json').read_text())
for name in ('train','validation'):
 graph['nodes'].append(dict(id=name+'_inputs',type='adapter.select',config=dict(data='data',targets='none')))
graph['nodes'].append(dict(id='vame',type='model.vame_native',config=dict(epochs=1,device='cpu',n_states=50,latent_dim=20,hidden_dim=128,batch_size=256,learning_rate=.001,kld_weight=.5,seed=156)))
graph['edges']+= [dict(source='sample.context',target='train_inputs.context'),dict(source='validation_prepare.context',target='validation_inputs.context'),dict(source='train_inputs.context',target='vame.train'),dict(source='validation_inputs.context',target='vame.validation')]
(root/'acceptance-graph.json').write_text(json.dumps(graph,indent=2))
catalog=default_catalog();register(catalog);catalog.experiment_artifact_root=root/'prepared'
t=time.monotonic();result=ExperimentExecutor(experiment_registry(catalog)).run(graph)
report=result.manifest();report.update(seed=156,variant_id='native-v2-acceptance',run_id='study8-native-v2-one-epoch-'+uuid4().hex,elapsed_seconds=time.monotonic()-t,acceptance_limits=dict(epochs=1,train_windows=256,device='cpu'),scientific_equivalence=False,test_policy='reserved source files never opened',calibration=json.loads((root/'calibration.json').read_text()))
ref=FileArtifactStore(args.artifact_root.resolve()).save(kind='experiment_outputs',artifact_id=report['run_id'],value=result.outputs,metadata={'fingerprint':result.fingerprint,'run_id':report['run_id']});report['outputs_ref']=ref.to_dict()
if 'vame' in result.outputs:
 context=result.outputs['vame']['predictions'];model=context.artifacts['model']
 report.update(training_history=model.checkpoint['training_history'],validation_predictions=len(context.data),validation_sessions=sorted(set(context.metadata['sessions'])),training_sampling=result.outputs['sample']['context'].state['sampling'],preparation={name:dict(windows=len(result.outputs[name]['context'].data),sessions=result.outputs[name]['context'].state['preparation_audit']) for name in ('train_prepare','validation_prepare')})
(root/'acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(status=result.status,nodes=result.records,seconds=report['elapsed_seconds'],validation_predictions=report.get('validation_predictions'),training_history=report.get('training_history'))),flush=True)
if result.status!='complete':sys.exit(1)
