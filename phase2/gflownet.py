"""Stationary multiobjective reward; predictor is frozen before generation."""
from core import *
import argparse, time, joblib, torch, pandas as pd
from networks import FlowPolicy, PeptideLM, seed_all, sample, log_probability, descriptors, trajectory_balance

DEST=ROOT/'phase1/artifacts/multiobjective'
def load():
    policy=FlowPolicy().cuda()
    policy.load_state_dict(torch.load(ROOT/'phase1/artifacts/checkpoints/gflownet.pt',weights_only=True))
    return policy

def train(args):
    DEST.mkdir(parents=True,exist_ok=True); seed_all(7042)
    policy=load(); teacher=PeptideLM().cuda()
    teacher.load_state_dict(torch.load(ROOT/'phase1/artifacts/checkpoints/lm.pt',weights_only=True))
    teacher.eval().requires_grad_(False)
    features=Features(); oracle=joblib.load(OUT/'mic_regressor.joblib')
    amp=fasta(ROOT/'phase1/artifacts/data/amp_train.fasta'); lengths_pool=np.array(list(map(len,amp)))
    means=descriptors(amp).mean(0); scales=descriptors(amp).std(0).clip(.05)
    refs=sorted(set(fasta(ROOT/'data/reference/antibacterial.fasta')))
    anchors=sorted(amp,key=lambda s:__import__('hashlib').sha256(s.encode()).hexdigest())[:512]
    config={'seed':7042,'steps':args.steps,'batch_size':64,'oracle_sha256':sha(OUT/'mic_regressor.joblib'),
        'initial_policy_sha256':sha(ROOT/'phase1/artifacts/checkpoints/gflownet.pt'),
        'teacher_sha256':sha(ROOT/'phase1/artifacts/checkpoints/lm.pt'),
        'reference_sha256':sha(ROOT/'data/reference/antibacterial.fasta'),
        'code_sha256':sha(__file__), 'reward':'LM log-density + 2*clip(log10(16)-pred_log10_MIC,-2,2) + 1.5*(1-max_reference_edit_similarity) + 0.5*(1-mean_anchor_similarity_squared) - property_penalty - 8*exact_known',
        'diversity':'Fixed 512-anchor repulsion, stationary proxy; NOT batch-dependent reward or proven diversity.',
        'limitations':'ESM2-8M and GRU policy; surrogate MIC; no convergence or potency guarantee. Conditional length distribution retained from training.'}
    optimizer=torch.optim.Adam([{'params':[p for n,p in policy.named_parameters() if n!='log_z'],'lr':1e-4},{'params':[policy.log_z],'lr':.01}])
    history=[]; start=0
    if args.resume:
        state=torch.load(DEST/'resume.pt',weights_only=False)
        if state['config']!=config:raise RuntimeError('Resume mismatch')
        policy.load_state_dict(state['policy']);optimizer.load_state_dict(state['optimizer']);start=state['step'];history=state['history']
    elif (DEST/'resume.pt').exists():raise FileExistsError('Use --resume')
    begun=time.time()
    for step in range(start,args.steps):
        seed_all(7042+step); rng=np.random.default_rng(7042+step)
        sequences=sample(policy,rng.choice(lengths_pool,64),'cuda')
        pred=oracle.predict(features(sequences))
        novelty=1-max_similarity(sequences,refs)
        density=process.cdist(sequences,anchors,scorer=Levenshtein.normalized_similarity,workers=4,dtype=np.float32)
        spread=1-(density**2).mean(1)
        penalty=np.minimum(((descriptors(sequences)-means)/scales)**2,9).mean(1)
        with torch.no_grad():
            prior,lengths=log_probability(teacher,sequences,'cuda')
            reward=prior+torch.tensor(2*np.clip(np.log10(16)-pred,-2,2)+1.5*novelty+.5*spread-penalty-8*(novelty==0),device='cuda',dtype=torch.float32)
        policy.train(); logpf,lengths=log_probability(policy,sequences,'cuda')
        loss=trajectory_balance(policy.log_z[lengths],logpf,reward)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite TB loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),5);optimizer.step()
        if (step+1)%25==0 or step+1==args.steps:
            row={'step':step+1,'tb_mse':loss.item(),'mean_pred_log10_MIC':float(pred.mean()),'mean_novelty':float(novelty.mean()),'batch_unique_fraction':len(set(sequences))/64,'elapsed_seconds':time.time()-begun}
            history.append(row);print(json.dumps(row),flush=True)
            temp=DEST/'resume.tmp';torch.save({'config':config,'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'history':history},temp);temp.replace(DEST/'resume.pt')
            dump(DEST/'history.json',history)
    torch.save(policy.state_dict(),DEST/'gflownet.pt')
    dump(DEST/'metadata.json',{**config,'status':'completed','history':history,'checkpoint_sha256':sha(DEST/'gflownet.pt'),'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})

def generate(args):
    seed_all(8042);policy=FlowPolicy().cuda()
    policy.load_state_dict(torch.load(DEST/'gflownet.pt',weights_only=True));policy.eval()
    lengths=np.array(list(map(len,fasta(ROOT/'phase1/artifacts/data/amp_train.fasta'))))
    known=set(fasta(ROOT/'data/reference/antibacterial.fasta'))
    for name in ['train','validation']:
        known.update(r['sequence'] for r in json.loads((ROOT/f'protein_lm/artifacts/data/{name}.json').read_text()))
    known.update(pd.read_csv(OUT/'dataset.csv').sequence)
    path=Path(args.output) if args.output else DEST/'pool.fasta';progress=path.with_suffix('.json')
    config={'seed':8042,'batch_size':128,'checkpoint_sha256':sha(DEST/'gflownet.pt'),'code_sha256':sha(__file__),'dataset_sha256':sha(OUT/'dataset.csv')}
    seqs=[];seen=set();start=0
    if path.exists():
        if not args.resume:raise FileExistsError('Use --resume')
        state=json.loads(progress.read_text())
        if state['config']!=config or state['sha256']!=sha(path):raise RuntimeError('Resume mismatch')
        seqs=fasta(path);seen=set(seqs);start=state['next_round']
    for r in range(start,2000):
        if len(seqs)>=args.count:break
        seed_all(8042+r);rng=np.random.default_rng(8042+r)
        batch=sample(policy,rng.choice(lengths,128),'cuda')
        for s in batch:
            if valid(s) and s not in known and s not in seen:seen.add(s);seqs.append(s)
        temp=path.with_suffix('.tmp');write_fasta(temp,seqs,'gflownet_multiobjective');temp.replace(path)
        dump(progress,{'config':config,'accepted':len(seqs),'next_round':r+1,'sha256':sha(path)})
        if r%20==0:print('accepted',len(seqs),flush=True)
    if len(seqs)<args.count:raise RuntimeError('Quota unmet')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','generate']);p.add_argument('--steps',type=int,default=1000);p.add_argument('--count',type=int,default=17500);p.add_argument('--resume',action='store_true');p.add_argument('--output');args=p.parse_args()
    (train if args.stage=='train' else generate)(args)
