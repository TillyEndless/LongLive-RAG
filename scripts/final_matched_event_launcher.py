import os, sys, json, time, runpy, csv
from pathlib import Path
import torch

repo = Path(os.environ['PROFILE_REPO'])
config = os.environ['PROFILE_CONFIG']
out = Path(os.environ['PROFILE_OUT']); out.mkdir(parents=True, exist_ok=True)
start_block = int(os.environ.get('PROFILE_START_BLOCK','6'))
end_block = int(os.environ.get('PROFILE_END_BLOCK','8'))
block_frames = int(os.environ.get('PROFILE_BLOCK_FRAMES','3'))
frame_seq = int(os.environ.get('PROFILE_FRAME_SEQ','1560'))
events=[]; active_layers={}; current_layer=-1; active=False

def block_ok(current_start):
    try: return start_block <= int(current_start)//(block_frames*frame_seq) < end_block
    except Exception: return False

def pair(label, layer=-1, extra=None):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
    return {'label':label,'layer':int(layer),'start':s,'end':e,'host_start':time.perf_counter_ns(),'extra':extra or {}}

def finish(rec):
    rec['end'].record()
    rec['host_end']=time.perf_counter_ns()
    events.append(rec)

# Import attention module and wrap its public attention function. Each call is
# timed with CUDA events, and no event is synchronized until inference ends.
if os.environ.get('PROFILE_MODULE','latentmem') == 'native':
    import wan.modules.causal_model as cm
elif (repo/'wan/modules/causal_model_latentmem.py').exists():
    import wan.modules.causal_model_latentmem as cm
else:
    import wan.modules.causal_model as cm
orig_attn=cm.attention
def timed_attention(*a, **kw):
    if not active: return orig_attn(*a, **kw)
    r=pair('BF16_ATTENTION', current_layer, {'q_tokens':int(a[0].shape[1]),'kv_tokens':int(a[1].shape[1])})
    outv=orig_attn(*a, **kw); finish(r); return outv
cm.attention=timed_attention

def pre_self(mod,args,kwargs):
    global active,current_layer
    cs=kwargs.get('current_start', args[6] if len(args)>6 else -1)
    active=block_ok(cs); current_layer=int(kwargs.get('layer_index',args[10] if len(args)>10 else -1))
    mod._final_profile_rec = pair('ATTN_WRAPPER', current_layer) if active else None
    if active:
        for name in ('q','k','v'):
            sub=getattr(mod,name,None)
            if sub is not None:
                sub._final_profile_pending=pair('QKV_'+name.upper(), current_layer)

def post_self(mod,args,kwargs,outv):
    global active
    if getattr(mod,'_final_profile_rec',None) is not None: finish(mod._final_profile_rec)
    active=False

def lin_pre(name, layer):
    def h(mod,args,kwargs):
        if active: mod._final_profile_pending=pair(name,layer)
    return h
def lin_post(name, layer):
    def h(mod,args,kwargs,outv):
        r=getattr(mod,'_final_profile_pending',None)
        if r is not None: finish(r); mod._final_profile_pending=None
    return h

# Load pipeline by executing the normal inference entry; install hooks after
# pipeline construction is difficult without editing production code, so patch
# the attention class registration hooks through nn.Module hook interception.
import torch.nn as nn
orig_register = nn.Module.register_forward_pre_hook
orig_register_post = nn.Module.register_forward_hook
old_init = cm.CausalWanSelfAttention.__init__
def new_init(self,*a,**kw):
    old_init(self,*a,**kw)
    self.register_forward_pre_hook(pre_self, with_kwargs=True)
    self.register_forward_hook(post_self, with_kwargs=True)
    for name in ('q','k','v'):
        sub=getattr(self,name); sub.register_forward_pre_hook(lin_pre('QKV_'+name.upper(),-1),with_kwargs=True); sub.register_forward_hook(lin_post('QKV_'+name.upper(),-1),with_kwargs=True)
    self.o.register_forward_pre_hook(lin_pre('OUTPUT_PROJECTION',-1),with_kwargs=True); self.o.register_forward_hook(lin_post('OUTPUT_PROJECTION',-1),with_kwargs=True)
cm.CausalWanSelfAttention.__init__=new_init

sys.argv=['inference.py','--config_path',config]
os.chdir(repo)
try:
    if os.environ.get('PROFILE_MODULE','latentmem') == 'native':
        import pipeline.causal_inference as _ci
        _ci.CausalInferencePipeline.num_heads = 12
        _ci.CausalInferencePipeline.head_dim = 128
    runpy.run_path(str(repo/'inference.py'), run_name='__main__')
finally:
    torch.cuda.synchronize()

rows=[]
for r in events:
    try: ms=float(r['start'].elapsed_time(r['end']))
    except Exception: ms=None
    rows.append({'phase':r['label'],'layer':r['layer'],'cuda_event_ms':ms,'host_start_ns':r['host_start'],'host_wall_ms':(r.get('host_end',r['host_start'])-r['host_start'])/1e6,'extra':json.dumps(r['extra'],sort_keys=True)})
with (out/'phase_events.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['phase','layer','cuda_event_ms','host_wall_ms','host_start_ns','extra']); w.writeheader(); w.writerows(rows)
summary={}
for r in rows:
    x=summary.setdefault(r['phase'],{'calls':0,'cuda_event_ms':0.0}); x['calls']+=1; x['cuda_event_ms']+=r['cuda_event_ms'] or 0
json.dump({'repo':str(repo),'config':config,'interval':[start_block,end_block],'event_pairs':'YES','single_final_synchronize':'YES','phases':summary},open(out/'summary.json','w'),indent=2)
print(json.dumps(summary))
