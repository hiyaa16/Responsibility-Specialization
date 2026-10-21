#!/usr/bin/env python3
"""Build length-matched C1/C2 prompts using the experiment's live Ollama runtime.

Run on the machine that has the exact experimental Ollama models loaded:
  python build_and_count_prompts.py --url http://localhost:11434 --model qwen2.5:7b

The script counts the FULL prompt with a fixed probe (same approach as v2's token audit),
then greedily appends whole generic sentences to each specialized prompt until it is
within the declared tolerance of the generalist target. It never edits A or B. It emits
candidate prompts, counts, hashes, and a report. It does not establish semantic neutrality.
"""
from pathlib import Path
import argparse, hashlib, json, urllib.request, csv, sys
HERE=Path(__file__).resolve().parent
PROBE='TASK:\nPROBE QUESTION\n'

def sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def api_count(system, url, model, timeout):
    messages=[{'role':'system','content':system},{'role':'user','content':PROBE}]
    body={'model':model,'messages':messages,'stream':False,'options':{'num_predict':1,'temperature':0,'seed':0}}
    req=urllib.request.Request(url.rstrip('/')+'/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r: data=json.loads(r.read().decode())
    if 'prompt_eval_count' not in data: raise RuntimeError(f'No prompt_eval_count in response: {data.keys()}')
    return int(data['prompt_eval_count'])
def read(n): return (HERE/n).read_text(encoding='utf-8').strip()
def make_prompt(pre, blocks): return pre.strip()+'\n\n'+'\n\n'.join(b.strip() for b in blocks)+'\n'
def append_filler(base, family, target, url, model, timeout):
    sentences=[x.strip() for x in family.replace('\n',' ').split('. ') if x.strip()]
    sentences=[s if s.endswith('.') else s+'.' for s in sentences]
    current=base.rstrip(); used=[]; count=api_count(current,url,model,timeout)
    # Greedy sentence selection; count actual full prompt each time. Sentences are used at most once.
    remaining=sentences[:]
    while count < target and remaining:
        options=[]
        for s in remaining:
            trial=current+' '+s
            tc=api_count(trial,url,model,timeout)
            options.append((abs(target-tc), tc, s))
        _, best_count, best= min(options, key=lambda x:(x[0], x[1]))
        # Never add a sentence that overshoots more than the current gap if another smaller move exists.
        current=current+' '+best; used.append(best); remaining.remove(best); count=best_count
        if count>=target: break
    return current+'\n',count,used

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--url',default='http://localhost:11434'); ap.add_argument('--model',default='qwen2.5:7b'); ap.add_argument('--tolerance',type=float,default=0.01); ap.add_argument('--timeout',type=int,default=180); args=ap.parse_args()
    try:
        pre=read('preamble.txt'); blocks={r:read('block_'+r+'.txt') for r in ['retrieve','solve','verify']}
        A=make_prompt(pre,[blocks['retrieve'],blocks['solve'],blocks['verify']])
        B={r:make_prompt(pre,[blocks[r]]) for r in blocks}
        fam={k:read('filler_family_'+k.lower()+'.txt') for k in ['C1','C2']}
        # counts include fixed user probe/template, constant cancels in comparisons
        target=api_count(A,args.url,args.model,args.timeout)
        rows=[]; prompts={'A_generalist':A}
        rows.append({'model':args.model,'condition':'A','agent':'all','target_or_actual_prompt_eval_count':target,'target_count':target,'delta_tokens':0,'delta_percent':0,'within_tolerance':True,'sha256':sha(A)})
        for role,bprompt in B.items():
            bc=api_count(bprompt,args.url,args.model,args.timeout)
            rows.append({'model':args.model,'condition':'B','agent':role,'target_or_actual_prompt_eval_count':bc,'target_count':target,'delta_tokens':bc-target,'delta_percent':100*(bc-target)/target,'within_tolerance':abs(bc-target)<=args.tolerance*target,'sha256':sha(bprompt)})
            for c in ['C1','C2']:
                padded,count,used=append_filler(bprompt,fam[c],target,args.url,args.model,args.timeout)
                key=f'{c}_{role}'; prompts[key]=padded
                delta=count-target; ok=abs(delta)<=args.tolerance*target
                rows.append({'model':args.model,'condition':c,'agent':role,'target_or_actual_prompt_eval_count':count,'target_count':target,'delta_tokens':delta,'delta_percent':100*delta/target,'within_tolerance':ok,'sha256':sha(padded),'sentences_added':len(used)})
        (HERE/'generated_candidate_prompts').mkdir(exist_ok=True)
        for key,value in prompts.items(): (HERE/'generated_candidate_prompts'/f'{key}.txt').write_text(value,encoding='utf-8')
        with (HERE/'runtime_token_counts.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row})); w.writeheader(); w.writerows(rows)
        report=[f'Model: {args.model}',f'Endpoint: {args.url.rstrip("/")}/api/chat',f'Target A full-prompt prompt_eval_count: {target}',f'Tolerance: +/-{args.tolerance*100:.2f}%','', 'Counts include fixed probe/template; compare only within this model/runtime.']
        for r in rows[1:]: report.append(f"{r['condition']} / {r['agent']}: {r['target_or_actual_prompt_eval_count']} (delta {r['delta_tokens']}, {r['delta_percent']:.2f}%), within tolerance={r['within_tolerance']}")
        report += ['', 'IMPORTANT: Token matching is not proof of behavioral neutrality. Review sentence choices, freeze prompt hashes, and treat pilot only as operational validation.']
        (HERE/'runtime_token_count_report.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
        print('\n'.join(report))
    except Exception as e:
        print('TOKENIZER CHECK FAILED:',repr(e),file=sys.stderr)
        print('Start Ollama and confirm the exact model tag is installed; no token counts were fabricated.',file=sys.stderr)
        sys.exit(2)
if __name__=='__main__': main()
