import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d import scheduler as s
from lima3d.artifacts import request_spec
from lima3d.pipeline import coalitions,save_json
from test_stage_portability import arguments,make_scene


class SchedulerTests(unittest.TestCase):
    def test_one_retrieval_job_shared_by_two_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene=make_scene(root)
            a=arguments(root,'sg');a.configs=['sp-sg']
            b=arguments(root,'lg');b.configs=['sp-lg']
            cfg=SimpleNamespace(output_root=root/'outputs',path=root/'config.yaml',
                scenes=[scene],experiments={'sg':a,'lg':b},
                res={'bank':{'threads':1},'bank_ids':[0,1],
                     'sfm':{'threads':2,'workers':2},'mvs':{'threads':1},'mvs_ids':[0]})
            bankroot=s.context(scene,a)[3]
            calls=[]
            class FakeProcess:
                def __init__(self,cmd,**kwargs):
                    self_outer.assertEqual(cmd[1:3], ['-m', 'lima3d.scheduler'])
                    self_outer.assertIn(str(s.REPO_ROOT), kwargs['env']['PYTHONPATH'].split(os.pathsep))
                    kind=cmd[cmd.index('--worker')+1]
                    exp=cmd[cmd.index('--experiment')+1]
                    calls.append((kind,exp))
                    self.pid=10000+len(calls)
                    if kind=='pairs':
                        save_json(bankroot/f'bank_index-{exp}.json',{'shared':True})
                    elif kind=='bank':
                        self_outer.assertTrue((bankroot/f'bank_index-{exp}.json').is_file())
                        args=cfg.experiments[exp]
                        jobs=[{'label':'+'.join(c),'members':[p+'/clip__000001.jpg' for p in c],
                               'pairs':'pairs.txt','features':'features.h5','matches':'matches.h5'}
                              for c in coalitions(a.platforms)]
                        save_json(bankroot/f'jobs/{exp}.json',
                                  {'preset':args.configs[0],'run_base':{'matches':exp,'seed':0,'camera_mode':'PER_FOLDER'},
                                   'request':request_spec(args),'config':{'preset':args.configs[0]},'jobs':jobs})
                    elif kind=='sfm':
                        job=json.loads(Path(cmd[cmd.index('--task-json')+1]).read_text())
                        self_outer.assertEqual(job['scene'],str(scene))
                        save_json(Path(job['result_file']),{'status':'no_model'})
                def poll(self): return 0
            self_outer=self
            with patch.object(s,'tqdm',None),patch.object(s.os,'sched_getaffinity',return_value=set(range(8))),patch.object(s.subprocess,'Popen',FakeProcess),patch.object(s.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
                failures=s.orchestrate(cfg,{'pairs','bank','sfm'})
            self.assertEqual(failures,0)
            self.assertEqual(sum(k=='pairs' for k,e in calls),1)
            self.assertEqual(sum(k=='bank' for k,e in calls),2)
            self.assertEqual(sum(k=='sfm' for k,e in calls),14)

    def test_mvs_reports_missing_sfm_instead_of_succeeding_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene=make_scene(root);a=arguments(root)
            cfg=SimpleNamespace(output_root=root/'outputs',path=root/'config.yaml',scenes=[scene],experiments={'a':a},
                res={'bank':{'threads':100},'bank_ids':[0,1,2,3],
                     'sfm':{'threads':100,'workers':1},'mvs':{'threads':1},'mvs_ids':[0]})
            bankroot=s.context(scene,a)[3]
            save_json(bankroot/'jobs/a.json',{'preset':'sift','request':request_spec(a),
                'run_base':{'matches':'sift','seed':0,'camera_mode':'PER_FOLDER'},'config':{},
                'jobs':[{'label':'Car','members':['Car/clip__000001.jpg'],'pairs':'p','features':'f','matches':'m'}]})
            with patch.object(s,'tqdm',None),patch.object(s.os,'sched_getaffinity',return_value={0,1}),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(s.orchestrate(cfg,{'mvs'}),1)


if __name__=='__main__':unittest.main()
