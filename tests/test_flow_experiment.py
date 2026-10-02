import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from lima3d.flow_experiment import run, sample_indices
from lima3d.optical_flow import block_motion, validate_options

class FlowExperimentTests(unittest.TestCase):
    def test_sampling(self):
        self.assertEqual(sample_indices(11,2,3),[0,4,8])
        self.assertEqual(sample_indices(2,2,5),[])
        self.assertEqual(sample_indices(4,1,10),[0,1,2])

    def test_diagnostics_preserve_statistics(self):
        options=validate_options(dict(block_size=16,block_stride=16,search_radius=8,mode_threshold_px=3))
        reference=np.random.default_rng(8).integers(0,256,(64,96),dtype=np.uint8)
        current=cv2.warpAffine(reference,np.float32([[1,0,3],[0,1,0]]),(96,64))
        plain=block_motion(reference,current,options)
        detailed=block_motion(reference,current,options,return_blocks=True)
        blocks=detailed.pop('blocks')
        self.assertEqual(plain,detailed)
        self.assertEqual(len(blocks),24)
        self.assertEqual(sum(b['status']=='valid' for b in blocks),plain['valid_blocks'])
        masked=block_motion(reference,current,options,np.zeros_like(reference),return_blocks=True)
        self.assertIsNone(masked['mode_px'])
        self.assertTrue(all(b['status']=='masked_reference' for b in masked['blocks']))

    def test_small_experiment_exports_and_keeps_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video_dir=root/'dataset/Scene/Car/cam0';video_dir.mkdir(parents=True)
            video=video_dir/'clip.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'FFV1'),10,(64,48))
            self.assertTrue(writer.isOpened())
            base=np.random.default_rng(3).integers(0,256,(48,64,3),dtype=np.uint8)
            for i in range(11):writer.write(cv2.warpAffine(base,np.float32([[1,0,i],[0,1,0]]),(64,48)))
            writer.release()
            original=video.read_bytes()
            config=root/'config.yaml'
            config.write_text('data: {datasets_root: dataset}\nextraction:\n  optical_flow: {block_size: 16, block_stride: 16, search_radius: 8, mode_threshold_px: 3}\n')
            args=SimpleNamespace(config=config,scene='Scene',platforms=None,output=root/'report',
                                 samples_per_video=3,lag_seconds=.2,threads=1)
            out=run(args)
            records=[json.loads(line) for line in (out/'pairs.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),3)
            self.assertTrue(all(r['mode_px']==2 for r in records))
            self.assertEqual(json.loads((out/'experiment.json').read_text())['status'],'complete')
            self.assertIn('data:image/png;base64', (out/'report.html').read_text())
            self.assertTrue((out/'figures/platform_distributions.pdf').is_file())
            self.assertEqual(video.read_bytes(),original)
            with self.assertRaises(ValueError):run(args)
