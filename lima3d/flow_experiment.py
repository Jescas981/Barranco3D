"""Reproducible, fixed-lag optical-flow diagnostics on stratified video samples."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from .optical_flow import block_motion, magnitude_statistics, validate_options
from .regions import parse_regions, region_for_name


def sample_indices(frame_count, lag_frames, samples):
    import numpy as np
    last = frame_count - lag_frames - 1
    if last < 0:
        return []
    return np.unique(np.linspace(0, last, min(samples, last + 1)).astype(int)).tolist()


def summarize(records, bin_width):
    import numpy as np
    values = [r['mode_px'] for r in records if r['mode_px'] is not None]
    valid = sum(r['valid_blocks'] for r in records)
    total = sum(len(r['blocks']) for r in records)
    supports=[];ties=[]
    for record in records:
        bins=Counter(math.floor(b['magnitude_px']/bin_width) for b in record['blocks'] if b['status']=='valid')
        if bins:
            largest=max(bins.values())
            supports.append(largest/sum(bins.values()))
            ties.append(sum(n==largest for n in bins.values())>1)
    statuses = Counter(b['status'] for r in records for b in r['blocks'])
    return dict(pairs=len(records), pairs_with_flow=len(values), pairs_without_flow=len(records)-len(values),
                median_mode_px=float(np.median(values)) if values else None,
                p90_mode_px=float(np.percentile(values, 90)) if values else None,
                median_modal_bin_support=float(np.median(supports)) if supports else None,
                tied_mode_fraction=float(np.mean(ties)) if ties else None,
                valid_blocks=valid, total_blocks=total, valid_block_fraction=valid/total if total else None,
                block_statuses=dict(statuses),
                search_limit_fraction=sum(b['at_search_limit'] for r in records for b in r['blocks'])/valid if valid else None)


def run(args):
    import cv2
    import numpy as np
    import yaml
    started = time.perf_counter()
    config = args.config.resolve()
    raw = yaml.safe_load(config.read_text())
    data = raw.get('data', {})
    dataset = (config.parent / data.get('datasets_root', 'dataset')).resolve()
    scene = dataset / args.scene
    if not scene.is_dir():
        raise ValueError(f'Scene does not exist: {scene}')
    platforms = args.platforms or sorted(p.name for p in scene.iterdir() if p.is_dir() and not p.name.startswith('.'))
    regions = parse_regions(raw.get('regions'), config.parent, platforms)
    options = validate_options(raw.get('extraction', {}).get('optical_flow'))
    videos = sorted(p for p in scene.rglob('*') if p.suffix.lower() in ('.mp4','.mkv','.mov','.avi','.m4v','.webm','.mts')
                    and p.relative_to(scene).parts[0] in platforms)
    if not videos:
        raise ValueError('No videos found for the requested platforms')
    output = args.output.resolve() if args.output else config.parent / 'outputs/flow_experiments' / args.scene / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    (output/'previews').mkdir(exist_ok=True)
    metadata = dict(scene=args.scene, dataset=str(dataset), comparison='fixed_time_lag',
        requested_lag_seconds=args.lag_seconds, samples_per_video=args.samples_per_video,
        platforms=platforms, options=options, regions=regions, threads=args.threads,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).with_name('optical_flow.py').read_bytes()).hexdigest(),
        opencv_version=cv2.__version__, created_utc=datetime.now(timezone.utc).isoformat(),
        weighting='Each sampled pair has equal weight; each clip has up to samples_per_video pairs.',
        status='running', videos=[])
    records=[];previews={};skipped=[]
    old_threads=cv2.getNumThreads();cv2.setNumThreads(args.threads)
    try:
        with (output/'pairs.jsonl').open('w') as log:
            for video_number, video in enumerate(videos,1):
                relative=video.relative_to(scene).as_posix();camera=video.relative_to(scene).parent.as_posix()
                platform=video.relative_to(scene).parts[0]
                region=region_for_name(regions,relative)
                if region['mode']=='mask' and 'path' not in region:
                    raise ValueError(f'{relative}: shared mask path required for video analysis')
                cap=cv2.VideoCapture()
                params=[cv2.CAP_PROP_N_THREADS,args.threads] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
                if not cap.open(str(video),cv2.CAP_FFMPEG,params):
                    raise ValueError(f'Cannot open {video}')
                try:
                    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
                    fps=cap.get(cv2.CAP_PROP_FPS);frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if not math.isfinite(fps) or fps<=0 or frame_count<=0:
                        raise ValueError(f'{relative}: source FPS and frame count are required')
                    lag=max(1,round(args.lag_seconds*fps))
                    indices=sample_indices(frame_count,lag,args.samples_per_video)
                    mask=cv2.imread(region['path'],cv2.IMREAD_GRAYSCALE) if region['mode']=='mask' else None
                    if region['mode']=='mask' and mask is None:
                        raise ValueError(f'Cannot read mask: {region["path"]}')
                    if mask is not None:
                        mask_hash=hashlib.sha256(Path(region['path']).read_bytes()).hexdigest()
                    else:mask_hash=None
                    video_info=dict(video=relative, camera=camera, fps=fps, frame_count=frame_count,
                        lag_frames=lag, nominal_lag_seconds=lag/fps, planned_pairs=len(indices),
                        size_bytes=video.stat().st_size,mtime_ns=video.stat().st_mtime_ns,
                        region=region,mask_sha256=mask_hash)
                    metadata['videos'].append(video_info)
                    print(f'[{video_number}/{len(videos)}] {relative}: {len(indices)} pairs, lag={lag/fps:.4f}s',flush=True)
                    for sample_number, reference_index in enumerate(indices,1):
                        pair=[];times=[];actual_indices=[]
                        for index in (reference_index,reference_index+lag):
                            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                            ok,frame=cap.read()
                            if not ok:break
                            actual_indices.append(int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1)
                            times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                            if region['mode']=='crop':
                                l,t,r,b=region['box']
                                if not (0<=l<r<=frame.shape[1] and 0<=t<b<=frame.shape[0]):
                                    raise ValueError(f'Crop exceeds video: {relative}')
                                frame=frame[t:b,l:r]
                            pair.append(frame)
                        if len(pair)!=2 or actual_indices != [reference_index,reference_index+lag]:
                            skipped.append(dict(video=relative,reference_index=reference_index,reason='decode_or_seek_failed'))
                            continue
                        gray0=cv2.cvtColor(pair[0],cv2.COLOR_BGR2GRAY);gray1=cv2.cvtColor(pair[1],cv2.COLOR_BGR2GRAY)
                        if mask is not None and mask.shape!=gray0.shape:
                            raise ValueError(f'Mask shape differs from video: {relative}')
                        result=block_motion(gray0,gray1,options,mask,return_blocks=True)
                        # Never merge cameras/clips with different grids or dimensions into one map.
                        grid=f'{camera}@{gray0.shape[1]}x{gray0.shape[0]}'
                        if grid not in previews:
                            filename=hashlib.sha256(grid.encode()).hexdigest()[:12]+'.jpg'
                            if not cv2.imwrite(str(output/'previews'/filename),pair[0]):
                                raise RuntimeError('Failed to write preview')
                            previews[grid]=dict(path=f'previews/{filename}',camera=camera,width=gray0.shape[1],height=gray0.shape[0],
                                                video=relative,source_frame_index=reference_index,region=region)
                        measured_dt=times[1]-times[0]
                        timestamp_valid=all(math.isfinite(t) and t>=0 for t in times) and measured_dt>0
                        record=dict(platform=platform,camera=camera,grid=grid,video=relative,
                            reference_index=reference_index,current_index=reference_index+lag,
                            reference_seconds=times[0] if math.isfinite(times[0]) else None,
                            current_seconds=times[1] if math.isfinite(times[1]) else None,
                            dt_seconds=measured_dt if timestamp_valid else lag/fps,
                            time_source='decoder' if timestamp_valid else 'nominal_fps',**result)
                        records.append(record);log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                        if sample_number%6==0 or sample_number==len(indices):
                            print(f'  {sample_number}/{len(indices)} pairs | mode={result["mode_px"]} px | valid blocks={result["valid_blocks"]}/{len(result["blocks"])}',flush=True)
                    video_info['measured_pairs']=sum(r['video']==relative for r in records)
                    (output/'experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
                finally:
                    cap.release()
        if not records:
            raise ValueError('No pairs could be measured')
        summary={platform:summarize([r for r in records if r['platform']==platform],options['mode_bin_width_px']) for platform in platforms}
        metadata.update(status='complete',elapsed_seconds=time.perf_counter()-started,skipped_pairs=skipped,measured_pairs=len(records))
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        render(output,records,previews,summary,metadata)
        print(json.dumps(summary,indent=2),flush=True)
        print(f'Report: {output / "report.html"}',flush=True)
    except BaseException as exc:
        metadata.update(status='failed',error=str(exc))
        raise
    finally:
        metadata['elapsed_seconds']=time.perf_counter()-started
        (output/'experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
        cv2.setNumThreads(old_threads)
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',type=Path,help='Regenerate figures/HTML from an existing experiment without measuring videos again')
    parser.add_argument('--config',type=Path,default=Path('config.yaml'))
    parser.add_argument('--scene',default='DavidHouse')
    parser.add_argument('--platforms',nargs='+',help='Default: all platform folders, independent of reconstruction platform filters')
    parser.add_argument('--samples-per-video',type=int,default=24)
    parser.add_argument('--lag-seconds',type=float,default=.2)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.samples_per_video<1 or args.threads<1 or not math.isfinite(args.lag_seconds) or args.lag_seconds<=0:
        parser.error('Sample count, threads and lag must be positive')
    if Path(args.scene).name!=args.scene or args.scene in ('.','..'):
        parser.error('scene must be one folder name')
    try:
        if args.report_only:refresh_report(args.report_only.resolve())
        else:run(args)
    except (ValueError,OSError,RuntimeError) as exc:parser.exit(1,f'Error: {exc}\n')


def render(output, records, previews, summary, metadata):
    import base64
    import html
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable
    from PIL import Image
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    platforms=[p for p in summary if summary[p]['pairs']]
    colors=['#276978','#b26d29','#62616a']
    maximum=math.sqrt(2)*metadata['options']['search_radius']
    edges=np.arange(0,math.ceil(maximum/5)*5+5,5)
    fig,axes=plt.subplots(2,len(platforms),figsize=(5*len(platforms),8),squeeze=False,sharex=True,sharey='row')
    for column,platform in enumerate(platforms):
        group=[r for r in records if r['platform']==platform]
        modes=[r['mode_px'] for r in group if r['mode_px'] is not None]
        blocks=[b['magnitude_px'] for r in group for b in r['blocks'] if b['status']=='valid']
        for row,values in enumerate((modes,blocks)):
            ax=axes[row,column]
            if values:
                ax.hist(values,bins=edges,weights=np.full(len(values),100/len(values)),color=colors[column%3],edgecolor='white',linewidth=.4)
                ax.axvline(np.median(values),color='#242424',linestyle='--',linewidth=1,label=f'Median = {np.median(values):.2f} px')
                ax.legend(fontsize=8)
            ax.set_xlabel('Displacement magnitude (px)')
            ax.set_ylabel('Valid observations per 5 px bin (%)')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            metric='Mode across blocks per frame pair' if row==0 else 'All valid block displacements'
            ax.set_title(f'{platform} — {metric}\nn={len(values)} valid observations',fontsize=11)
        info=summary[platform]
        axes[0,column].text(.98,.70,f'{info["pairs_without_flow"]}/{info["pairs"]} pairs without flow\n{100*info["valid_block_fraction"]:.1f}% of blocks valid',
                            transform=axes[0,column].transAxes,ha='right',fontsize=9)
    fig.suptitle(f'{metadata["scene"]}: fixed-lag optical-flow distributions',fontsize=17)
    fig.text(.5,.015,f'Source: original scene videos, uniformly sampled throughout each clip | requested lag {metadata["requested_lag_seconds"]:g} s | masks/crops from config | no selection threshold applied',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95))
    overview=output/'figures/platform_distributions.png';fig.savefig(overview,dpi=160);fig.savefig(overview.with_suffix('.pdf'));plt.close(fig)
    maps=[];block_rows=[]
    for grid,preview in sorted(previews.items()):
        group=[r for r in records if r['grid']==grid];by_block=defaultdict(list)
        for record in group:
            for block in record['blocks']:by_block[(block['x'],block['y'])].append(block)
        summaries=[]
        for (x,y),observations in sorted(by_block.items()):
            magnitudes=[b['magnitude_px'] for b in observations if b['status']=='valid']
            stats=magnitude_statistics(magnitudes,metadata['options']['mode_bin_width_px'])
            support=max(Counter(math.floor(v/metadata['options']['mode_bin_width_px']) for v in magnitudes).values())/len(magnitudes) if magnitudes else None
            item=dict(grid=grid,x=x,y=y,width=observations[0]['width'],height=observations[0]['height'],
                      observations=len(observations),valid_fraction=len(magnitudes)/len(observations),
                      modal_bin_support=support,**stats)
            summaries.append(item);block_rows.append(item)
        background=np.asarray(Image.open(output/preview['path']))
        fig,axs=plt.subplots(1,3,figsize=(17,5.6))
        for ax,key,title in zip(axs,['mode_px','median_px','valid_fraction'],['Temporal mode per block','Temporal median per block','Valid measurements per block']):
            ax.imshow(background,alpha=.42)
            is_coverage=key=='valid_fraction';cmap=plt.get_cmap('Blues' if is_coverage else 'viridis')
            norm=Normalize(0,100 if is_coverage else maximum)
            for item in summaries:
                value=item[key]
                if is_coverage:value*=100
                rectangle=Rectangle((item['x']-.5,item['y']-.5),item['width'],item['height'],
                    facecolor=cmap(norm(value)) if value is not None else '#c9c9c9',
                    edgecolor='white',linewidth=.4,alpha=.9,hatch='///' if value is None else None)
                ax.add_patch(rectangle)
                label=f'{value:.0f}%' if is_coverage else f'{value:.1f}' if value is not None else 'N/A'
                ax.text(item['x']+item['width']/2,item['y']+item['height']/2,label,ha='center',va='center',fontsize=7,
                        color='black',bbox=dict(facecolor='white',alpha=.78,edgecolor='none',pad=1))
            ax.set_title(title);ax.set_xlabel('Image x (px, after crop)');ax.set_ylabel('Image y (px, after crop)')
            fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,orientation='horizontal',pad=.19,fraction=.06,
                         label='Valid observations (%)' if is_coverage else 'Displacement magnitude (px)')
        fig.suptitle(f'{grid} | {len(group)} sampled pairs | requested lag {metadata["requested_lag_seconds"]:g} s',fontsize=14)
        fig.text(.5,.01,f'Grid locations pooled across sampled times; gray/hatching = no valid measurement. Background: {preview["video"]}, frame {preview["source_frame_index"]}.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,.93))
        filename='blocks_'+hashlib.sha256(grid.encode()).hexdigest()[:12]+'.png'
        fig.savefig(output/'figures'/filename,dpi=160);fig.savefig((output/'figures'/filename).with_suffix('.pdf'));plt.close(fig)
        maps.append(dict(grid=grid,path='figures/'+filename))
    (output/'blocks_summary.json').write_text(json.dumps(block_rows,indent=2)+'\n')
    def embed(path):return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
    camera_options=''.join(f'<option value="map{i}">{html.escape(item["grid"])}</option>' for i,item in enumerate(maps))
    images=''.join(f'<figure id="map{i}" class="camera" {"hidden" if i else ""}><img alt="Block motion and validity for {html.escape(item["grid"])}" src="{embed(output/item["path"])}"></figure>' for i,item in enumerate(maps))
    findings=''.join(f'<li><b>{html.escape(platform)}</b>: {info["pairs_with_flow"]}/{info["pairs"]} pairs with valid flow; median of pair modes: '
                     f'{info["median_mode_px"]:.2f} px; p90: {info["p90_mode_px"]:.2f} px; valid blocks: {100*info["valid_block_fraction"]:.1f}%. '
                     f'Median winning-bin support: {100*info["median_modal_bin_support"]:.1f}%; tied winning bins: {100*info["tied_mode_fraction"]:.1f}% of valid pairs.</li>'
                     for platform,info in summary.items() if info['median_mode_px'] is not None)
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(metadata['scene'])} — optical-flow experiment</title>
<style>body{{font:16px system-ui;color:#202a30;background:#f6f7f8;margin:0 auto;padding:30px;max-width:1450px}}h1{{font-size:27px}}h2{{font-size:20px;margin-top:30px}}p,li{{max-width:1050px;line-height:1.55}}img{{width:100%;height:auto}}figure{{margin:20px 0}}select{{padding:9px;font:inherit}}.note{{border-left:3px solid #276978;padding-left:16px}}small{{color:#58656a}}</style>
<h1>{html.escape(metadata['scene'])}: where optical flow moves</h1>
<p>Fixed-lag exploratory experiment: {len(records)} frame pairs across {len(metadata['videos'])} clips, requested interval {metadata['requested_lag_seconds']:g} s, up to {metadata['samples_per_video']} uniformly spaced pairs per clip. Original resolution; configured crops and masks applied.</p>
<p class="note">This measures motion at a common interval, independently of hybrid reference resets. It is not the distribution of an existing selection run, nor a prediction of reconstruction quality. Each pair has equal weight; clips are sampled equally, not in proportion to duration. Car contains multiple cameras. This is a sampled pilot, not a full-video census.</p>
<ul>{findings}</ul>
<h2>Platform distributions</h2><p>Top: mode across valid blocks for each frame pair. Bottom: individual valid block magnitudes. Percentages are normalized separately within each platform. Invalid blocks are excluded, never replaced by zero. Dashed lines show medians.</p>
<figure><img alt="Distributions of pair modes and block displacement by platform" src="{embed(overview)}"></figure>
<h2>Block maps by camera</h2><p>Choose a camera. The mode is computed over time independently for each block position; the median and valid-observation fraction show whether that estimate is representative. All maps use the same color scales. Unpainted gaps are outside the sampled block grid.</p>
<select aria-label="Camera block maps" onchange="document.querySelectorAll('.camera').forEach(e=>e.hidden=e.id!==this.value)">{camera_options}</select>{images}
<h2>Method and limits</h2><p>Normalized-correlation block matching; block {metadata['options']['block_size']} px, stride {metadata['options']['block_stride']} px, search ±{metadata['options']['search_radius']} px per axis, minimum correlation {metadata['options']['min_correlation']}. Modal bins: {metadata['options']['mode_bin_width_px']} px, median observation in the winning bin; ties favor the smallest bin. Sparse samples and ties can bias modes toward smaller displacements. Gray blocks have no accepted matches.</p>
<p>The interval is rounded to source frames; actual decoder timestamp differences are stored per pair. Large movements outside the search radius, weak texture, and occlusion can be rejected or mismatched. Differences between platforms reflect viewpoint, motion, optics and scene content, not physical speed alone.</p>
<p>Source: local dataset/{html.escape(metadata['scene'])}, entire recorded span sampled per clip. Analysis created {html.escape(metadata['created_utc'])}. Exact frames, times, block displacements and rejection reasons: pairs.jsonl. Provenance and parameters: experiment.json. Per-camera block support: blocks_summary.json. Standalone PNG/PDF figures: figures/.</p></html>'''
    (output/'report.html').write_text(document)


def refresh_report(output):
    """Replot saved measurements; never read or recompute source videos."""
    metadata=json.loads((output/'experiment.json').read_text())
    if metadata['status']!='complete':
        raise ValueError('Only completed experiments can be replotted')
    records=[json.loads(line) for line in (output/'pairs.jsonl').read_text().splitlines()]
    previews={}
    for record in records:
        grid=record['grid']
        if grid in previews:continue
        width,height=map(int,grid.rsplit('@',1)[1].split('x'))
        previews[grid]=dict(path='previews/'+hashlib.sha256(grid.encode()).hexdigest()[:12]+'.jpg',
            camera=record['camera'],width=width,height=height,video=record['video'],
            source_frame_index=record['reference_index'],region=region_for_name(metadata['regions'],record['video']))
    summary={platform:summarize([r for r in records if r['platform']==platform],metadata['options']['mode_bin_width_px'])
             for platform in metadata['platforms']}
    render(output,records,previews,summary,metadata)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Report: {output / "report.html"}',flush=True)
