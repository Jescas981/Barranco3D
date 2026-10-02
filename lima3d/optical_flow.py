"""Streaming frame selection using modal block-matching displacement in pixels."""
import json
import math
from pathlib import Path

# DEFAULTS = dict(mode_threshold_px=20., mode_bin_width_px=1., block_size=180,
#                 block_stride=270, search_radius=100, min_correlation=.7,
#                 min_texture_std=5., keep_last=True)


DEFAULTS = dict(mode_threshold_px=20., mode_bin_width_px=1., block_size=180,
                block_stride=270, search_radius=300, min_correlation=.7,
                min_texture_std=5., keep_last=True)

def validate_options(options=None):
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - set(DEFAULTS):
        raise ValueError('Invalid extraction.optical_flow settings')
    result = {**DEFAULTS, **options}
    for key in ('mode_threshold_px', 'mode_bin_width_px', 'min_correlation', 'min_texture_std'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
            raise ValueError(f'optical_flow.{key} must be finite')
    for key in ('block_size', 'block_stride', 'search_radius'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'optical_flow.{key} must be a positive integer')
    if result['block_size'] < 2:
        raise ValueError('optical_flow.block_size must be at least 2')
    if result['mode_threshold_px'] <= 0 or result['mode_threshold_px'] > result['search_radius']:
        raise ValueError('mode_threshold_px must be positive and no larger than search_radius')
    if result['mode_bin_width_px'] <= 0 or result['min_texture_std'] < 0:
        raise ValueError('mode_bin_width_px must be positive; min_texture_std must be nonnegative')
    if not 0 <= result['min_correlation'] <= 1 or type(result['keep_last']) is not bool:
        raise ValueError('min_correlation must be in [0,1]; keep_last must be boolean')
    return result


def magnitude_statistics(magnitudes, bin_width):
    import numpy as np
    values = np.asarray(magnitudes, dtype=float)
    if not len(values):
        return dict(mean_px=None, median_px=None, mode_px=None, valid_blocks=0)
    bins = np.floor(values / bin_width).astype(np.int64)
    labels, counts = np.unique(bins, return_counts=True)
    winner = labels[np.argmax(counts)]  # Ties choose the lowest-magnitude bin.
    # Use actual observations in the winning bin, not a bin midpoint (zero stays zero).
    return dict(mean_px=float(values.mean()), median_px=float(np.median(values)),
                mode_px=float(np.median(values[bins == winner])), valid_blocks=len(values))


def block_motion(reference, current, options, mask=None, return_blocks=False):
    import cv2
    import numpy as np
    block, stride, radius = (options[k] for k in ('block_size','block_stride','search_radius'))
    height, width = reference.shape
    if current.shape != reference.shape or min(height,width) < block:
        raise ValueError('Frames must have matching dimensions at least as large as block_size')
    integral = cv2.integral((mask > 0).astype(np.uint8)) if mask is not None else None
    magnitudes = []
    blocks = []
    for y in range(0,height-block+1,stride):
        for x in range(0,width-block+1,stride):
            detail = dict(x=x, y=y, width=block, height=block, status='low_texture',
                          dx=None, dy=None, magnitude_px=None, correlation=None, at_search_limit=False)
            if return_blocks:
                blocks.append(detail)
            template = reference[y:y+block,x:x+block]
            if float(template.std()) <= options['min_texture_std']:
                continue
            if mask is not None and not (mask[y:y+block,x:x+block] > 0).all():
                detail['status'] = 'masked_reference'
                continue
            left, top = max(0,x-radius), max(0,y-radius)
            right, bottom = min(width,x+block+radius), min(height,y+block+radius)
            response = cv2.matchTemplate(current[top:bottom,left:right], template, cv2.TM_CCOEFF_NORMED)
            if integral is not None:
                rows, cols = response.shape
                counts = (integral[top+block:top+block+rows,left+block:left+block+cols]
                          -integral[top:top+rows,left+block:left+block+cols]
                          -integral[top+block:top+block+rows,left:left+cols]
                          +integral[top:top+rows,left:left+cols])
                response[counts != block*block] = -1
            _, score, _, location = cv2.minMaxLoc(response)
            detail.update(status='low_correlation', correlation=float(score) if math.isfinite(score) else None)
            if not math.isfinite(score) or score < options['min_correlation']:
                continue
            dx,dy=left+location[0]-x,top+location[1]-y
            magnitude = math.hypot(dx,dy)
            magnitudes.append(magnitude)
            detail.update(status='valid', dx=dx, dy=dy, magnitude_px=magnitude,
                          at_search_limit=abs(dx) == radius or abs(dy) == radius)
    result = magnitude_statistics(magnitudes, options['mode_bin_width_px'])
    if return_blocks:
        result['blocks'] = blocks
    return result


class VideoClock:
    """Elapsed video time; nominal frame duration is a fallback for invalid timestamps."""
    def __init__(self, source_fps):
        self.step = 1 / source_fps if math.isfinite(source_fps) and source_fps > 0 else None
        self.previous = None
        self.elapsed = 0.0
        self.started = False
        self.fallback_frames = 0

    def update(self, timestamp):
        valid = timestamp is not None and math.isfinite(timestamp) and timestamp >= 0
        if not self.started:
            self.started = True
        elif valid and self.previous is not None and timestamp > self.previous:
            self.elapsed += timestamp - self.previous
        elif self.step is not None:
            self.elapsed += self.step
            self.fallback_frames += 1
        else:
            raise ValueError('Hybrid selection needs increasing video timestamps or valid source FPS')
        self.previous = timestamp if valid else None
        return self.elapsed


def select_video(video, folder, prefix, image_format, options, threads, region, log_path, fallback_fps=None):
    """Decode in order, retain only the current frame and selected reference in RAM."""
    import cv2
    options = validate_options(options)
    if fallback_fps is not None and (isinstance(fallback_fps, bool) or
            not isinstance(fallback_fps, (int, float)) or not math.isfinite(fallback_fps) or fallback_fps <= 0):
        raise ValueError('fallback_fps must be finite and positive')
    interval = 1 / fallback_fps if fallback_fps is not None else None
    last_selection_time = None
    selection_time = None
    fallback_selections = 0
    clock = None
    label = 'Hybrid' if interval is not None else 'Optical flow'
    selection_counts = dict(first=0, threshold=0, fps_fallback=0, last=0)
    old_threads = cv2.getNumThreads()
    cap = cv2.VideoCapture()
    reference=None; reference_index=None; selected=0; last_saved=-1; index=-1; frame=None
    mask=None; timestamp=None; measured=None; invalid=0
    if region.get('mode') == 'mask':
        if 'path' not in region:
            raise ValueError('Optical flow requires a shared camera mask path, not per-image mask root')
        mask=cv2.imread(region['path'],cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Cannot read mask: {region["path"]}')
    log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
    params=[cv2.IMWRITE_JPEG_QUALITY,95] if image_format=='jpg' else [cv2.IMWRITE_PNG_COMPRESSION,1]
    def save(frame, reason):
        nonlocal selected,last_saved
        target=Path(folder)/f'{prefix}__{selected:06d}.{image_format}'
        if target.exists():
            raise ValueError(f'Refusing to overwrite {target}')
        if not cv2.imwrite(str(target),frame,params):
            raise RuntimeError(f'Could not write {target}')
        selected+=1;last_saved=index
        selection_counts[reason] += 1
        time_text = f'{timestamp:.3f}s' if timestamp is not None else 'N/A'
        mode = measured['mode_px'] if measured else None
        mode_text = f'{mode:.2f}px' if mode is not None else 'N/A'
        print(f'  [{label}] selected #{selected} | source_frame={index} | time={time_text} | '
              f'reason={reason} | mode={mode_text} | image={target.name}', flush=True)
        return target.name

    def report_progress(complete=False):
        status = 'complete' if complete else 'progress'
        print(f'  [{label}] {status}: {index+1} checked | {selected} selected | '
              f'flow={selection_counts["threshold"]} | fps_fallback={selection_counts["fps_fallback"]} | '
              f'first={selection_counts["first"]} | last={selection_counts["last"]}', flush=True)
    try:
        cv2.setNumThreads(threads)
        opened = cap.open(str(video),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,threads]) if hasattr(cv2,'CAP_PROP_N_THREADS') else cap.open(str(video))
        if not opened:
            raise ValueError(f'Cannot open video: {video}')
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
        if interval is not None:
            clock = VideoClock(cap.get(cv2.CAP_PROP_FPS))
        with log_path.open('w') as log:
            while True:
                ok,current=cap.read()
                if not ok:break
                index+=1
                value=cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp=value/1000 if math.isfinite(value) and value>=0 else None
                if clock is not None:
                    selection_time = clock.update(timestamp)
                if region.get('mode')=='crop':
                    l,t,r,b=region['box'];current=current[t:b,l:r]
                frame=current
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                if min(gray.shape)<options['block_size']:
                    raise ValueError('block_size exceeds the video/crop dimensions')
                if mask is not None and mask.shape!=gray.shape:
                    raise ValueError('Mask and video dimensions differ')
                measured=(block_motion(reference,gray,options,mask) if reference is not None else
                          dict(mean_px=None,median_px=None,mode_px=None,valid_blocks=0))
                threshold_reached = measured['mode_px'] is not None and measured['mode_px'] >= options['mode_threshold_px']
                timeout = (interval is not None and last_selection_time is not None and
                           selection_time - last_selection_time >= interval - 1e-9)
                choose = reference is None or threshold_reached or timeout
                if reference is None:
                    reason = 'first'
                elif threshold_reached:
                    reason = 'threshold'
                elif timeout:
                    reason = 'fps_fallback'
                    fallback_selections += 1
                else:
                    reason = 'below_threshold' if measured['valid_blocks'] else 'no_valid_blocks'
                if reference is not None and not measured['valid_blocks']:invalid+=1
                event=dict(source_frame_index=index,timestamp_seconds=timestamp,reference_frame_index=reference_index,
                           **measured,selected=choose,reason=reason)
                if clock is not None:
                    event['selection_time_seconds'] = selection_time
                if choose:
                    event['image']=save(frame, reason);reference=gray;reference_index=index
                    last_selection_time = selection_time
                log.write(json.dumps(event,allow_nan=False)+'\n')
                if choose or (index + 1) % 100 == 0:
                    log.flush()
                if (index + 1) % 100 == 0:
                    report_progress()
            if frame is None:
                raise ValueError('Video contains no decodable frames')
            if options['keep_last'] and last_saved!=index:
                log.write(json.dumps(dict(source_frame_index=index,timestamp_seconds=timestamp,
                    reference_frame_index=reference_index,selected=True,reason='last',image=save(frame, 'last'),**measured),allow_nan=False)+'\n')
        report_progress(complete=True)
    finally:
        cap.release();cv2.setNumThreads(old_threads)
    return dict(decoded_frames=index+1,selected_frames=selected,frames_without_valid_flow=invalid,
                selection_log=str(log_path),reference='last_selected',units='pixels',
                fallback_fps=fallback_fps, fallback_selected_frames=fallback_selections,
                selection_counts=selection_counts,
                timestamp_fallback_frames=clock.fallback_frames if clock else 0)
