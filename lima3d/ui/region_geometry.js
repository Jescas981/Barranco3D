/* Geometry in original image pixels, independent of the displayed canvas size. */
(function (root) {
  'use strict';
  const clamp = (value, low, high) => Math.max(low, Math.min(high, Math.round(value)));
  const valid = box => !!box && box[0] < box[2] && box[1] < box[3];
  function rectangle(a, b) {
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  }
  function polygon(box) {
    return [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]];
  }
  function handles(box) {
    const [l,t,r,b] = box, x = (l+r)/2, y = (t+b)/2;
    return {nw:[l,t], ne:[r,t], se:[r,b], sw:[l,b], n:[x,t], e:[r,y], s:[x,b], w:[l,y]};
  }
  function hitTest(box, point, tolerance) {
    if (!valid(box)) return 'new';
    for (const [key, p] of Object.entries(handles(box))) {
      if (Math.abs(point[0]-p[0]) <= tolerance && Math.abs(point[1]-p[1]) <= tolerance) return key;
    }
    return point[0] >= box[0] && point[0] <= box[2] && point[1] >= box[1] && point[1] <= box[3] ? 'move' : 'new';
  }
  function drag(box, handle, start, point, width, height) {
    const p = [clamp(point[0],0,width), clamp(point[1],0,height)];
    if (handle === 'new') return rectangle(start, p);
    const result = [...box];
    if (handle === 'move') {
      const dx = clamp(p[0]-start[0], -box[0], width-box[2]);
      const dy = clamp(p[1]-start[1], -box[1], height-box[3]);
      return [box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy];
    }
    if (handle.includes('w')) result[0] = clamp(p[0],0,box[2]-1);
    if (handle.includes('e')) result[2] = clamp(p[0],box[0]+1,width);
    if (handle.includes('n')) result[1] = clamp(p[1],0,box[3]-1);
    if (handle.includes('s')) result[3] = clamp(p[1],box[1]+1,height);
    return result;
  }
  function edit(box, field, value, width, height) {
    if (!Number.isInteger(value)) throw new Error('Use whole pixels.');
    const result = valid(box) ? [...box] : [0,0,width,height];
    const [l,t,r,b] = result;
    switch (field) {
      case 'left': result[0] = clamp(value,0,r-1); break;
      case 'top': result[1] = clamp(value,0,b-1); break;
      case 'right': result[2] = clamp(value,l+1,width); break;
      case 'bottom': result[3] = clamp(value,t+1,height); break;
      case 'width': result[2] = l+clamp(value,1,width-l); break;
      case 'height': result[3] = t+clamp(value,1,height-t); break;
      default: throw new Error('Unknown crop field');
    }
    return result;
  }
  const api = {clamp, valid, rectangle, polygon, handles, hitTest, drag, edit};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.RegionGeometry = api;
})(globalThis);
