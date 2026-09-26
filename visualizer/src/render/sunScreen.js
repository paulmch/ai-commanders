import * as THREE from 'three';

const _v = new THREE.Vector3();

/**
 * Where an infinitely distant light (unit world direction `dir`) lands on
 * screen for a perspective camera whose matrices are current.
 *
 * Writes into `out`: uv (0..1, may lie outside when off-screen), onScreen
 * (1 inside the frame, fading to 0 just past the edge and when behind the
 * camera) and radiusUv (the disc's angular radius in uv units, x and y).
 */
export function sunScreenInfo(camera, dir, angularRadius, out) {
  _v.copy(dir).transformDirection(camera.matrixWorldInverse);
  if (_v.z > -1e-4) {
    out.uv.set(-1, -1);
    out.onScreen = 0;
    out.radiusUv.set(0, 0);
    return out;
  }
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) / (camera.zoom || 1);
  const tanH = tanV * camera.aspect;
  const ndcX = (_v.x / -_v.z) / tanH;
  const ndcY = (_v.y / -_v.z) / tanV;
  out.uv.set(ndcX * 0.5 + 0.5, ndcY * 0.5 + 0.5);
  const edge = Math.max(Math.abs(ndcX), Math.abs(ndcY));
  out.onScreen = 1 - THREE.MathUtils.smoothstep(edge, 0.98, 1.2);
  const r = (Math.tan(angularRadius) / tanV) * 0.5;
  out.radiusUv.set(r / camera.aspect, r);
  return out;
}
