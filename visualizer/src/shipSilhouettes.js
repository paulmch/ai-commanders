/**
 * Ship silhouette SVG definitions for each ship class
 * Each ship has nose, lateral, tail armor sections plus hardpoint indicators
 * ViewBox is 120x80, ships face right
 */

export const SHIP_SILHOUETTES = {
  // Side profiles (port side, nose right, dorsal up) echoing the 3D hulls
  // (render/Hulls.js). Hardpoints follow data/fleet_ships.json: kind is
  // spinal | battery | heavy | pd | launcher; `far` marks a starboard mount
  // seen through the hull. Squares take `size`, bars take w x h.

  // Corvette - torpedo boat: hex hull, stepped needle bow, rails, bare torch
  corvette: {
    body: "M106,38.5 L106,41.5 L102,43.5 L101,44.5 L97,46 L96,47.5 L88,48 L32,48 L32,47 L24,47 L24,51 L20,51 L20,29 L24,29 L24,33 L32,33 L32,32 L42,32 L42,29 L80,29 L80,32 L88,32 L96,32.5 L97,34 L101,35.5 L102,36.5 Z",
    nose: "M106,38.5 L106,41.5 L102,43.5 L101,44.5 L97,46 L96,47.5 L88,48 L88,32 L96,32.5 L97,34 L101,35.5 L102,36.5 Z",
    lateral: "M88,32 L80,32 L80,29 L42,29 L42,32 L32,32 L32,48 L88,48 Z",
    tail: "M32,33 L24,33 L24,29 L20,29 L20,51 L24,51 L24,47 L32,47 Z",
    hardpoints: [
      { x: 76, y: 40.5, w: 15, h: 4, kind: 'launcher' },   // 8-round rails
      { x: 50, y: 27.5, size: 3, kind: 'pd' },             // dorsal PD aft
    ],
    engineGlow: { x: 15, y: 40, width: 8, height: 14 }
  },

  // Frigate - escort: gun head with sensor dome on a truss neck, drum body
  frigate: {
    body: "M104,36 L106,38 L106,42 L104,44 L100,46 L100,47 L84,47 L84,43.5 L76,43.5 L76,46 L36,46 L36,44.5 L28,44.5 L28,49 L24,49 L24,31 L28,31 L28,35.5 L36,35.5 L36,34 L76,34 L76,36.5 L84,36.5 L84,33 L100,33 L100,34 Z",
    nose: "M104,36 L106,38 L106,42 L104,44 L100,46 L100,47 L84,47 L84,33 L100,33 L100,34 Z",
    lateral: "M84,36.5 L76,36.5 L76,34 L36,34 L36,46 L76,46 L76,43.5 L84,43.5 Z",
    tail: "M36,35.5 L28,35.5 L28,31 L24,31 L24,49 L28,49 L28,44.5 L36,44.5 Z",
    hardpoints: [
      { x: 92, y: 31, size: 5, kind: 'battery' },          // nose battery
      { x: 49, y: 32.5, size: 3, kind: 'pd' },             // dorsal PD aft
      { x: 49, y: 47.5, size: 3, kind: 'pd' },             // ventral PD aft
    ],
    engineGlow: { x: 19, y: 40, width: 8, height: 14 }
  },

  // Destroyer - box sections, chisel bow, spinal along the keel
  destroyer: {
    body: "M105,38.4 L105,41.6 L100,43 L100,44.2 L96,45.2 L96,46.4 L92,47 L70,47 L70,48 L34,48 L34,47.5 L22,47.5 L22,32.5 L34,32.5 L34,32 L70,32 L70,33 L92,33 L96,33.6 L96,34.8 L100,35.8 L100,37 Z",
    nose: "M105,38.4 L105,41.6 L100,43 L100,44.2 L96,45.2 L96,46.4 L92,47 L92,33 L96,33.6 L96,34.8 L100,35.8 L100,37 Z",
    lateral: "M92,33 L70,33 L70,32 L34,32 L34,48 L70,48 L70,47 L92,47 Z",
    tail: "M34,32.5 L22,32.5 L22,47.5 L34,47.5 Z",
    hardpoints: [
      { x: 80, y: 49.5, w: 56, h: 2, kind: 'spinal' },     // keel spinal
      { x: 84, y: 31, size: 4.5, kind: 'battery' },        // dorsal battery
      { x: 44, y: 30.5, size: 3, kind: 'pd' },             // dorsal PD aft
      { x: 44, y: 49.5, size: 3, kind: 'pd' },             // ventral PD aft
    ],
    engineGlow: { x: 17, y: 40, width: 9, height: 14 }
  },

  // Cruiser - collar, stepped octagonal bow, ribbed drum, reactor drum aft
  cruiser: {
    body: "M104,37 L104,43 L102,44.5 L98,46 L98,47.5 L94,49 L94,50.5 L90,52 L82,52 L82,50.5 L38,50.5 L38,48.5 L26,48.5 L26,51 L14,50 L14,30 L26,29 L26,31.5 L38,31.5 L38,29.5 L82,29.5 L82,28 L90,28 L94,29.5 L94,31 L98,32.5 L98,34 L102,35.5 Z",
    nose: "M104,37 L104,43 L102,44.5 L98,46 L98,47.5 L94,49 L94,50.5 L90,52 L82,52 L82,28 L90,28 L94,29.5 L94,31 L98,32.5 L98,34 L102,35.5 Z",
    lateral: "M82,29.5 L38,29.5 L38,50.5 L82,50.5 Z",
    tail: "M38,31.5 L26,31.5 L26,29 L14,30 L14,50 L26,51 L26,48.5 L38,48.5 Z",
    hardpoints: [
      { x: 76, y: 53.2, w: 48, h: 2.2, kind: 'spinal' },   // keel spinal
      { x: 75.5, y: 41.5, size: 4.5, kind: 'battery', far: true }, // starboard battery
      { x: 74, y: 40, size: 5, kind: 'battery' },          // port battery
      { x: 52, y: 28, size: 3, kind: 'pd' },               // dorsal PD aft
      { x: 52, y: 52, size: 3, kind: 'pd' },               // ventral PD aft
    ],
    engineGlow: { x: 9, y: 40, width: 10, height: 20 }
  },

  // Torpedo cruiser - cruiser drum ringed by four VLS banks, short shield bow
  cruiser_torpedo: {
    body: "M98,33 L98,47 L94,48.5 L94,50.5 L90,52 L82,52 L82,50.5 L80,50.5 L80,55 L50,55 L50,50.5 L38,50.5 L38,48.5 L26,48.5 L26,51 L14,50 L14,30 L26,29 L26,31.5 L38,31.5 L38,29.5 L50,29.5 L50,25 L80,25 L80,29.5 L82,29.5 L82,28 L90,28 L94,29.5 L94,31.5 Z",
    nose: "M98,33 L98,47 L94,48.5 L94,50.5 L90,52 L82,52 L82,28 L90,28 L94,29.5 L94,31.5 Z",
    lateral: "M82,29.5 L80,29.5 L80,25 L50,25 L50,29.5 L38,29.5 L38,50.5 L50,50.5 L50,55 L80,55 L80,50.5 L82,50.5 Z",
    tail: "M38,31.5 L26,31.5 L26,29 L14,30 L14,50 L26,51 L26,48.5 L38,48.5 Z",
    hardpoints: [
      { x: 65, y: 24.5, w: 26, h: 2.5, kind: 'launcher' }, // dorsal bank (12 cells)
      { x: 65, y: 55.5, w: 26, h: 2.5, kind: 'launcher' }, // ventral bank
      { x: 59.5, y: 41.5, w: 24, h: 8, kind: 'launcher', far: true }, // starboard bank
      { x: 58, y: 40, w: 26, h: 9, kind: 'launcher' },     // port bank
      { x: 86, y: 27, size: 3, kind: 'pd' },               // collar PDs
      { x: 86, y: 53, size: 3, kind: 'pd' },
      { x: 87.5, y: 28.5, size: 2.6, kind: 'pd', far: true },
      { x: 87.5, y: 51.5, size: 2.6, kind: 'pd', far: true },
    ],
    engineGlow: { x: 9, y: 40, width: 10, height: 20 }
  },

  // Battlecruiser - stretched racer: knife bow, open waist, wide engine block
  battlecruiser: {
    body: "M111,37.8 L111,42.2 L104,42.8 L104,43.8 L98,44.4 L98,45.4 L92,46 L62,46 L62,42.5 L50,42.5 L50,47 L30,47 L30,45.5 L18,45.5 L18,34.5 L30,34.5 L30,33 L50,33 L50,37.5 L62,37.5 L62,34 L92,34 L98,34.6 L98,35.6 L104,36.2 L104,37.2 Z",
    nose: "M111,37.8 L111,42.2 L104,42.8 L104,43.8 L98,44.4 L98,45.4 L92,46 L92,34 L98,34.6 L98,35.6 L104,36.2 L104,37.2 Z",
    lateral: "M92,34 L62,34 L62,37.5 L50,37.5 L50,33 L30,33 L30,47 L50,47 L50,42.5 L62,42.5 L62,46 L92,46 Z",
    tail: "M30,34.5 L18,34.5 L18,45.5 L30,45.5 Z",
    hardpoints: [
      { x: 84, y: 47.5, w: 52, h: 2, kind: 'spinal' },     // keel spinal
      { x: 85.5, y: 41.5, size: 4, kind: 'battery', far: true }, // starboard battery
      { x: 84, y: 40, size: 4.5, kind: 'battery' },        // port battery
      { x: 36, y: 31.5, size: 3, kind: 'pd' },             // dorsal PD aft
      { x: 36, y: 48.5, size: 3, kind: 'pd' },             // ventral PD aft
    ],
    engineGlow: { x: 13, y: 40, width: 9, height: 12 }
  },

  // Battleship - broad armoured slab, wide flat chisel, heavy batteries
  battleship: {
    body: "M105,37.8 L105,42.2 L100,44 L100,45 L96,46.5 L96,47.5 L92,49 L84,49 L84,50.5 L34,50.5 L34,49 L26,49 L26,48 L14,48 L14,32 L26,32 L26,31 L34,31 L34,29.5 L84,29.5 L84,31 L92,31 L96,32.5 L96,33.5 L100,35 L100,36 Z",
    nose: "M105,37.8 L105,42.2 L100,44 L100,45 L96,46.5 L96,47.5 L92,49 L92,31 L96,32.5 L96,33.5 L100,35 L100,36 Z",
    lateral: "M92,31 L84,31 L84,29.5 L34,29.5 L34,31 L26,31 L26,49 L34,49 L34,50.5 L84,50.5 L84,49 L92,49 Z",
    tail: "M26,32 L14,32 L14,48 L26,48 Z",
    hardpoints: [
      { x: 80, y: 51.6, w: 42, h: 2, kind: 'spinal' },     // keel spinal
      { x: 78, y: 27.5, size: 5.5, kind: 'heavy' },        // heavy dorsal fwd
      { x: 78, y: 54.5, size: 5.5, kind: 'heavy' },        // heavy ventral fwd
      { x: 59, y: 27.5, size: 5.5, kind: 'heavy' },        // heavy mid dorsal
      { x: 59, y: 54, size: 4.5, kind: 'battery' },        // mid ventral battery
      { x: 44, y: 28, size: 3, kind: 'pd' },               // dorsal PD aft
      { x: 44, y: 52.5, size: 3, kind: 'pd' },             // ventral PD aft
      { x: 73, y: 40, size: 3, kind: 'pd' },               // port PD
    ],
    engineGlow: { x: 9, y: 40, width: 10, height: 16 }
  },

  // Dreadnought - quad-lobe fortress, stepped citadel bow, keel spinal
  dreadnought: {
    body: "M110,33.5 L110,46.5 L108,48 L108,49 L106,50.5 L106,51.5 L104,53 L96,53 L96,57 L84,57 L84,64 L30,64 L30,58 L16,57 L16,23 L30,22 L30,16 L84,16 L84,23 L96,23 L96,27 L104,27 L106,28.5 L106,29.5 L108,31 L108,32 Z",
    nose: "M110,33.5 L110,46.5 L108,48 L108,49 L106,50.5 L106,51.5 L104,53 L96,53 L96,57 L84,57 L84,23 L96,23 L96,27 L104,27 L106,28.5 L106,29.5 L108,31 L108,32 Z",
    lateral: "M84,16 L30,16 L30,64 L84,64 Z",
    tail: "M30,22 L16,23 L16,57 L30,58 Z",
    hardpoints: [
      { x: 97, y: 57.5, w: 24, h: 2.6, kind: 'spinal' },   // keel spinal, muzzle at the bow
      { x: 90, y: 21, size: 6, kind: 'heavy' },            // heavy dorsal fwd
      { x: 88, y: 60.5, size: 6, kind: 'heavy' },          // heavy ventral fwd
      { x: 71.5, y: 29.5, size: 5.5, kind: 'heavy', far: true }, // heavy starboard
      { x: 70, y: 28, size: 6, kind: 'heavy' },            // heavy port
      { x: 58, y: 13.5, size: 6, kind: 'heavy' },          // heavy mid dorsal
      { x: 58, y: 66.5, size: 5, kind: 'battery' },        // mid ventral battery
      { x: 46, y: 13.5, size: 3.2, kind: 'pd' },           // dorsal PD aft
      { x: 46, y: 66.5, size: 3.2, kind: 'pd' },           // ventral PD aft
      { x: 57.5, y: 41.5, size: 3, kind: 'pd', far: true }, // starboard PD
      { x: 56, y: 40, size: 3.2, kind: 'pd' },             // port PD
    ],
    engineGlow: { x: 10, y: 40, width: 10, height: 30 }
  }
};

// Siege dreadnought: the same fortress with the siege coiler's long barrel
SHIP_SILHOUETTES.dreadnought_siege = {
  ...SHIP_SILHOUETTES.dreadnought,
  hardpoints: [
    { x: 102, y: 50, w: 34, h: 3, kind: 'spinal' },        // siege coiler
    ...SHIP_SILHOUETTES.dreadnought.hardpoints.slice(1)
  ]
};

/**
 * Generate SVG elements for a ship silhouette
 * @param {string} shipType - Ship class (corvette, frigate, destroyer, etc.)
 * @returns {string} SVG inner HTML
 */
export function generateShipSVG(shipType) {
  // Handle ship variants (e.g., dreadnought_siege uses dreadnought silhouette)
  const baseType = getBaseShipType(shipType);
  const ship = SHIP_SILHOUETTES[baseType] || SHIP_SILHOUETTES.destroyer;

  let svg = `
    <defs>
      <linearGradient id="armorGradient" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" style="stop-color:var(--armor-color);stop-opacity:0.8" />
        <stop offset="100%" style="stop-color:var(--armor-color);stop-opacity:0.3" />
      </linearGradient>
      <filter id="glow">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    </defs>

    <!-- Ship body outline -->
    <polygon class="ship-body" points="${pathToPoints(ship.body)}" />

    <!-- Armor sections -->
    <polygon id="armorNose" class="armor-face nose" points="${pathToPoints(ship.nose)}" />
    <polygon id="armorLateral" class="armor-face lateral" points="${pathToPoints(ship.lateral)}" />
    <polygon id="armorTail" class="armor-face tail" points="${pathToPoints(ship.tail)}" />

    <!-- Hardpoints (far-side ghosts first, behind their near twins) -->
    ${[...ship.hardpoints.filter(hp => hp.far), ...ship.hardpoints.filter(hp => !hp.far)].map((hp) => {
      const w = hp.w ?? hp.size, h = hp.h ?? hp.size;
      return `
      <rect class="hardpoint ${hp.kind || ''}${hp.far ? ' far' : ''}"
            x="${hp.x - w / 2}" y="${hp.y - h / 2}"
            width="${w}" height="${h}"
            ${hardpointLook(hp, w, h)} />`;
    }).join('')}

    <!-- Engine glow -->
    <ellipse class="engine-glow"
             cx="${ship.engineGlow.x}" cy="${ship.engineGlow.y}"
             rx="${ship.engineGlow.width/2}" ry="${ship.engineGlow.height/2}" />

    <!-- Direction indicator -->
    <polygon class="direction-arrow" points="${getDirectionArrow(ship)}" />
  `;

  return svg;
}

/**
 * Per-kind shape and the see-through look of a starboard (`far`) mount, as
 * SVG presentation attributes: styles.css styles .hardpoint's fill and
 * stroke, and these only set what it leaves alone, so a far mount reads as
 * a faint dashed ghost behind its port twin instead of a doubled square,
 * and a PD mount as a dot. Class names stay on the rect for any CSS.
 */
export function hardpointLook(hp, w, h) {
  const r = Math.min(w, h);
  const rx = hp.kind === 'pd' ? r / 2 : hp.kind === 'battery' ? Math.min(1, r * 0.25)
    : hp.kind === 'heavy' ? Math.min(0.8, r * 0.15) : 0.5;
  let a = `rx="${+rx.toFixed(2)}"`;
  if (hp.far) a += ' fill-opacity="0.3" stroke-opacity="0.55" stroke-dasharray="0.9 0.6"';
  return a;
}

/**
 * Convert SVG path string to points string for polygon
 */
function pathToPoints(pathD) {
  // Path format: "M100,40 L85,20 L35,15 L15,40 L35,65 L85,60 Z"
  // Extract coordinates
  return pathD
    .replace(/[MLZ]/g, '')
    .trim()
    .split(/\s+/)
    .join(' ');
}

/**
 * Generate direction arrow based on ship nose position
 */
function getDirectionArrow(ship) {
  // Nose tip: the forward-most edge of the nose polygon, centred on it
  const pts = pathToPoints(ship.nose).split(' ').map(p => p.split(',').map(Number));
  const tipX = Math.max(...pts.map(p => p[0]));
  const edge = pts.filter(p => p[0] === tipX);
  const tipY = edge.reduce((a, p) => a + p[1], 0) / edge.length;

  // Arrow pointing right from nose
  return `${tipX + 8},${tipY} ${tipX},${tipY - 4} ${tipX},${tipY + 4}`;
}

/**
 * Map ship variants to their base type for silhouette lookup
 */
function getBaseShipType(shipType) {
  // Variants without a profile of their own fall back to their base hull
  const variantMap = {};
  return variantMap[shipType] || shipType;
}

/**
 * Get ship class display name
 */
export function getShipClassName(shipType) {
  const names = {
    corvette: 'CORVETTE',
    frigate: 'FRIGATE',
    destroyer: 'DESTROYER',
    cruiser: 'CRUISER',
    battlecruiser: 'BATTLECRUISER',
    battleship: 'BATTLESHIP',
    dreadnought: 'DREADNOUGHT',
    dreadnought_siege: 'DREADNOUGHT (SIEGE)',
    cruiser_torpedo: 'TORPEDO CRUISER'
  };
  return names[shipType] || 'UNKNOWN';
}
