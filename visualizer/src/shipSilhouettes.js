/**
 * Ship silhouette SVG definitions for each ship class
 * Each ship has nose, lateral, tail armor sections plus hardpoint indicators
 * ViewBox is 120x80, ships face right
 */

export const SHIP_SILHOUETTES = {
  // Corvette - smallest, sleekest, needle-like
  corvette: {
    body: "M105,40 L85,30 L25,32 L15,40 L25,48 L85,50 Z",
    nose: "M105,40 L85,30 L85,50 Z",
    lateral: "M85,30 L25,32 L25,48 L85,50 Z",
    tail: "M25,32 L15,40 L25,48 Z",
    hardpoints: [
      { x: 70, y: 40, size: 4 },  // Main gun
      { x: 45, y: 35, size: 3 },  // PD
    ],
    engineGlow: { x: 10, y: 40, width: 8, height: 12 }
  },

  // Frigate - small but slightly wider
  frigate: {
    body: "M102,40 L82,26 L30,24 L12,40 L30,56 L82,54 Z",
    nose: "M102,40 L82,26 L82,54 Z",
    lateral: "M82,26 L30,24 L30,56 L82,54 Z",
    tail: "M30,24 L12,40 L30,56 Z",
    hardpoints: [
      { x: 68, y: 40, size: 4 },  // Main gun
      { x: 50, y: 32, size: 3 },  // Turret
      { x: 50, y: 48, size: 3 },  // PD
    ],
    engineGlow: { x: 6, y: 40, width: 10, height: 16 }
  },

  // Destroyer - medium, balanced profile (current default)
  destroyer: {
    body: "M100,40 L85,20 L35,15 L15,40 L35,65 L85,60 Z",
    nose: "M100,40 L85,20 L85,60 Z",
    lateral: "M85,20 L35,15 L35,65 L85,60 Z",
    tail: "M35,15 L15,40 L35,65 Z",
    hardpoints: [
      { x: 72, y: 40, size: 5 },  // Spinal mount
      { x: 55, y: 28, size: 4 },  // Turret 1
      { x: 55, y: 52, size: 4 },  // Turret 2
      { x: 42, y: 35, size: 3 },  // PD 1
      { x: 42, y: 45, size: 3 },  // PD 2
    ],
    engineGlow: { x: 8, y: 40, width: 12, height: 22 }
  },

  // Cruiser - wider, more substantial beam
  cruiser: {
    body: "M98,40 L82,16 L32,12 L10,40 L32,68 L82,64 Z",
    nose: "M98,40 L82,16 L82,64 Z",
    lateral: "M82,16 L32,12 L32,68 L82,64 Z",
    tail: "M32,12 L10,40 L32,68 Z",
    hardpoints: [
      { x: 70, y: 40, size: 6 },  // Heavy spinal
      { x: 58, y: 24, size: 4 },  // Turret 1
      { x: 58, y: 56, size: 4 },  // Turret 2
      { x: 45, y: 30, size: 4 },  // Turret 3
      { x: 45, y: 50, size: 4 },  // Turret 4
      { x: 38, y: 40, size: 3 },  // PD
    ],
    engineGlow: { x: 4, y: 40, width: 12, height: 28 }
  },

  // Battlecruiser - long, lean, powerful forward section
  battlecruiser: {
    body: "M105,40 L88,18 L28,16 L8,40 L28,64 L88,62 Z",
    nose: "M105,40 L88,18 L88,62 Z",
    lateral: "M88,18 L28,16 L28,64 L88,62 Z",
    tail: "M28,16 L8,40 L28,64 Z",
    hardpoints: [
      { x: 78, y: 40, size: 6 },  // Heavy spinal
      { x: 65, y: 26, size: 5 },  // Main turret 1
      { x: 65, y: 54, size: 5 },  // Main turret 2
      { x: 50, y: 24, size: 4 },  // Secondary 1
      { x: 50, y: 56, size: 4 },  // Secondary 2
      { x: 38, y: 32, size: 3 },  // PD 1
      { x: 38, y: 48, size: 3 },  // PD 2
    ],
    engineGlow: { x: 2, y: 40, width: 12, height: 26 }
  },

  // Battleship - bulky, heavily armed, wide beam
  battleship: {
    body: "M95,40 L78,12 L28,8 L6,40 L28,72 L78,68 Z",
    nose: "M95,40 L78,12 L78,68 Z",
    lateral: "M78,12 L28,8 L28,72 L78,68 Z",
    tail: "M28,8 L6,40 L28,72 Z",
    hardpoints: [
      { x: 68, y: 40, size: 7 },  // Massive spinal
      { x: 58, y: 20, size: 5 },  // Heavy turret 1
      { x: 58, y: 60, size: 5 },  // Heavy turret 2
      { x: 48, y: 28, size: 5 },  // Turret 3
      { x: 48, y: 52, size: 5 },  // Turret 4
      { x: 38, y: 18, size: 4 },  // Secondary 1
      { x: 38, y: 62, size: 4 },  // Secondary 2
      { x: 32, y: 35, size: 3 },  // PD 1
      { x: 32, y: 45, size: 3 },  // PD 2
    ],
    engineGlow: { x: 0, y: 40, width: 12, height: 34 }
  },

  // Dreadnought - massive, fortress-like, intimidating
  dreadnought: {
    body: "M92,40 L75,8 L25,5 L4,40 L25,75 L75,72 Z",
    nose: "M92,40 L75,8 L75,72 Z",
    lateral: "M75,8 L25,5 L25,75 L75,72 Z",
    tail: "M25,5 L4,40 L25,75 Z",
    hardpoints: [
      { x: 65, y: 40, size: 8 },  // Devastating spinal
      { x: 55, y: 16, size: 6 },  // Heavy battery 1
      { x: 55, y: 64, size: 6 },  // Heavy battery 2
      { x: 48, y: 28, size: 5 },  // Main turret 1
      { x: 48, y: 52, size: 5 },  // Main turret 2
      { x: 40, y: 16, size: 5 },  // Broadside 1
      { x: 40, y: 64, size: 5 },  // Broadside 2
      { x: 35, y: 35, size: 4 },  // Secondary 1
      { x: 35, y: 45, size: 4 },  // Secondary 2
      { x: 28, y: 25, size: 3 },  // PD 1
      { x: 28, y: 55, size: 3 },  // PD 2
    ],
    engineGlow: { x: 0, y: 40, width: 10, height: 38 }
  }
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

    <!-- Hardpoints -->
    ${ship.hardpoints.map((hp, i) => `
      <rect class="hardpoint"
            x="${hp.x - hp.size/2}" y="${hp.y - hp.size/2}"
            width="${hp.size}" height="${hp.size}"
            rx="1" />
    `).join('')}

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
  // Get the nose tip from the nose polygon
  const nosePoints = pathToPoints(ship.nose).split(' ');
  const [tipX, tipY] = nosePoints[0].split(',').map(Number);

  // Arrow pointing right from nose
  return `${tipX + 8},${tipY} ${tipX},${tipY - 4} ${tipX},${tipY + 4}`;
}

/**
 * Map ship variants to their base type for silhouette lookup
 */
function getBaseShipType(shipType) {
  const variantMap = {
    dreadnought_siege: 'dreadnought',
    // Add more variants here as needed
  };
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
    dreadnought_siege: 'DREADNOUGHT (SIEGE)'
  };
  return names[shipType] || 'UNKNOWN';
}
