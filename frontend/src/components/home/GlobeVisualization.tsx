"use client";

import React, { useRef, useEffect, useState } from "react";
import Globe from "react-globe.gl";
import type { GlobeMethods } from "react-globe.gl";

interface ArgoDevice {
  lat: number;
  lng: number;
  altitude: number;
}

interface ArcData {
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
}

interface SatelliteData {
  lat: number;
  lng: number;
  altitude: number;
}

const GlobeVisualization: React.FC = () => {
  const globeRef = useRef<GlobeMethods | undefined>(undefined);
  const [argoDevices, setArgoDevices] = useState<ArgoDevice[]>([]);
  const [arcsData, setArcsData] = useState<ArcData[]>([]);
  const [satelliteArcsData, setSatelliteArcsData] = useState<ArcData[]>([]);
  const [landPolygons, setLandPolygons] = useState<any[]>([]);
  const [satellite, setSatellite] = useState<SatelliteData>({ lat: 0, lng: 0, altitude: 0.5 });
  const animationRef = useRef<number | null>(null);

  // Predefined ocean coordinates for Argo floats
  const oceanCoordinates = [
    // Atlantic Ocean
    { lat: 35.0, lng: -40.0 }, { lat: 25.0, lng: -50.0 }, { lat: 15.0, lng: -35.0 },
    { lat: 5.0, lng: -25.0 }, { lat: -10.0, lng: -30.0 }, { lat: -25.0, lng: -15.0 },
    { lat: -35.0, lng: -20.0 }, { lat: 45.0, lng: -30.0 }, { lat: 55.0, lng: -25.0 },

    // Pacific Ocean
    { lat: 35.0, lng: -150.0 }, { lat: 25.0, lng: -160.0 }, { lat: 15.0, lng: -140.0 },
    { lat: 5.0, lng: -170.0 }, { lat: -10.0, lng: -150.0 }, { lat: -25.0, lng: -160.0 },
    { lat: -35.0, lng: -140.0 }, { lat: 45.0, lng: -135.0 }, { lat: 0.0, lng: 170.0 },
    { lat: 20.0, lng: 160.0 }, { lat: -20.0, lng: 175.0 }, { lat: 10.0, lng: -120.0 },

    // Indian Ocean
    { lat: 10.0, lng: 75.0 }, { lat: 0.0, lng: 85.0 }, { lat: -10.0, lng: 80.0 },
    { lat: -20.0, lng: 90.0 }, { lat: -30.0, lng: 85.0 }, { lat: -40.0, lng: 75.0 },
    { lat: 20.0, lng: 65.0 }, { lat: -15.0, lng: 105.0 }, { lat: -25.0, lng: 110.0 },

    // Southern Ocean
    { lat: -55.0, lng: 0.0 }, { lat: -60.0, lng: 45.0 }, { lat: -58.0, lng: 90.0 },
    { lat: -62.0, lng: 135.0 }, { lat: -57.0, lng: 180.0 }, { lat: -59.0, lng: -135.0 },
    { lat: -61.0, lng: -90.0 }, { lat: -56.0, lng: -45.0 },

    // Arctic Ocean
    { lat: 75.0, lng: 0.0 }, { lat: 78.0, lng: 45.0 }, { lat: 80.0, lng: 90.0 },
    { lat: 77.0, lng: 135.0 }, { lat: 79.0, lng: 180.0 }, { lat: 76.0, lng: -135.0 },
    { lat: 81.0, lng: -90.0 }, { lat: 74.0, lng: -45.0 },

    // Additional scattered ocean points
    { lat: 40.0, lng: 10.0 }, { lat: -15.0, lng: 25.0 }, { lat: 30.0, lng: 120.0 },
    { lat: -45.0, lng: 165.0 }, { lat: 12.0, lng: -80.0 }, { lat: -8.0, lng: -35.0 },
    { lat: 28.0, lng: -65.0 }, { lat: -32.0, lng: 25.0 }, { lat: 18.0, lng: 155.0 },
    { lat: -18.0, lng: -125.0 }, { lat: 42.0, lng: -145.0 }, { lat: -42.0, lng: 145.0 },
    { lat: 8.0, lng: 95.0 }, { lat: -28.0, lng: 115.0 }, { lat: 22.0, lng: -45.0 },
    { lat: -12.0, lng: 55.0 }, { lat: 38.0, lng: -155.0 }, { lat: -38.0, lng: 175.0 },
    { lat: 14.0, lng: -110.0 }, { lat: -14.0, lng: 70.0 }, { lat: 26.0, lng: -75.0 },
    { lat: -26.0, lng: 95.0 }, { lat: 48.0, lng: -140.0 }, { lat: -48.0, lng: 140.0 },
    { lat: 6.0, lng: 45.0 }, { lat: -6.0, lng: -155.0 }, { lat: 32.0, lng: -125.0 },
    { lat: -32.0, lng: 125.0 }, { lat: 16.0, lng: 85.0 }, { lat: -16.0, lng: -85.0 },
    { lat: 44.0, lng: -165.0 }, { lat: -44.0, lng: 165.0 }, { lat: 2.0, lng: 15.0 },
    { lat: -2.0, lng: -175.0 }, { lat: 36.0, lng: -115.0 }, { lat: -36.0, lng: 115.0 },
    { lat: 24.0, lng: 105.0 }, { lat: -24.0, lng: -105.0 }, { lat: 52.0, lng: -175.0 },
    { lat: -52.0, lng: 5.0 }, { lat: 4.0, lng: 125.0 }, { lat: -4.0, lng: -125.0 },
    { lat: 34.0, lng: -85.0 }, { lat: -34.0, lng: 85.0 }, { lat: 46.0, lng: -25.0 },
    { lat: -46.0, lng: 25.0 }, { lat: 19.0, lng: 35.0 }, { lat: -19.0, lng: -35.0 }
  ];

  // Load land polygons for borders
  useEffect(() => {
    fetch("https://unpkg.com/world-atlas@2/countries-110m.json")
      .then((res) => res.json())
      .then((worldData) => {
        // @ts-ignore
        import("topojson-client").then((topojson) => {
          const geoJson = topojson.feature(worldData, worldData.objects.countries).features;
          setLandPolygons(geoJson);
        });
      });
  }, []);

  // Generate floats using predefined ocean coordinates
  useEffect(() => {
    const floats: ArgoDevice[] = [];

    // Select 80 random ocean coordinates and add slight variations
    for (let i = 0; i < 80; i++) {
      const baseCoord = oceanCoordinates[Math.floor(Math.random() * oceanCoordinates.length)];
      const lat = baseCoord.lat + (Math.random() - 0.5) * 10; // ±5 degree variation
      const lng = baseCoord.lng + (Math.random() - 0.5) * 20; // ±10 degree variation

      floats.push({
        lat: Math.max(-85, Math.min(85, lat)), // Keep within valid range
        lng: lng,
        altitude: 0.01 + Math.random() * 0.02
      });
    }

    setArgoDevices(floats);

    // Create arcs between floats
    const arcs: ArcData[] = [];
    for (let i = 0; i < 40; i++) {
      const source = floats[Math.floor(Math.random() * floats.length)];
      const target = floats[Math.floor(Math.random() * floats.length)];
      if (source !== target) {
        arcs.push({
          startLat: source.lat,
          startLng: source.lng,
          endLat: target.lat,
          endLng: target.lng,
        });
      }
    }
    setArcsData(arcs);
  }, []);

  // Animate satellite orbit and create satellite-to-float connections
  useEffect(() => {
    let angle = 0;

    const animate = () => {
      angle += 0.01; // Satellite orbit speed

      // Satellite orbital position
      const satLat = Math.sin(angle * 0.3) * 30; // Varies between -30 and +30 degrees
      const satLng = angle * 10; // Continuous rotation

      setSatellite({
        lat: satLat,
        lng: satLng,
        altitude: 0.4 + Math.sin(angle * 0.5) * 0.1 // Varying altitude
      });

      // Create arcs from satellite to random Argo floats
      if (argoDevices.length > 0) {
        const satelliteArcs: ArcData[] = [];
        for (let i = 0; i < 8; i++) { // Connect to 8 random floats
          const randomFloat = argoDevices[Math.floor(Math.random() * argoDevices.length)];
          satelliteArcs.push({
            startLat: satLat,
            startLng: satLng,
            endLat: randomFloat.lat,
            endLng: randomFloat.lng,
          });
        }
        setSatelliteArcsData(satelliteArcs);
      }

      // Rotate the globe
      if (globeRef.current) {
        const globe = globeRef.current;
        const currentRotation = globe.scene().rotation;
        currentRotation.y += 0.002; // Earth rotation speed
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [argoDevices]);

  useEffect(() => {
    if (globeRef.current) {
      globeRef.current.pointOfView({ lat: 0, lng: 0, altitude: 2 }, 2000);
    }
  }, []);

  return (
    <div className="relative">
      <Globe
        ref={globeRef}
        globeImageUrl="//unpkg.com/three-globe/example/img/earth-dark.jpg"
        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"

        // Land borders
        polygonsData={landPolygons}
        polygonCapColor={() => "rgba(0,0,0,0)"}
        polygonSideColor={() => "rgba(0,100,255,0.3)"}
        polygonStrokeColor={() => "rgba(0,200,255,0.8)"}
        polygonAltitude={0.003}

        // Argo floats
        pointsData={[...argoDevices, satellite]}
        pointLat={(d: any) => d.lat}
        pointLng={(d: any) => d.lng}
        pointAltitude={(d: any) => d.altitude}
        pointColor={(d: any) => d === satellite ? "red" : "orange"}
        pointRadius={(d: any) => d === satellite ? 0.4 : 0.2}

        // Float-to-float connections
        arcsData={arcsData}
        arcColor={() => ["cyan", "deepskyblue", "white"]}
        arcAltitude={() => 0.15 + Math.random() * 0.1}
        arcStroke={() => Math.random() * 0.8 + 0.3}
        arcDashLength={0.4}
        arcDashGap={0.2}
        arcDashAnimateTime={() => 2000 + Math.random() * 2000}

        // Satellite-to-float connections
        ringsData={satelliteArcsData}
        ringColor={() => ["yellow", "gold", "orange"]}
        ringMaxRadius={() => 0.8}
        ringPropagationSpeed={() => 2}
        ringRepeatPeriod={() => 1000}

        width={900}
        height={900}
        backgroundColor="rgba(0,0,0,0)"
        enablePointerInteraction={true}

        // Restrict zooming by setting minZoom and maxZoom to the same value
        minZoom={2}
        maxZoom={2}
      />
      <div className="mt-8 flex flex-wrap gap-4 justify-center text-sm">
      <a
        href="#about-argo"
        className="flex items-center gap-2 px-4 py-2 rounded-xl border border-cyan-300/30 text-cyan-200/80 
                   hover:text-cyan-100 hover:border-cyan-300 transition-all duration-300
                   backdrop-blur-md bg-white/5 shadow-md shadow-cyan-500/10"
      >
        <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
        Know about Argo
      </a>

      <a
        href="#how-it-works"
        className="flex items-center gap-2 px-4 py-2 rounded-xl border border-cyan-300/30 text-cyan-200/80 
                   hover:text-cyan-100 hover:border-cyan-300 transition-all duration-300
                   backdrop-blur-md bg-white/5 shadow-md shadow-cyan-500/10"
      >
        <span className="w-2 h-2 rounded-full bg-green-400"></span>
        How Argo Works
      </a>

      <a
        href="#data-storage"
        className="flex items-center gap-2 px-4 py-2 rounded-xl border border-cyan-300/30 text-cyan-200/80 
                   hover:text-cyan-100 hover:border-cyan-300 transition-all duration-300
                   backdrop-blur-md bg-white/5 shadow-md shadow-cyan-500/10"
      >
        <span className="w-2 h-2 rounded-full bg-purple-400"></span>
        How Argo Stores Data
      </a>
    </div>
    </div>
  );
};

export default GlobeVisualization;