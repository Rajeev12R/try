"use client";

import React from "react";

import Navbar from "@/components/home/Navbar";
import HeroSection from "@/components/home/HeroSection";
import LightRays from "@/components/ui/LightRays";
import HowItWorks from "@/components/home/HowItWorks";
import TechnologiesUsed from "@/components/home/TechnologiesUsed";
import TechStackShowcase from "@/components/home/TechStackShowcase";

const Home: React.FC = () => {
  return (
    <div className="home-container relative w-full min-h-screen overflow-hidden">
      {/* Background LightRays */}
      <div className="fixed inset-0 z-0">
        <LightRays 
          raysColor="#00f5ff" 
          raysOrigin="top-center"
          raysSpeed={1.2}
          lightSpread={1.5}
          rayLength={2.5}
          fadeDistance={1.2}
        />
      </div>
      {/* Foreground content */}
      <div className="relative z-20 flex flex-col min-h-screen">
        <Navbar />
        {/* Hero Section wrapped with LiquidCursor */}
        <div className="relative z-20">
            <HeroSection />
        </div>
        {/* GlobeVisualization without LiquidCursor */}
        <div className="relative z-20">
          <HowItWorks/>
        </div>
        {/* <div className="relative z-20">
          <TechnologiesUsed/>
        </div> */}
        <div className="relative z-20">
          <TechStackShowcase/>
        </div>
      </div>
    </div>
  );
};

export default Home;