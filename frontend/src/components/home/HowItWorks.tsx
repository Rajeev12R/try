"use client";

import React from "react";
import GlobeVisualization from "@/components/home/GlobeVisualization";
import { motion } from "framer-motion";
import { ArrowRight, Database, Brain, Globe, BarChart3 } from "lucide-react";
import { ShootingStars } from "@/components/ui/shooting-stars";
import { StarsBackground } from "@/components/ui/stars-background";

const HowItWorks: React.FC = () => {
  const steps = [
    {
      icon: Database,
      title: "Data Ingestion",
      description: "NetCDF files processed into structured PostgreSQL database with temperature, salinity, location, and time parameters.",
      color: "from-cyan-400 to-blue-500"
    },
    {
      icon: Brain,
      title: "Vectorization",
      description: "Metadata and semantic summaries indexed in ChromaDB for intelligent contextual retrieval.",
      color: "from-blue-500 to-purple-500"
    },
    {
      icon: Globe,
      title: "Multi-Agent System",
      description: "Specialized agents handle data retrieval, analysis, and geospatial mapping with supervisor orchestration.",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: BarChart3,
      title: "Visualization",
      description: "Interactive maps, depth profiles, and time-series graphs in intuitive React-based interface.",
      color: "from-pink-500 to-red-500"
    }
  ];

  return (
    <div className="relative w-full min-h-screen bg-neutral-900 overflow-hidden">
      {/* Shooting stars and star background */}
      <ShootingStars />
      <StarsBackground />

      {/* Subtle overlay for content readability */}
      <div className="absolute inset-0 bg-neutral-900/30 z-[1]" />

      <div className="relative flex flex-col lg:flex-row items-center justify-center w-full min-h-screen px-6 lg:px-16 py-12 z-10">
        {/* Left Section - Globe */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8, rotateY: -45 }}
          whileInView={{ opacity: 1, scale: 1, rotateY: 0 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          viewport={{ once: true }}
          className="w-full lg:w-1/2 h-[400px] pointer-events-none lg:h-[700px] flex items-center justify-center relative"
        >
          {/* Glowing ring around globe */}
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-400/20 via-transparent to-purple-400/20 blur-xl animate-spin-slow" />
          <div className="relative z-10">
            <GlobeVisualization />
          </div>
          
          {/* Floating particles */}
          <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-cyan-400 rounded-full animate-bounce delay-300" />
          <div className="absolute bottom-1/3 right-1/4 w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce delay-700" />
          <div className="absolute top-1/2 right-1/3 w-1 h-1 bg-blue-400 rounded-full animate-bounce delay-1000" />
        </motion.div>

        {/* Right Section - Content */}
        <motion.div
          initial={{ opacity: 0, x: 100 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
          viewport={{ once: true }}
          className="w-full lg:w-1/2 mt-10 lg:mt-0 lg:pl-12"
        >
          {/* Main heading with enhanced styling */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl sm:text-5xl lg:text-6xl font-black leading-tight mb-4">
              <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
                How FloatChat
              </span>
              <br />
              <span className="text-white/90">
                Works
              </span>
            </h2>
            
            {/* Subtitle */}
            <p className="text-xl text-slate-300 font-medium mb-8 leading-relaxed">
              AI-powered oceanographic data democratization through intelligent multi-agent architecture
            </p>
          </motion.div>

          {/* Process steps */}
          <div className="space-y-6 mb-10">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 + 0.4 }}
                viewport={{ once: true }}
                className="group relative"
              >
                <div className="flex items-start gap-4 p-6 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-500 hover:scale-105">
                  <div className={`p-3 rounded-xl bg-gradient-to-r ${step.color} shadow-lg group-hover:shadow-xl transition-all duration-300`}>
                    <step.icon className="w-6 h-6 text-white" />
                  </div>
                  
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-white mb-2">
                      {step.title}
                    </h3>
                    <p className="text-slate-300 leading-relaxed">
                      {step.description}
                    </p>
                  </div>
                  
                  <div className="text-2xl font-black text-white/20 group-hover:text-white/40 transition-colors duration-300">
                    {String(index + 1).padStart(2, '0')}
                  </div>
                </div>
                
                {/* Connection line */}
                {index < steps.length - 1 && (
                  <div className="absolute left-8 top-full w-0.5 h-6 bg-gradient-to-b from-white/20 to-transparent" />
                )}
              </motion.div>
            ))}
          </div>

          {/* Enhanced CTA */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            viewport={{ once: true }}
            className="relative group"
          >
            <motion.button
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.98 }}
              className="relative inline-flex items-center gap-3 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-600 text-white px-8 py-4 rounded-2xl font-bold text-lg shadow-2xl hover:shadow-cyan-500/25 transition-all duration-300 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <span className="relative">Explore FloatChat</span>
              <ArrowRight className="w-6 h-6 relative group-hover:translate-x-1 transition-transform duration-300" />
            </motion.button>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default HowItWorks;