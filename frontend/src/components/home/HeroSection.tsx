"use client"
import { ArrowRight } from "lucide-react";
import { useState } from "react";
const HeroSection = () => {
  const [text, setText] = useState("");
  const wordCount = text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
  const charCount = text.length;

  return (
    <section className="min-h-screen relative flex items-center justify-center overflow-hidden py-12">
      <div className="container mx-auto px-6 flex flex-col justify-center items-center gap-8 lg:gap-16">
        <div className="px-6 py-2 -mb-20 border-[0.25px] border-cyan-200 rounded-2xl backdrop-blur-2xl">AI Powered Conversational Interface</div>
        <h1 className="text-[10rem] font-bold -mb-20 text-white">FloatChat</h1>
        <p className="text-sm text-gray-300 mt-6 -mb-8">
          Want to know something about the oceanographic data?
        </p>

        <div className="relative w-full max-w-2xl group">
          <div className="relative">
            <textarea
              className="w-full h-56 bg-[#010f1a] border border-cyan-500/30 rounded-xl resize-none outline-none p-6 text-white placeholder-cyan-200/60
                        transition-all duration-500 focus:border-cyan-400 group-hover:border-cyan-400/60
                        shadow-lg shadow-cyan-500/10 focus:shadow-cyan-400/30
                        text-lg font-light pr-14"
              placeholder="Ask anything about the oceanographic data.."
              value={text}
              onChange={(e) => setText(e.target.value)}
            ></textarea>

            {/* Word counter - positioned outside the textarea */}
            <div className="absolute -bottom-7 left-0 text-xs text-cyan-300/70 transition-opacity duration-300 
                           opacity-0 group-hover:opacity-100 group-focus-within:opacity-100">
              {wordCount} words • {charCount} characters
            </div>

            {/* Glowing send button */}
            <button className="absolute bottom-4 right-4 p-2 bg-cyan-500/10 border border-cyan-400/30 rounded-full 
                             hover:bg-cyan-400/20 hover:border-cyan-400/50 transition-all duration-300
                             group-hover:opacity-100 group-focus-within:opacity-100 opacity-70
                             shadow-lg shadow-cyan-400/20 hover:shadow-cyan-400/40">
              <ArrowRight className="text-cyan-300 w-6 h-6" />

              {/* Button glow effect */}
              <div className="absolute inset-0 rounded-full bg-cyan-400/20 blur-md group-hover:bg-cyan-400/40 
                             transition-all duration-300 opacity-0 group-hover:opacity-100"></div>
            </button>
          </div>

          {/* Enhanced glowing border effect */}
          <div className="absolute -inset-2 rounded-xl bg-gradient-to-r from-cyan-400/30 via-cyan-400/60 to-cyan-400/30 
                          opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-700 blur-lg z-[-1]"></div>
          <div className="absolute -inset-1 rounded-xl bg-gradient-to-r from-cyan-400/20 via-cyan-400/40 to-cyan-400/20 
                          opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-500 z-[-1]"></div>
        </div>
        {/* Links Row (below textarea) */}
        {/* Links Row (below textarea) */}
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <a
            href="#statistics"
            className="flex items-center gap-2 px-3 py-1 rounded-lg border border-cyan-300/30 text-cyan-200/80 
               hover:text-cyan-100 hover:border-cyan-300 transition-all duration-300"
          >
            <span className="w-2 h-2 rounded-full bg-green-400"></span>
            Know through Statistics
          </a>

          <a
            href="#explore"
            className="flex items-center gap-2 px-3 py-1 rounded-lg border border-cyan-300/30 text-cyan-200/80 
               hover:text-cyan-100 hover:border-cyan-300 transition-all duration-300"
          >
            <span className="w-2 h-2 rounded-full bg-blue-400"></span>
            Explore ARGO Data
          </a>

          <a
            href="#ai"
            className="flex items-center gap-2 px-3 py-1 rounded-lg border border-cyan-300/30 text-cyan-200/80 
               hover:text-cyan-100 hover:border-cyan-300 transition-all duration-300"
          >
            <span className="w-2 h-2 rounded-full bg-purple-400"></span>
            AI-Powered Insights
          </a>

          <a
            href="#community"
            className="flex items-center gap-2 px-3 py-1 rounded-lg border border-cyan-300/30 text-cyan-200/80 
               hover:text-cyan-100 hover:border-cyan-300 transition-all duration-300"
          >
            <span className="w-2 h-2 rounded-full bg-pink-400"></span>
            Join the Community
          </a>
        </div>


      </div>
    </section>
  )
}
export default HeroSection;