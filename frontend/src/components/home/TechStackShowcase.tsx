"use client";

import { useState, useRef } from "react";
import { motion } from "framer-motion";
import {
  Layout,
  Sparkles,
  Palette,
  Server,
  Database,
  Box,
  Globe,
  FileCode,
  RotateCcw,
  Brain,
  Cpu,
  Cloud,
  BarChart3,
  Network,
} from "lucide-react";

const TechStackShowcase = () => {
  const constraintsRef = useRef(null);
  const [activeCard, setActiveCard] = useState<number | null>(null);

  const initialItems = [
    {
      id: 1,
      name: "React",
      icon: <Layout className="w-6 h-6 text-blue-400" />,
      description:
        "React powers the FloatChat frontend. It enables us to build reusable components for the conversational UI and interactive visualizations, ensuring a smooth user experience.",
      tagTitle: "Frontend",
      tagColor: "blue",
      usage: 95,
      close: false,
    },
    {
      id: 2,
      name: "TypeScript",
      icon: <FileCode className="w-6 h-6 text-sky-400" />,
      description:
        "TypeScript ensures type safety across the FloatChat stack, reducing errors when building multi-agent orchestration logic and database queries.",
      tagTitle: "Language",
      tagColor: "sky",
      usage: 90,
      close: false,
    },
    {
      id: 3,
      name: "Tailwind CSS",
      icon: <Palette className="w-6 h-6 text-teal-400" />,
      description:
        "Tailwind CSS accelerates styling with utility classes. FloatChat uses it for modern, responsive layouts with transparent layers and glowing effects.",
      tagTitle: "Styling",
      tagColor: "teal",
      usage: 95,
      close: false,
    },
    {
      id: 4,
      name: "Framer Motion",
      icon: <Sparkles className="w-6 h-6 text-purple-400" />,
      description:
        "Framer Motion is used to create smooth, physics-based animations. In FloatChat, it powers draggable, bouncy cards and animated data visualizations.",
      tagTitle: "Animation",
      tagColor: "purple",
      usage: 85,
      close: false,
    },
    {
      id: 5,
      name: "PostgreSQL",
      icon: <Database className="w-6 h-6 text-cyan-400" />,
      description:
        "PostgreSQL stores structured ARGO parameters like temperature, salinity, and time. It ensures reliable queries for FloatChat's analysis workflows.",
      tagTitle: "Database",
      tagColor: "cyan",
      usage: 92,
      close: false,
    },
    {
      id: 6,
      name: "ChromaDB",
      icon: <Box className="w-6 h-6 text-pink-400" />,
      description:
        "ChromaDB indexes vectorized metadata and summaries. It enables FloatChat to perform contextual retrieval for user queries, making conversations knowledge-driven.",
      tagTitle: "Vector DB",
      tagColor: "pink",
      usage: 87,
      close: false,
    },
    {
      id: 7,
      name: "LangChain",
      icon: <Brain className="w-6 h-6 text-yellow-400" />,
      description:
        "LangChain orchestrates our multi-agent system, enabling specialized agents to handle data retrieval, analysis, and geospatial mapping with supervisor coordination.",
      tagTitle: "AI Framework",
      tagColor: "yellow",
      usage: 88,
      close: false,
    },
    {
      id: 8,
      name: "Three.js & OGL",
      icon: <Network className="w-6 h-6 text-indigo-400" />,
      description:
        "Three.js and OGL provide hardware-accelerated 3D graphics for our globe visualization and other data visualizations in FloatChat.",
      tagTitle: "3D Graphics",
      tagColor: "indigo",
      usage: 82,
      close: false,
    },
    {
      id: 9,
      name: "FastAPI",
      icon: <Server className="w-6 h-6 text-green-400" />,
      description:
        "FastAPI powers our backend services, providing high-performance API endpoints for data ingestion, multi-agent orchestration, and conversational workflows.",
      tagTitle: "Backend",
      tagColor: "green",
      usage: 90,
      close: false,
    },
  ]

  const [techItems, setTechItems] = useState(initialItems);
  const [resetToggle, setResetToggle] = useState(false);

  const toggleCard = (id: number) => {
    setTechItems((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, close: !item.close } : item
      )
    );
  };

  const resetPositions = () => {
    setResetToggle((prev) => !prev);
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    active: {
      scale: 1.03,
      zIndex: 10,
      boxShadow:
        "0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2)",
    },
  };

  return (
    <section className="relative py-16 bg-transparent overflow-hidden">
      <div className="container mx-auto px-4 relative z-10">
        {/* Reset Button */}
        <div className="absolute top-4 right-6 z-20">
          <button
            onClick={resetPositions}
            className="p-2 rounded-full bg-zinc-800/70 hover:bg-zinc-700 transition-colors border border-zinc-600"
          >
            <RotateCcw className="w-5 h-5 text-zinc-300" />
          </button>
        </div>

        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            Technology Stack
          </h2>
          <p className="text-zinc-400 max-w-2xl mx-auto text-lg">
            FloatChat leverages cutting-edge technologies to democratize access to ARGO oceanographic data through AI-powered conversations.
          </p>
        </motion.div>

        {/* Tech Stack Grid */}
        <motion.div
          ref={constraintsRef}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
        >
          {techItems.map((item, index) => (
            <motion.div
              key={item.id + resetToggle.toString()}
              className="bg-zinc-900/40 backdrop-blur-sm rounded-xl p-6 flex flex-col border border-zinc-700/50 hover:border-zinc-500/30 transition-colors relative overflow-hidden"
              variants={cardVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              whileHover="active"
              drag
              dragConstraints={constraintsRef}
              dragElastic={0.2}
              whileDrag={{
                scale: 1.05,
                zIndex: 20,
                transition: { type: "spring", stiffness: 500, damping: 15 },
              }}
              onHoverStart={() => setActiveCard(item.id)}
              onHoverEnd={() => setActiveCard(null)}
            >
              {/* Background glow */}
              <div
                className={`absolute inset-0 rounded-xl opacity-0 transition-opacity duration-300 ${
                  activeCard === item.id ? "opacity-100" : ""
                } ${
                  item.tagColor === "blue"
                    ? "bg-blue-500/5"
                    : item.tagColor === "purple"
                    ? "bg-purple-500/5"
                    : item.tagColor === "teal"
                    ? "bg-teal-500/5"
                    : item.tagColor === "green"
                    ? "bg-green-500/5"
                    : item.tagColor === "pink"
                    ? "bg-pink-500/5"
                    : item.tagColor === "cyan"
                    ? "bg-cyan-500/5"
                    : item.tagColor === "sky"
                    ? "bg-sky-500/5"
                    : item.tagColor === "yellow"
                    ? "bg-yellow-500/5"
                    : item.tagColor === "indigo"
                    ? "bg-indigo-500/5"
                    : item.tagColor === "orange"
                    ? "bg-orange-500/5"
                    : "bg-gray-500/5"
                }`}
              ></div>

              {/* Header */}
              <div className="flex items-start justify-between mb-4 relative z-10">
                <div className="flex items-center space-x-3">
                  {item.icon}
                  <h3 className="text-xl font-semibold">{item.name}</h3>
                </div>
                <button
                  onClick={() => toggleCard(item.id)}
                  className="text-zinc-500 hover:text-white transition-colors p-1 rounded-full hover:bg-zinc-700/50"
                >
                  {item.close ? "＋" : "－"}
                </button>
              </div>

              {/* Usage tag instead of proficiency bar */}
              <div className="mb-4 relative z-10">
                <span
                  className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${
                    item.tagColor === "blue"
                      ? "bg-blue-500/20 text-blue-300"
                      : item.tagColor === "purple"
                      ? "bg-purple-500/20 text-purple-300"
                      : item.tagColor === "teal"
                      ? "bg-teal-500/20 text-teal-300"
                      : item.tagColor === "green"
                      ? "bg-green-500/20 text-green-300"
                      : item.tagColor === "pink"
                      ? "bg-pink-500/20 text-pink-300"
                      : item.tagColor === "cyan"
                      ? "bg-cyan-500/20 text-cyan-300"
                      : item.tagColor === "sky"
                      ? "bg-sky-500/20 text-sky-300"
                      : item.tagColor === "yellow"
                      ? "bg-yellow-500/20 text-yellow-300"
                      : item.tagColor === "indigo"
                      ? "bg-indigo-500/20 text-indigo-300"
                      : item.tagColor === "orange"
                      ? "bg-orange-500/20 text-orange-300"
                      : "bg-gray-500/20 text-gray-300"
                  }`}
                >
                  {item.usage}% used in project
                </span>
              </div>

              {/* Description */}
              <motion.div
                initial={false}
                animate={{
                  height: item.close ? 0 : "auto",
                  opacity: item.close ? 0 : 1,
                }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden relative z-10"
              >
                <p className="text-zinc-400 text-sm mb-4">{item.description}</p>
              </motion.div>

              {/* Footer */}
              <div className="mt-auto pt-4 relative z-10">
                {item.tagTitle && (
                  <div
                    className={`inline-block px-3 py-1 rounded-full text-xs font-medium mb-3 ${
                      item.tagColor === "blue"
                        ? "bg-blue-500/20 text-blue-300"
                        : item.tagColor === "purple"
                        ? "bg-purple-500/20 text-purple-300"
                        : item.tagColor === "teal"
                        ? "bg-teal-500/20 text-teal-300"
                        : item.tagColor === "green"
                        ? "bg-green-500/20 text-green-300"
                        : item.tagColor === "pink"
                        ? "bg-pink-500/20 text-pink-300"
                        : item.tagColor === "cyan"
                        ? "bg-cyan-500/20 text-cyan-300"
                        : item.tagColor === "sky"
                        ? "bg-sky-500/20 text-sky-300"
                        : item.tagColor === "yellow"
                        ? "bg-yellow-500/20 text-yellow-300"
                        : item.tagColor === "indigo"
                        ? "bg-indigo-500/20 text-indigo-300"
                        : item.tagColor === "orange"
                        ? "bg-orange-500/20 text-orange-300"
                        : "bg-gray-500/20 text-gray-300"
                    }`}
                  >
                    {item.tagTitle}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default TechStackShowcase;