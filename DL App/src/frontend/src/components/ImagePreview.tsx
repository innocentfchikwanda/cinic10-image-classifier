import React from 'react';
import { motion } from 'motion/react';
import { X, RotateCcw, Scan, Layers } from 'lucide-react';
import { Button } from './ui/button';

interface ImagePreviewProps {
  imageUrl: string;
  fileName: string;
  onRemove: () => void;
  onRestart: () => void;
}

export function ImagePreview({ imageUrl, fileName, onRemove, onRestart }: ImagePreviewProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 30 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.7, ease: "easeOut" }}
      className="w-full max-w-3xl mx-auto relative"
    >
      {/* Outer glow */}
      <div className="absolute inset-0 bg-gradient-to-r from-teal-500/10 via-cyan-500/8 to-emerald-500/10 rounded-3xl blur-2xl" />
      
      <div className="relative bg-gradient-to-br from-white/5 via-white/3 to-white/5 rounded-3xl p-8 shadow-2xl backdrop-blur-md overflow-hidden">
        {/* Animated background pattern */}
        <div className="absolute inset-0 overflow-hidden rounded-3xl">
          <motion.div
            animate={{ 
              backgroundPosition: ["0% 0%", "100% 100%", "0% 0%"]
            }}
            transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
            className="absolute inset-0 opacity-5"
            style={{
              backgroundImage: `
                radial-gradient(circle at 30% 30%, rgba(20, 184, 166, 0.4) 0%, transparent 50%),
                radial-gradient(circle at 70% 70%, rgba(34, 211, 238, 0.4) 0%, transparent 50%)
              `,
              backgroundSize: "400% 400%"
            }}
          />
        </div>

        <div className="flex items-center justify-between mb-6 relative z-10">
          <div className="flex items-center space-x-4 flex-1 min-w-0">
            <motion.div
              animate={{ 
                rotate: [0, 360],
                scale: [1, 1.1, 1]
              }}
              transition={{ 
                rotate: { duration: 8, repeat: Infinity, ease: "linear" },
                scale: { duration: 2, repeat: Infinity, ease: "easeInOut" }
              }}
              className="w-12 h-12 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg"
            >
              <Scan className="w-6 h-6 text-white" />
            </motion.div>
            
            <div className="flex-1 min-w-0">
              <h3 className="truncate text-xl bg-gradient-to-r from-teal-200 to-cyan-200 bg-clip-text text-transparent">
                {fileName}
              </h3>
              <div className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-teal-400" />
                <p className="text-sm text-teal-300">Ready for visual processing</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3 ml-4">
            <Button
              variant="outline"
              size="sm"
              onClick={onRestart}
              className="flex items-center space-x-2 bg-white/5 backdrop-blur-md text-teal-200 hover:bg-white/10 hover:text-teal-100 transition-all duration-300 border-0"
            >
              <RotateCcw className="w-4 h-4" />
              <span>New Vision</span>
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={onRemove}
              className="w-10 h-10 p-0 text-teal-300 hover:text-teal-100 hover:bg-white/5 transition-all duration-300"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="relative overflow-hidden rounded-2xl shadow-2xl"
        >
          {/* Image container with artistic effects */}
          <div className="relative bg-gradient-to-br from-white/5 to-white/3 p-2 rounded-2xl backdrop-blur-sm">
            <div className="relative overflow-hidden rounded-xl">
              <img
                src={imageUrl}
                alt="Uploaded image for classification"
                className="w-full h-auto max-h-[500px] object-contain rounded-xl"
              />
              
              {/* Artistic overlay gradients */}
              <div className="absolute inset-0 bg-gradient-to-t from-teal-900/5 via-transparent to-cyan-900/3 pointer-events-none rounded-xl" />
              <div className="absolute inset-0 bg-gradient-to-br from-transparent via-transparent to-emerald-900/3 pointer-events-none rounded-xl" />
              
              {/* Scanning line effect */}
              <motion.div
                animate={{ 
                  y: ["-100%", "100%"],
                  opacity: [0, 0.3, 0]
                }}
                transition={{ 
                  duration: 3,
                  repeat: Infinity,
                  ease: "easeInOut",
                  repeatDelay: 2
                }}
                className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-teal-400 to-transparent pointer-events-none"
              />
            </div>
          </div>
          
          {/* Corner accents - subtle glass effect */}
          <div className="absolute top-4 left-4 w-6 h-6 bg-gradient-to-br from-teal-400/20 to-transparent rounded-tl-lg backdrop-blur-sm" />
          <div className="absolute top-4 right-4 w-6 h-6 bg-gradient-to-bl from-teal-400/20 to-transparent rounded-tr-lg backdrop-blur-sm" />
          <div className="absolute bottom-4 left-4 w-6 h-6 bg-gradient-to-tr from-teal-400/20 to-transparent rounded-bl-lg backdrop-blur-sm" />
          <div className="absolute bottom-4 right-4 w-6 h-6 bg-gradient-to-tl from-teal-400/20 to-transparent rounded-br-lg backdrop-blur-sm" />
        </motion.div>

        {/* Status indicator */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="flex items-center justify-center mt-6 space-x-2 relative z-10"
        >
          <motion.div
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.5, 1, 0.5]
            }}
            transition={{ 
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="w-2 h-2 bg-green-400 rounded-full"
          />
          <span className="text-sm text-teal-300">Ready for computational vision</span>
        </motion.div>
      </div>
    </motion.div>
  );
}