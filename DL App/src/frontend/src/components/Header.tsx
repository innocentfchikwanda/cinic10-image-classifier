import React from 'react';
import { motion } from 'motion/react';
import { Eye, Zap, Stars } from 'lucide-react';

export function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="w-full text-center py-16 relative overflow-hidden"
    >
      {/* Cosmic background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{ 
            rotate: 360,
            scale: [1, 1.1, 1],
          }}
          transition={{ 
            rotate: { duration: 20, repeat: Infinity, ease: "linear" },
            scale: { duration: 4, repeat: Infinity, ease: "easeInOut" }
          }}
          className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-conic from-teal-500/20 via-cyan-500/10 to-teal-500/20 rounded-full blur-3xl"
        />
        <motion.div
          animate={{ rotate: -360 }}
          transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
          className="absolute top-1/4 right-1/4 w-32 h-32 bg-gradient-radial from-cyan-400/30 to-transparent rounded-full blur-2xl"
        />
      </div>

      <motion.div
        initial={{ scale: 0.8, rotateY: -15 }}
        animate={{ scale: 1, rotateY: 0 }}
        transition={{ delay: 0.2, duration: 0.8, ease: "easeOut" }}
        className="relative z-10 inline-flex items-center space-x-4 mb-8"
      >
        <div className="relative">
          <motion.div
            animate={{ 
              boxShadow: [
                "0 0 20px rgba(20, 184, 166, 0.3)",
                "0 0 40px rgba(20, 184, 166, 0.6)",
                "0 0 20px rgba(20, 184, 166, 0.3)"
              ]
            }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="w-16 h-16 bg-gradient-to-br from-teal-500 via-teal-600 to-cyan-500 rounded-3xl flex items-center justify-center relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent" />
            <Eye className="w-8 h-8 text-white relative z-10" />
          </motion.div>
          
          <motion.div
            animate={{ 
              rotate: [0, 10, -10, 0],
              scale: [1, 1.2, 1],
            }}
            transition={{ 
              duration: 3, 
              repeat: Infinity, 
              ease: "easeInOut",
              delay: 1
            }}
            className="absolute -top-2 -right-2"
          >
            <Stars className="w-6 h-6 text-yellow-400 drop-shadow-lg" />
          </motion.div>
          
          <motion.div
            animate={{ 
              y: [0, -2, 0],
              opacity: [0.7, 1, 0.7]
            }}
            transition={{ 
              duration: 2, 
              repeat: Infinity, 
              ease: "easeInOut" 
            }}
            className="absolute -bottom-2 -left-2"
          >
            <Zap className="w-5 h-5 text-cyan-400" />
          </motion.div>
        </div>
        
        <div className="text-left">
          <motion.h1 
            className="text-4xl bg-gradient-to-r from-white via-teal-200 to-cyan-200 bg-clip-text text-transparent"
            animate={{ 
              backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
            }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            style={{ backgroundSize: "200% 200%" }}
          >
            Iris
          </motion.h1>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="text-sm text-teal-300 tracking-wider"
          >
            THESE ROCKS CAN SEE
          </motion.div>
        </div>
      </motion.div>
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.8 }}
        className="space-y-3 relative z-10"
      >
        <p className="text-xl text-teal-200 max-w-3xl mx-auto leading-relaxed">
          When silicon dreams and sees the world
        </p>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Upload any image and watch as these computational rocks decode the essence of visual reality
        </p>
      </motion.div>

      {/* Floating particles */}
      {[...Array(6)].map((_, i) => (
        <motion.div
          key={i}
          animate={{ 
            y: [0, -20, 0],
            opacity: [0.3, 0.8, 0.3],
            scale: [1, 1.2, 1]
          }}
          transition={{ 
            duration: 3 + i * 0.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: i * 0.3
          }}
          className={`absolute w-2 h-2 bg-gradient-to-r from-teal-400 to-cyan-400 rounded-full blur-sm`}
          style={{
            left: `${20 + i * 12}%`,
            top: `${30 + (i % 2) * 40}%`,
          }}
        />
      ))}
    </motion.header>
  );
}