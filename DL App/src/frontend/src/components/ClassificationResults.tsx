import React from 'react';
import { motion } from 'motion/react';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Card } from './ui/card';
import { Brain, Sparkles, Target, Zap, Crown, CircuitBoard } from 'lucide-react';

export interface ClassificationResult {
  label: string;
  confidence: number;
  description?: string;
}

interface ClassificationResultsProps {
  results: ClassificationResult[];
  processingTime: number;
}

export function ClassificationResults({ results, processingTime }: ClassificationResultsProps) {
  const topResult = results[0];
  const otherResults = results.slice(1, 4); // Show top 4 results

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="w-full max-w-3xl mx-auto space-y-8"
    >
      {/* Neural Analysis Header */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2, duration: 0.6 }}
        className="text-center space-y-3"
      >
        <div className="inline-flex items-center space-x-3">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
            className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center"
          >
            <Brain className="w-5 h-5 text-white" />
          </motion.div>
          <h2 className="text-2xl bg-gradient-to-r from-teal-200 via-cyan-200 to-emerald-200 bg-clip-text text-transparent">
            Vision Analysis Complete
          </h2>
          <motion.div
            animate={{ 
              scale: [1, 1.2, 1],
              rotate: [0, 180, 360]
            }}
            transition={{ 
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <Sparkles className="w-5 h-5 text-yellow-400" />
          </motion.div>
        </div>
        <p className="text-teal-300">Silicon consciousness has parsed your visual reality</p>
      </motion.div>

      {/* Primary Vision */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.7 }}
        className="relative"
      >
        {/* Outer glow */}
        <div className="absolute inset-0 bg-gradient-to-r from-teal-500/10 via-cyan-500/8 to-emerald-500/10 rounded-3xl blur-2xl" />
        
        <Card className="relative p-8 bg-gradient-to-br from-white/8 via-white/5 to-white/8 rounded-3xl overflow-hidden backdrop-blur-md border-0">
          {/* Animated background */}
          <div className="absolute inset-0 overflow-hidden rounded-3xl">
            <motion.div
              animate={{ 
                backgroundPosition: ["0% 0%", "100% 100%", "0% 0%"]
              }}
              transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
              className="absolute inset-0 opacity-10"
              style={{
                backgroundImage: `
                  radial-gradient(circle at 25% 25%, rgba(139, 92, 246, 0.5) 0%, transparent 50%),
                  radial-gradient(circle at 75% 75%, rgba(236, 72, 153, 0.5) 0%, transparent 50%)
                `,
                backgroundSize: "300% 300%"
              }}
            />
          </div>

          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.6, duration: 0.5 }}
            className="relative z-10 space-y-6"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <motion.div
                  animate={{ 
                    boxShadow: [
                      "0 0 20px rgba(20, 184, 166, 0.4)",
                      "0 0 40px rgba(20, 184, 166, 0.8)",
                      "0 0 20px rgba(20, 184, 166, 0.4)"
                    ]
                  }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  className="w-16 h-16 bg-gradient-to-br from-teal-500 via-teal-600 to-cyan-500 rounded-2xl flex items-center justify-center relative overflow-hidden"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent" />
                  <Crown className="w-8 h-8 text-white relative z-10" />
                </motion.div>
                
                <div>
                  <motion.h3 
                    className="text-3xl bg-gradient-to-r from-white via-teal-200 to-cyan-200 bg-clip-text text-transparent"
                    animate={{ 
                      backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
                    }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                    style={{ backgroundSize: "200% 200%" }}
                  >
                    {topResult.label}
                  </motion.h3>
                  <div className="flex items-center space-x-2">
                    <CircuitBoard className="w-4 h-4 text-teal-400" />
                    <p className="text-teal-300">Primary computational classification</p>
                  </div>
                </div>
              </div>
              
              <motion.div
                animate={{ 
                  scale: [1, 1.05, 1],
                  rotate: [0, 2, -2, 0]
                }}
                transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
              >
                <Badge 
                  variant="secondary" 
                  className="text-2xl px-6 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 text-white border-0 shadow-lg backdrop-blur-sm"
                >
                  {Math.round(topResult.confidence * 100)}%
                </Badge>
              </motion.div>
            </div>
            
            <div className="space-y-4">
              <div className="flex justify-between text-sm">
                <span className="text-teal-300">Computational Confidence</span>
                <span className="text-teal-200">{(topResult.confidence * 100).toFixed(1)}%</span>
              </div>
              
              <div className="relative">
                <Progress 
                  value={topResult.confidence * 100} 
                  className="h-3 bg-white/10 backdrop-blur-sm"
                />
                <motion.div
                  animate={{ 
                    x: ["-100%", "100%"]
                  }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut",
                    repeatDelay: 3
                  }}
                  className="absolute inset-y-0 w-1/4 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                />
              </div>
            </div>
            
            {topResult.description && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8, duration: 0.5 }}
                className="bg-gradient-to-r from-white/5 via-white/3 to-white/5 rounded-2xl p-6 backdrop-blur-sm"
              >
                <p className="text-teal-200 leading-relaxed">{topResult.description}</p>
              </motion.div>
            )}
          </motion.div>
        </Card>
      </motion.div>

      {/* Alternative Visions */}
      {otherResults.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="space-y-4"
        >
          <div className="flex items-center space-x-3">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
            >
              <Target className="w-6 h-6 text-purple-400" />
            </motion.div>
            <h4 className="text-xl text-teal-200">Alternative Computational Pathways</h4>
          </div>
          
          <div className="grid gap-4">
            {otherResults.map((result, index) => (
              <motion.div
                key={result.label}
                initial={{ opacity: 0, x: -30, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                transition={{ 
                  delay: 0.9 + index * 0.15, 
                  duration: 0.6,
                  ease: "easeOut"
                }}
              >
                <Card className="p-6 bg-gradient-to-r from-white/5 to-white/3 hover:from-white/8 hover:to-white/5 transition-all duration-500 hover:shadow-lg hover:shadow-teal-500/10 backdrop-blur-md border-0">
                  <div className="flex items-center justify-between">
                    <div className="flex-1 space-y-3">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-full" />
                        <h5 className="text-lg text-teal-200">{result.label}</h5>
                      </div>
                      
                      <div className="flex items-center space-x-3">
                        <div className="flex-1 relative">
                          <Progress 
                            value={result.confidence * 100} 
                            className="h-2 bg-white/10 backdrop-blur-sm"
                          />
                          <motion.div
                            animate={{ 
                              x: ["-100%", "100%"]
                            }}
                            transition={{ 
                              duration: 3,
                              repeat: Infinity,
                              ease: "easeInOut",
                              repeatDelay: 4,
                              delay: index * 0.5
                            }}
                            className="absolute inset-y-0 w-1/6 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                          />
                        </div>
                        <span className="text-sm text-teal-300 min-w-[3rem] text-right">
                          {Math.round(result.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Processing Metrics */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1.2, duration: 0.6 }}
        className="flex items-center justify-center space-x-4 p-6 bg-gradient-to-r from-white/5 via-white/3 to-white/5 rounded-2xl backdrop-blur-md"
      >
        <motion.div
          animate={{ 
            rotate: [0, 360],
            scale: [1, 1.1, 1]
          }}
          transition={{ 
            rotate: { duration: 4, repeat: Infinity, ease: "linear" },
            scale: { duration: 2, repeat: Infinity, ease: "easeInOut" }
          }}
        >
          <Zap className="w-5 h-5 text-cyan-400" />
        </motion.div>
        <span className="text-teal-300">Computational processing completed in</span>
        <span className="text-teal-200 px-3 py-1 bg-white/10 rounded-lg backdrop-blur-sm">
          {processingTime.toFixed(2)}s
        </span>
      </motion.div>
    </motion.div>
  );
}