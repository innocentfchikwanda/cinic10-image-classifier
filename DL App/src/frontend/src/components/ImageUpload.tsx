import React, { useCallback, useState } from 'react';
import { motion } from 'motion/react';
import { Upload, Image as ImageIcon, Sparkles, Zap } from 'lucide-react';

interface ImageUploadProps {
  onImageUpload: (file: File) => void;
  isUploading: boolean;
  disabled?: boolean;
}

export function ImageUpload({ onImageUpload, isUploading, disabled = false }: ImageUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (disabled) return;
    
    const files = Array.from(e.dataTransfer.files) as File[];
    const imageFile = files.find((file: File) => {
      const fileType = file.type || '';
      return fileType.startsWith('image/');
    });
    
    if (imageFile) {
      onImageUpload(imageFile);
    }
  }, [onImageUpload, disabled]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file);
    }
  }, [onImageUpload]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="w-full max-w-3xl mx-auto relative"
    >
      {/* Outer glow effect */}
      <motion.div
        animate={isDragOver ? { 
          scale: 1.02,
          opacity: 0.8
        } : { 
          scale: 1,
          opacity: 0.4
        }}
        className="absolute inset-0 bg-gradient-to-r from-teal-500/10 via-cyan-500/10 to-emerald-500/10 rounded-3xl blur-2xl"
      />
      
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={[
          'relative rounded-3xl p-16 text-center transition-all duration-500 backdrop-blur-md',
          isDragOver && !disabled
            ? 'bg-gradient-to-br from-teal-500/15 via-cyan-500/10 to-emerald-500/15 scale-105 shadow-2xl shadow-teal-500/20'
            : disabled
              ? 'bg-gray-800/30 opacity-70 cursor-not-allowed'
              : 'bg-gradient-to-br from-white/5 via-white/3 to-white/5 hover:from-teal-500/10 hover:via-cyan-500/5 hover:to-emerald-500/10 hover:shadow-lg hover:shadow-teal-500/10',
          isUploading || disabled ? 'pointer-events-none' : 'cursor-pointer',
        ].join(' ')}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className={`absolute inset-0 w-full h-full opacity-0 ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}
          disabled={isUploading || disabled}
        />
        
        {/* Animated background pattern */}
        <div className="absolute inset-0 overflow-hidden rounded-3xl">
          <motion.div
            animate={{ 
              backgroundPosition: ["0% 0%", "100% 100%", "0% 0%"]
            }}
            transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
            className="absolute inset-0 opacity-10"
            style={{
              backgroundImage: `
                radial-gradient(circle at 20% 20%, rgba(20, 184, 166, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(34, 211, 238, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 70%, rgba(5, 150, 105, 0.3) 0%, transparent 50%)
              `,
              backgroundSize: "300% 300%"
            }}
          />
        </div>
        
        <motion.div
          animate={isDragOver || isUploading ? { scale: 1.1, y: -5 } : { scale: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          className="flex flex-col items-center space-y-8 relative z-10"
        >
          <div className="relative">
            <motion.div
              animate={isUploading ? { 
                rotate: 360,
                scale: [1, 1.1, 1]
              } : isDragOver ? {
                scale: [1, 1.2, 1],
                rotate: [0, 5, -5, 0]
              } : {}}
              transition={isUploading ? { 
                rotate: { repeat: Infinity, duration: 3, ease: "linear" },
                scale: { repeat: Infinity, duration: 2, ease: "easeInOut" }
              } : isDragOver ? {
                duration: 0.6,
                ease: "easeInOut"
              } : {}}
              className="relative"
            >
              <div className="w-24 h-24 bg-gradient-to-br from-teal-500 via-teal-600 to-cyan-500 rounded-2xl flex items-center justify-center relative overflow-hidden shadow-lg shadow-teal-500/25">
                <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent" />
                {isUploading ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                    className="w-8 h-8 border-3 border-white border-t-transparent rounded-full"
                  />
                ) : (
                  <Upload className="w-10 h-10 text-white relative z-10" />
                )}
              </div>
              
              {/* Floating sparkles */}
              <motion.div
                animate={{ 
                  scale: [1, 1.2, 1],
                  rotate: [0, 180, 360],
                  opacity: [0.7, 1, 0.7]
                }}
                transition={{ 
                  duration: 2, 
                  repeat: Infinity, 
                  ease: "easeInOut",
                  delay: 0.2
                }}
                className="absolute -top-2 -right-2"
              >
                <Sparkles className="w-6 h-6 text-yellow-400" />
              </motion.div>
              
              <motion.div
                animate={{ 
                  y: [0, -3, 0],
                  opacity: [0.6, 1, 0.6]
                }}
                transition={{ 
                  duration: 1.5, 
                  repeat: Infinity, 
                  ease: "easeInOut",
                  delay: 0.8
                }}
                className="absolute -bottom-2 -left-2"
              >
                <Zap className="w-5 h-5 text-cyan-400" />
              </motion.div>
            </motion.div>
          </div>
          
          <div className="space-y-4 text-center">
            <motion.h3 
              className="text-2xl bg-gradient-to-r from-white via-teal-200 to-cyan-200 bg-clip-text text-transparent"
              animate={isUploading ? { 
                backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
              } : {}}
              transition={isUploading ? { 
                duration: 2, 
                repeat: Infinity, 
                ease: "easeInOut" 
              } : {}}
              style={{ backgroundSize: "200% 200%" }}
            >
              {isUploading ? 'Neural Processing...' : isDragOver ? 'Release to Analyze' : 'Drop Your Vision Here'}
            </motion.h3>
            
            <p className="text-teal-300 max-w-md mx-auto leading-relaxed">
              {isUploading 
                ? 'Decoding visual patterns through deep neural networks'
                : isDragOver
                ? 'Let the AI see through your lens'
                : 'Drag and drop an image or click to browse your creative universe'
              }
            </p>
          </div>
          
          {!isUploading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="flex items-center space-x-4 text-sm text-teal-400"
            >
              <div className="flex items-center space-x-2">
                <ImageIcon className="w-4 h-4" />
                <span>JPG, PNG, GIF, WebP</span>
              </div>
              <div className="w-1 h-1 bg-teal-400 rounded-full" />
              <span>Up to 10MB</span>
            </motion.div>
          )}
        </motion.div>

        {/* Processing animation overlay */}
        {isUploading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute inset-0 rounded-3xl overflow-hidden"
          >
            <motion.div
              animate={{ 
                x: ["-100%", "100%"]
              }}
              transition={{ 
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-teal-400/20 to-transparent"
            />
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}