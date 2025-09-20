import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Header } from './components/Header';
import { ImageUpload } from './components/ImageUpload';
import { ImagePreview } from './components/ImagePreview';
import { ClassificationResults, ClassificationResult } from './components/ClassificationResults';
import { classifyImage, checkHealth } from './utils/api';

interface UploadedImage {
  file: File;
  url: string;
}

interface ImageUploadProps {
  onImageUpload: (file: File) => void;
  isUploading: boolean;
  disabled?: boolean;
}

export default function App() {
  const [uploadedImage, setUploadedImage] = useState<UploadedImage | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classificationResults, setClassificationResults] = useState<{
    results: ClassificationResult[];
    processingTime: number;
  } | null>(null);

  const [apiStatus, setApiStatus] = useState<'idle' | 'checking' | 'ready' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);

  // Check API health on component mount
  useEffect(() => {
    const checkApiHealth = async () => {
      setApiStatus('checking');
      try {
        const health = await checkHealth();
        console.log('API Health:', health);
        setApiStatus('ready');
      } catch (err) {
        console.error('Failed to connect to API:', err);
        setApiStatus('error');
        setError('Failed to connect to the classification service. Please make sure the backend server is running.');
      }
    };

    checkApiHealth();
  }, []);

  const handleImageUpload = useCallback(async (file: File) => {
    const url = URL.createObjectURL(file);
    setUploadedImage({ file, url });
    setClassificationResults(null);
    setError(null);
    setIsClassifying(true);

    try {
      const startTime = Date.now();
      const results = await classifyImage(file);
      const processingTime = (Date.now() - startTime) / 1000;
      
      setClassificationResults({
        ...results,
        processingTime: parseFloat(processingTime.toFixed(2))
      });
    } catch (err) {
      console.error('Classification failed:', err);
      setError(
        err instanceof Error 
          ? err.message 
          : 'Failed to classify image. Please try again.'
      );
    } finally {
      setIsClassifying(false);
    }
  }, []);

  const handleRemoveImage = useCallback(() => {
    if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage.url);
    }
    setUploadedImage(null);
    setClassificationResults(null);
    setIsClassifying(false);
  }, [uploadedImage]);

  const handleRestart = useCallback(() => {
    if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage.url);
    }
    setUploadedImage(null);
    setClassificationResults(null);
    setIsClassifying(false);
  }, [uploadedImage]);

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* API Status Indicator */}
      <div className={`fixed top-4 right-4 z-50 px-4 py-2 rounded-lg text-sm font-medium ${
        apiStatus === 'ready' ? 'bg-green-500/20 text-green-400' :
        apiStatus === 'checking' ? 'bg-yellow-500/20 text-yellow-400' :
        apiStatus === 'error' ? 'bg-red-500/20 text-red-400' :
        'bg-gray-500/20 text-gray-400'
      }`}>
        {apiStatus === 'ready' ? 'API Connected' :
         apiStatus === 'checking' ? 'Connecting to API...' :
         apiStatus === 'error' ? 'API Connection Failed' :
         'Checking API...'}
      </div>
      
      {/* Error Banner */}
      {error && (
        <div className="fixed top-0 left-0 right-0 z-50 bg-red-500/90 text-white p-4 text-center">
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <span>{error}</span>
            <button 
              onClick={() => setError(null)}
              className="ml-4 p-1 hover:bg-white/20 rounded-full"
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        </div>
      )}
      
      {/* Futuristic background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {/* Main gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-black via-gray-900 to-black" />
        
        {/* Animated elements */}
        <motion.div
          animate={{ 
            rotate: 360,
            scale: [1, 1.2, 1],
          }}
          transition={{ 
            rotate: { duration: 50, repeat: Infinity, ease: "linear" },
            scale: { duration: 8, repeat: Infinity, ease: "easeInOut" }
          }}
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-conic from-teal-500/20 via-teal-400/10 via-cyan-400/10 to-teal-500/20 rounded-full blur-3xl"
        />
        
        <motion.div
          animate={{ 
            rotate: -360,
            x: [0, 100, 0],
            y: [0, -50, 0]
          }}
          transition={{ 
            rotate: { duration: 60, repeat: Infinity, ease: "linear" },
            x: { duration: 20, repeat: Infinity, ease: "easeInOut" },
            y: { duration: 15, repeat: Infinity, ease: "easeInOut" }
          }}
          className="absolute top-3/4 right-1/4 w-64 h-64 bg-gradient-radial from-cyan-400/30 via-teal-400/20 to-transparent rounded-full blur-2xl"
        />
        
        <motion.div
          animate={{ 
            rotate: 360,
            x: [0, -80, 0],
            y: [0, 60, 0]
          }}
          transition={{ 
            rotate: { duration: 40, repeat: Infinity, ease: "linear" },
            x: { duration: 25, repeat: Infinity, ease: "easeInOut" },
            y: { duration: 18, repeat: Infinity, ease: "easeInOut" }
          }}
          className="absolute bottom-1/4 left-1/3 w-48 h-48 bg-gradient-radial from-teal-400/25 via-emerald-400/15 to-transparent rounded-full blur-xl"
        />

        {/* Floating particles */}
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            animate={{ 
              y: [0, -30, 0],
              x: [0, Math.sin(i) * 20, 0],
              opacity: [0.2, 0.8, 0.2],
              scale: [1, 1.5, 1]
            }}
            transition={{ 
              duration: 4 + i * 0.3,
              repeat: Infinity,
              ease: "easeInOut",
              delay: i * 0.2
            }}
            className="absolute w-1 h-1 bg-gradient-to-r from-teal-400 to-cyan-400 rounded-full"
            style={{
              left: `${10 + (i * 4.5) % 80}%`,
              top: `${20 + (i * 3) % 60}%`,
            }}
          />
        ))}
      </div>
      
      <div className="relative z-10 container mx-auto px-4 py-8">
        <Header />
        
        <main className="space-y-16">
          <AnimatePresence mode="wait">
            {!uploadedImage ? (
              <>
                {apiStatus === 'error' && (
                  <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-300">
                    <p className="font-medium">Backend Connection Error</p>
                    <p className="text-sm mt-1">
                      Unable to connect to the classification service. Please make sure the backend server is running.
                    </p>
                    <button
                      onClick={async () => {
                        setApiStatus('checking');
                        try {
                          await checkHealth();
                          setApiStatus('ready');
                          setError(null);
                        } catch (err) {
                          setApiStatus('error');
                        }
                      }}
                      className="mt-2 text-sm text-red-300 hover:text-white underline"
                    >
                      Retry Connection
                    </button>
                  </div>
                )}
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                >
                  <ImageUpload 
                    onImageUpload={handleImageUpload} 
                    isUploading={isClassifying}
                    disabled={apiStatus !== 'ready'}
                  />
                </motion.div>
              </>
            ) : (
              <motion.div
                key="preview"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -30 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
                className="space-y-12"
              >
                <ImagePreview
                  imageUrl={uploadedImage.url}
                  fileName={uploadedImage.file.name}
                  onRemove={handleRemoveImage}
                  onRestart={handleRestart}
                />
                
                {isClassifying && (
                  <motion.div
                    initial={{ opacity: 0, y: 20, scale: 0.9 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    className="flex flex-col items-center space-y-6"
                  >
                    <div className="relative">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                        className="w-16 h-16 border-4 border-teal-500/30 border-t-teal-500 rounded-full"
                      />
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
                        className="absolute inset-2 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-full blur-sm"
                      />
                    </div>
                    
                    <div className="text-center space-y-2">
                      <motion.p 
                        className="text-xl text-teal-200"
                        animate={{ 
                          backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
                        }}
                        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                      >
                        Silicon consciousness awakening...
                      </motion.p>
                      <p className="text-teal-400">These computational rocks are learning to see</p>
                    </div>
                  </motion.div>
                )}
                
                {classificationResults && !isClassifying && (
                  <ClassificationResults
                    results={classificationResults.results}
                    processingTime={classificationResults.processingTime}
                  />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
        
        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.8 }}
          className="text-center py-16 relative"
        >
          <div className="space-y-3">
            <motion.div
              animate={{ 
                scale: [1, 1.05, 1],
                opacity: [0.7, 1, 0.7]
              }}
              transition={{ 
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="inline-flex items-center space-x-2 text-teal-300"
            >
              <div className="w-2 h-2 bg-teal-400 rounded-full" />
              <span>Powered by advanced neural architectures</span>
              <div className="w-2 h-2 bg-cyan-400 rounded-full" />
            </motion.div>
            <p className="text-sm text-teal-500">
              Where artificial intelligence meets creative vision
            </p>
          </div>
        </motion.footer>
      </div>
    </div>
  );
}