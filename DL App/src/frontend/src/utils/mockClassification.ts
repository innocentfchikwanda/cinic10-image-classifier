import { ClassificationResult } from '../components/ClassificationResults';

// Mock classification results for demonstration
const mockResults: Record<string, ClassificationResult[]> = {
  default: [
    {
      label: "Golden Retriever",
      confidence: 0.94,
      description: "A friendly, large-sized dog breed known for their golden coat and gentle temperament."
    },
    {
      label: "Labrador Retriever",
      confidence: 0.78,
      description: "A medium-large sized working dog breed originally from Newfoundland."
    },
    {
      label: "Dog",
      confidence: 0.99,
      description: "A domesticated carnivorous mammal that typically has a long snout and an acute sense of smell."
    },
    {
      label: "Pet",
      confidence: 0.85
    }
  ],
  cat: [
    {
      label: "Domestic Cat",
      confidence: 0.96,
      description: "A small carnivorous mammal that has been domesticated for thousands of years."
    },
    {
      label: "Tabby Cat",
      confidence: 0.82,
      description: "A cat with a coat featuring distinctive stripes, dots, lines or swirling patterns."
    },
    {
      label: "Feline",
      confidence: 0.98,
      description: "A member of the cat family, characterized by retractable claws and excellent night vision."
    },
    {
      label: "Pet",
      confidence: 0.89
    }
  ],
  car: [
    {
      label: "Sports Car",
      confidence: 0.91,
      description: "A high-performance automobile designed for speed, handling, and aesthetic appeal."
    },
    {
      label: "Automobile",
      confidence: 0.97,
      description: "A wheeled motor vehicle used for transportation, typically with four wheels."
    },
    {
      label: "Vehicle",
      confidence: 0.95,
      description: "A machine designed for transporting people or cargo from one place to another."
    },
    {
      label: "Sedan",
      confidence: 0.73
    }
  ],
  flower: [
    {
      label: "Rose",
      confidence: 0.88,
      description: "A woody perennial flowering plant known for its beauty and fragrance."
    },
    {
      label: "Flower",
      confidence: 0.95,
      description: "The reproductive structure found in flowering plants, often colorful and fragrant."
    },
    {
      label: "Plant",
      confidence: 0.92,
      description: "A living organism that typically grows in soil and uses photosynthesis for energy."
    },
    {
      label: "Garden Plant",
      confidence: 0.79
    }
  ]
};

export async function mockClassifyImage(file: File): Promise<{ results: ClassificationResult[], processingTime: number }> {
  // Simulate processing time
  const startTime = Date.now();
  
  await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 2000));
  
  const processingTime = (Date.now() - startTime) / 1000;
  
  // Determine mock results based on filename (for demo purposes)
  const fileName = file.name.toLowerCase();
  let results = mockResults.default;
  
  if (fileName.includes('cat') || fileName.includes('kitten')) {
    results = mockResults.cat;
  } else if (fileName.includes('car') || fileName.includes('vehicle')) {
    results = mockResults.car;
  } else if (fileName.includes('flower') || fileName.includes('rose') || fileName.includes('plant')) {
    results = mockResults.flower;
  }
  
  // Add some randomness to confidence scores
  const randomizedResults = results.map(result => ({
    ...result,
    confidence: Math.min(0.99, result.confidence + (Math.random() - 0.5) * 0.1)
  }));
  
  // Sort by confidence
  randomizedResults.sort((a, b) => b.confidence - a.confidence);
  
  return {
    results: randomizedResults,
    processingTime
  };
}