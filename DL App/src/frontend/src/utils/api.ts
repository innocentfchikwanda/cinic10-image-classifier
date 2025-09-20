export interface ClassificationResult {
  label: string;
  confidence: number;
  description?: string;
}

export interface ApiResponse {
  success: boolean;
  results: ClassificationResult[];
  processingTime?: number;
  model?: string;
}

const API_BASE_URL = process.env.VITE_API_URL || 'https://cinic10-backend-api.onrender.com/api';

export async function classifyImage(file: File): Promise<{
  results: ClassificationResult[];
  processingTime: number;
}> {
  const formData = new FormData();
  formData.append('file', file);

  const startTime = Date.now();
  
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Request failed with status ${response.status}`
      );
    }

    const data: ApiResponse = await response.json();
    
    if (!data.success || !data.results) {
      throw new Error('Invalid response format from server');
    }

    // Format results to match the expected interface
    const results = data.results.map(result => ({
      label: result.label,
      confidence: result.confidence,
      description: result.description || `Confidence: ${(result.confidence * 100).toFixed(2)}%`
    }));

    return {
      results,
      processingTime: data.processingTime || (Date.now() - startTime) / 1000,
    };
  } catch (error) {
    console.error('Classification error:', error);
    throw new Error(
      error instanceof Error 
        ? error.message 
        : 'Failed to classify image. Please try again.'
    );
  }
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
}
