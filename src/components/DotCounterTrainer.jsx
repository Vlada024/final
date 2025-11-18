import React, { useState, useRef, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const DotCounterTrainer = () => {
  const [config, setConfig] = useState({
    maxDots: 6,
    imageSize: 64,
    dotRadius: 5,
    allowOverlap: false,
    randomSeed: 42,
    batchSize: 8,
    epochs: 5,
    stepsPerEpoch: 20,
    validationSteps: 5,
    updateEveryBatches: 1,
    learningRate: 0.001,
    optimizer: 'adam'
  });

  const [isTraining, setIsTraining] = useState(false);
  const [modelArchitecture, setModelArchitecture] = useState(null);
  const [trainingData, setTrainingData] = useState([]);
  const [currentBatch, setCurrentBatch] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [generatedImages, setGeneratedImages] = useState([]);
  const canvasRefs = useRef([]);
  const shouldStopRef = useRef(false);

  class SeededRandom {
    constructor(seed) {
      this.seed = seed;
    }
    
    random() {
      const x = Math.sin(this.seed++) * 10000;
      return x - Math.floor(x);
    }
  }

  class AdamOptimizer {
    constructor(learningRate, beta1, beta2, epsilon) {
      this.learningRate = learningRate || 0.001;
      this.beta1 = beta1 || 0.9;
      this.beta2 = beta2 || 0.999;
      this.epsilon = epsilon || 1e-8;
      this.m = null;
      this.v = null;
      this.t = 0;
    }

    initialize(shapes) {
      this.m = shapes.map(shape => {
        if (Array.isArray(shape[0])) {
          return shape.map(row => new Array(row.length).fill(0));
        }
        return new Array(shape.length).fill(0);
      });
      this.v = shapes.map(shape => {
        if (Array.isArray(shape[0])) {
          return shape.map(row => new Array(row.length).fill(0));
        }
        return new Array(shape.length).fill(0);
      });
    }

    update(params, grads) {
      if (!this.m) {
        this.initialize(params);
      }
      this.t++;

      const updated = [];
      for (let i = 0; i < params.length; i++) {
        if (Array.isArray(params[i][0])) {
          const updatedParam = params[i].map((row, r) => 
            row.map((val, c) => {
              this.m[i][r][c] = this.beta1 * this.m[i][r][c] + (1 - this.beta1) * grads[i][r][c];
              this.v[i][r][c] = this.beta2 * this.v[i][r][c] + (1 - this.beta2) * grads[i][r][c] * grads[i][r][c];
              
              const mHat = this.m[i][r][c] / (1 - Math.pow(this.beta1, this.t));
              const vHat = this.v[i][r][c] / (1 - Math.pow(this.beta2, this.t));
              
              return val - this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
            })
          );
          updated.push(updatedParam);
        } else {
          const updatedParam = params[i].map((val, idx) => {
            this.m[i][idx] = this.beta1 * this.m[i][idx] + (1 - this.beta1) * grads[i][idx];
            this.v[i][idx] = this.beta2 * this.v[i][idx] + (1 - this.beta2) * grads[i][idx] * grads[i][idx];
            
            const mHat = this.m[i][idx] / (1 - Math.pow(this.beta1, this.t));
            const vHat = this.v[i][idx] / (1 - Math.pow(this.beta2, this.t));
            
            return val - this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
          });
          updated.push(updatedParam);
        }
      }
      return updated;
    }
  }

  class SimpleNN {
    constructor(inputSize, hiddenSize, outputSize, learningRate, optimizer) {
      this.inputSize = inputSize;
      this.hiddenSize = hiddenSize;
      this.outputSize = outputSize;
      this.learningRate = learningRate || 0.001;
      this.optimizerType = optimizer || 'adam';
      
      this.w1 = this.initWeights(inputSize, hiddenSize);
      this.b1 = new Array(hiddenSize).fill(0);
      this.w2 = this.initWeights(hiddenSize, outputSize);
      this.b2 = new Array(outputSize).fill(0);
      
      if (optimizer === 'adam') {
        this.optimizer = new AdamOptimizer(learningRate);
      }
    }
    
    initWeights(rows, cols) {
      const scale = Math.sqrt(2.0 / rows);
      const weights = [];
      for (let i = 0; i < rows; i++) {
        weights[i] = [];
        for (let j = 0; j < cols; j++) {
          weights[i][j] = (Math.random() * 2 - 1) * scale;
        }
      }
      return weights;
    }
    
    relu(x) {
      return Math.max(0, x);
    }
    
    reluDerivative(x) {
      return x > 0 ? 1 : 0;
    }
    
    softmax(arr) {
      const max = Math.max(...arr);
      const exps = arr.map(x => Math.exp(x - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map(x => x / sum);
    }
    
    forward(input) {
      const hidden = new Array(this.hiddenSize);
      for (let i = 0; i < this.hiddenSize; i++) {
        let sum = this.b1[i];
        for (let j = 0; j < this.inputSize; j++) {
          sum += input[j] * this.w1[j][i];
        }
        hidden[i] = this.relu(sum);
      }
      
      const output = new Array(this.outputSize);
      for (let i = 0; i < this.outputSize; i++) {
        let sum = this.b2[i];
        for (let j = 0; j < this.hiddenSize; j++) {
          sum += hidden[j] * this.w2[j][i];
        }
        output[i] = sum;
      }
      
      const probs = this.softmax(output);
      
      return { hidden, output, probs };
    }
    
    train(inputs, labels) {
      let totalLoss = 0;
      let correct = 0;
      
      const dW1Acc = this.w1.map(row => new Array(row.length).fill(0));
      const dB1Acc = new Array(this.hiddenSize).fill(0);
      const dW2Acc = this.w2.map(row => new Array(row.length).fill(0));
      const dB2Acc = new Array(this.outputSize).fill(0);
      
      for (let idx = 0; idx < inputs.length; idx++) {
        const input = inputs[idx];
        const label = labels[idx];
        
        const result = this.forward(input);
        const hidden = result.hidden;
        const probs = result.probs;
        
        totalLoss -= Math.log(probs[label] + 1e-10);
        
        const predicted = probs.indexOf(Math.max(...probs));
        if (predicted === label) correct++;
        
        const dOutput = [...probs];
        dOutput[label] -= 1;
        
        for (let i = 0; i < this.hiddenSize; i++) {
          for (let j = 0; j < this.outputSize; j++) {
            dW2Acc[i][j] += hidden[i] * dOutput[j];
          }
        }
        for (let j = 0; j < this.outputSize; j++) {
          dB2Acc[j] += dOutput[j];
        }
        
        const dHidden = new Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
          for (let j = 0; j < this.outputSize; j++) {
            dHidden[i] += this.w2[i][j] * dOutput[j];
          }
          dHidden[i] *= this.reluDerivative(hidden[i]);
        }
        
        for (let i = 0; i < this.inputSize; i++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            dW1Acc[i][j] += input[i] * dHidden[j];
          }
        }
        for (let j = 0; j < this.hiddenSize; j++) {
          dB1Acc[j] += dHidden[j];
        }
      }
      
      const n = inputs.length;
      for (let i = 0; i < this.w1.length; i++) {
        for (let j = 0; j < this.w1[0].length; j++) {
          dW1Acc[i][j] /= n;
        }
      }
      for (let i = 0; i < this.hiddenSize; i++) {
        dB1Acc[i] /= n;
      }
      for (let i = 0; i < this.w2.length; i++) {
        for (let j = 0; j < this.w2[0].length; j++) {
          dW2Acc[i][j] /= n;
        }
      }
      for (let i = 0; i < this.outputSize; i++) {
        dB2Acc[i] /= n;
      }
      
      if (this.optimizerType === 'adam') {
        const updated = this.optimizer.update(
          [this.w1, this.b1, this.w2, this.b2],
          [dW1Acc, dB1Acc, dW2Acc, dB2Acc]
        );
        this.w1 = updated[0];
        this.b1 = updated[1];
        this.w2 = updated[2];
        this.b2 = updated[3];
      } else {
        for (let i = 0; i < this.w1.length; i++) {
          for (let j = 0; j < this.w1[0].length; j++) {
            this.w1[i][j] -= this.learningRate * dW1Acc[i][j];
          }
        }
        for (let i = 0; i < this.hiddenSize; i++) {
          this.b1[i] -= this.learningRate * dB1Acc[i];
        }
        for (let i = 0; i < this.w2.length; i++) {
          for (let j = 0; j < this.w2[0].length; j++) {
            this.w2[i][j] -= this.learningRate * dW2Acc[i][j];
          }
        }
        for (let i = 0; i < this.outputSize; i++) {
          this.b2[i] -= this.learningRate * dB2Acc[i];
        }
      }
      
      return {
        loss: totalLoss / inputs.length,
        accuracy: correct / inputs.length
      };
    }
    
    evaluate(inputs, labels) {
      let correct = 0;
      for (let idx = 0; idx < inputs.length; idx++) {
        const result = this.forward(inputs[idx]);
        const probs = result.probs;
        const predicted = probs.indexOf(Math.max(...probs));
        if (predicted === labels[idx]) correct++;
      }
      return correct / inputs.length;
    }
  }

  const generateShapeImage = (numShapes, size, shapeSize, allowOverlap, rng) => {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, size, size);
    
    const shapes = [];
    let attempts = 0;
    const maxAttempts = 1000;
    
    while (shapes.length < numShapes && attempts < maxAttempts) {
      const x = rng.random() * (size - 2 * shapeSize) + shapeSize;
      const y = rng.random() * (size - 2 * shapeSize) + shapeSize;
      const shapeType = Math.floor(rng.random() * 3);
      
      let valid = true;
      if (!allowOverlap) {
        for (const shape of shapes) {
          const dist = Math.sqrt((x - shape.x) * (x - shape.x) + (y - shape.y) * (y - shape.y));
          if (dist < 2 * shapeSize) {
            valid = false;
            break;
          }
        }
      }
      
      if (valid) {
        shapes.push({ x: x, y: y, type: shapeType });
      }
      attempts++;
    }
    
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    
    for (const shape of shapes) {
      // Draw all shapes as dots (filled circles)
      ctx.beginPath();
      ctx.arc(shape.x, shape.y, shapeSize, 0, 2 * Math.PI);
      ctx.fill();
    }
    
    return canvas;
  };

  const generateDataset = (numSamples, maxShapes, size, shapeSize, allowOverlap, seed) => {
    const images = [];
    const labels = [];
    const rng = new SeededRandom(seed);
    
    for (let i = 0; i < numSamples; i++) {
      const numShapes = Math.floor(rng.random() * maxShapes) + 1;
      const canvas = generateShapeImage(numShapes, size, shapeSize, allowOverlap, rng);
      
      const ctx = canvas.getContext('2d');
      const imageData = ctx.getImageData(0, 0, size, size);
      const normalized = [];
      
      for (let j = 0; j < imageData.data.length; j += 4) {
        normalized.push(imageData.data[j] / 255.0);
      }
      
      images.push(normalized);
      labels.push(numShapes - 1);
    }
    
    return { images: images, labels: labels };
  };

  const getModelSummary = (inputSize, hiddenSize, outputSize, optimizer) => {
    const summary = [];
    const line1 = '═';
    const line2 = '─';
    summary.push(line1.repeat(80));
    summary.push('  NEURAL NETWORK ARCHITECTURE: Shape Counter MLP');
    summary.push(line1.repeat(80));
    summary.push('');
    summary.push('  Network Type: Multi-Layer Perceptron (MLP)');
    summary.push('  Optimizer: ' + optimizer.toUpperCase());
    summary.push('  Loss Function: Sparse Categorical Cross-Entropy');
    summary.push('');
    summary.push(line2.repeat(80));
    summary.push('Layer (type)'.padEnd(35) + 'Output Shape'.padEnd(25) + 'Param #');
    summary.push(line1.repeat(80));
    
    summary.push('input_layer (InputLayer)'.padEnd(35) + ('(None, ' + inputSize + ')').padEnd(25) + '0');
    summary.push(line2.repeat(80));
    
    const w1Params = inputSize * hiddenSize;
    const b1Params = hiddenSize;
    summary.push('dense_hidden (Dense)'.padEnd(35) + ('(None, ' + hiddenSize + ')').padEnd(25) + (w1Params + b1Params).toString());
    summary.push(line2.repeat(80));
    
    summary.push('activation_relu (ReLU)'.padEnd(35) + ('(None, ' + hiddenSize + ')').padEnd(25) + '0');
    summary.push(line2.repeat(80));
    
    const w2Params = hiddenSize * outputSize;
    const b2Params = outputSize;
    summary.push('dense_output (Dense)'.padEnd(35) + ('(None, ' + outputSize + ')').padEnd(25) + (w2Params + b2Params).toString());
    summary.push(line2.repeat(80));
    
    summary.push('activation_softmax (Softmax)'.padEnd(35) + ('(None, ' + outputSize + ')').padEnd(25) + '0');
    summary.push(line1.repeat(80));
    
    const totalParams = w1Params + b1Params + w2Params + b2Params;
    summary.push('');
    summary.push('  Total params: ' + totalParams.toLocaleString());
    summary.push('  Trainable params: ' + totalParams.toLocaleString());
    summary.push('  Non-trainable params: 0');
    summary.push('');
    summary.push(line1.repeat(80));
    
    return summary.join('\n');
  };

  const startTraining = async () => {
    setIsTraining(true);
    shouldStopRef.current = false;
    setTrainingData([]);
    setCurrentBatch(0);
    setCurrentEpoch(0);
    setGeneratedImages([]);
    
    const maxDots = config.maxDots;
    const imageSize = config.imageSize;
    const dotRadius = config.dotRadius;
    const allowOverlap = config.allowOverlap;
    const randomSeed = config.randomSeed;
    const batchSize = config.batchSize;
    const epochs = config.epochs;
    const stepsPerEpoch = config.stepsPerEpoch;
    const validationSteps = config.validationSteps;
    const updateEveryBatches = config.updateEveryBatches;
    const learningRate = config.learningRate;
    const optimizer = config.optimizer;
    
    const trainData = generateDataset(stepsPerEpoch * batchSize, maxDots, imageSize, dotRadius, allowOverlap, randomSeed);
    const valData = generateDataset(validationSteps * batchSize, maxDots, imageSize, dotRadius, allowOverlap, randomSeed + 1000);
    
    const inputSize = imageSize * imageSize;
    const hiddenSize = 128;
    const outputSize = maxDots;
    const model = new SimpleNN(inputSize, hiddenSize, outputSize, learningRate, optimizer);
    
    const summary = getModelSummary(inputSize, hiddenSize, outputSize, optimizer);
    setModelArchitecture(summary);
    
    const chartData = [];
    let globalStep = 0;
    
    for (let epoch = 0; epoch < epochs && !shouldStopRef.current; epoch++) {
      setCurrentEpoch(epoch + 1);
      
      for (let step = 0; step < stepsPerEpoch && !shouldStopRef.current; step++) {
        const batchStart = step * batchSize;
        const batchEnd = Math.min(batchStart + batchSize, trainData.images.length);
        
        const batchInputs = trainData.images.slice(batchStart, batchEnd);
        const batchLabels = trainData.labels.slice(batchStart, batchEnd);
        
        const result = model.train(batchInputs, batchLabels);
        
        if (step % updateEveryBatches === 0) {
          const sampleRng = new SeededRandom(randomSeed + globalStep);
          const numShapesInSample = Math.floor(sampleRng.random() * maxDots) + 1;
          const canvas = generateShapeImage(numShapesInSample, imageSize, dotRadius, allowOverlap, sampleRng);
          setGeneratedImages([{ canvas: canvas, label: numShapesInSample }]);
          
          let valAcc = null;
          if (step % (updateEveryBatches * 3) === 0) {
            valAcc = model.evaluate(valData.images, valData.labels);
          }
          
          chartData.push({
            step: globalStep,
            trainAcc: result.accuracy * 100,
            valAcc: valAcc ? valAcc * 100 : null
          });
          
          setTrainingData([...chartData]);
          setCurrentBatch(step + 1);
        }
        
        globalStep++;
        
        if (shouldStopRef.current) {
          break;
        }
        
        await new Promise(resolve => setTimeout(resolve, 20));
      }
      
      if (shouldStopRef.current) {
        break;
      }
    }
    
    setIsTraining(false);
  };

  const stopTraining = () => {
    shouldStopRef.current = true;
  };

  useEffect(() => {
    generatedImages.forEach((img, idx) => {
      if (canvasRefs.current[idx]) {
        const ctx = canvasRefs.current[idx].getContext('2d');
        ctx.drawImage(img.canvas, 0, 0);
      }
    });
  }, [generatedImages]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          Shape Counter Neural Network Trainer
        </h1>
        
        <div className="bg-gray-800 rounded-lg p-6 mb-8 shadow-xl">
          <h2 className="text-2xl font-bold mb-4 text-blue-400">Configuration</h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm mb-2 text-gray-300">Max Shapes (1-9)</label>
              <input type="number" min="1" max="9" value={config.maxDots} onChange={(e) => setConfig({...config, maxDots: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none" />
            </div>
            
            <div>
              <label className="block text-sm mb-2 text-gray-300">Image Size (px)</label>
              <select value={config.imageSize} onChange={(e) => setConfig({...config, imageSize: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none">
                <option value="32">32</option>
                <option value="64">64</option>
                <option value="128">128</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm mb-2 text-gray-300">Shape Size (px)</label>
              <select value={config.dotRadius} onChange={(e) => setConfig({...config, dotRadius: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none">
                <option value="3">3</option>
                <option value="5">5</option>
                <option value="7">7</option>
                <option value="10">10</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm mb-2 text-gray-300">Random Seed</label>
              <input type="number" value={config.randomSeed} onChange={(e) => setConfig({...config, randomSeed: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none" />
            </div>
            
            <div>
              <label className="block text-sm mb-2 text-gray-300">Batch Size</label>
              <input type="number" value={config.batchSize} onChange={(e) => setConfig({...config, batchSize: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none" />
            </div>
            
            <div>
              <label className="block text-sm mb-2 text-gray-300">Epochs</label>
              <input type="number" value={config.epochs} onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none" />
            </div>
            
            <div>
              <label className="block text-sm mb-2 text-gray-300">Steps/Epoch</label>
              <input type="number" value={config.stepsPerEpoch} onChange={(e) => setConfig({...config, stepsPerEpoch: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none" />
            </div>
            
            <div>
              <label className="block text-sm mb-2 text-gray-300">Validation Steps</label>
              <input type="number" value={config.validationSteps} onChange={(e) => setConfig({...config, validationSteps: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none" />
            </div>

            <div>
              <label className="block text-sm mb-2 text-gray-300">Update Every (batches)</label>
              <input type="number" min="1" value={config.updateEveryBatches} onChange={(e) => setConfig({...config, updateEveryBatches: parseInt(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none" />
            </div>

            <div>
              <label className="block text-sm mb-2 text-gray-300">Learning Rate</label>
              <input type="number" step="0.0001" value={config.learningRate} onChange={(e) => setConfig({...config, learningRate: parseFloat(e.target.value)})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none" />
            </div>

            <div>
              <label className="block text-sm mb-2 text-gray-300">Optimizer</label>
              <select value={config.optimizer} onChange={(e) => setConfig({...config, optimizer: e.target.value})} disabled={isTraining} className="w-full bg-gray-700 rounded px-3 py-2 text-white border border-gray-600 focus:border-blue-500 focus:outline-none">
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
              </select>
            </div>
          </div>
          
          <div className="mt-4">
            <label className="flex items-center text-gray-300">
              <input type="checkbox" checked={config.allowOverlap} onChange={(e) => setConfig({...config, allowOverlap: e.target.checked})} disabled={isTraining} className="mr-2" />
              <span>Allow Shapes to Overlap</span>
            </label>
          </div>
          
          <div className="flex gap-4 mt-6">
            <button onClick={startTraining} disabled={isTraining} className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed py-3 px-6 rounded-lg font-bold text-lg transition">
              {isTraining ? 'Training... (Epoch ' + currentEpoch + ', Batch ' + currentBatch + ')' : 'Start Training'}
            </button>
            
            {isTraining && (
              <button onClick={stopTraining} className="bg-red-600 hover:bg-red-700 py-3 px-6 rounded-lg font-bold text-lg transition">
                Stop Training
              </button>
            )}
          </div>
        </div>
        
        {modelArchitecture && (
          <div className="bg-gray-800 rounded-lg p-6 mb-8 shadow-xl">
            <h2 className="text-2xl font-bold mb-4 text-green-400">Model Architecture</h2>
            <pre className="bg-gray-900 p-4 rounded overflow-x-auto text-sm text-green-300 font-mono">
              {modelArchitecture}
            </pre>
          </div>
        )}
        
        {generatedImages.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-6 mb-8 shadow-xl">
            <h2 className="text-2xl font-bold mb-4 text-purple-400">Generated Training Samples</h2>
            <div className="flex justify-center">
              {generatedImages.map((img, idx) => (
                <div key={idx} className="text-center">
                  <canvas
                    ref={el => canvasRefs.current[idx] = el}
                    width={config.imageSize}
                    height={config.imageSize}
                    className="border-4 border-gray-600 rounded bg-black"
                    style={{ width: '300px', height: '300px', imageRendering: 'pixelated' }}
                  />
                  <p className="mt-4 text-lg font-bold text-gray-300">{img.label} shape{img.label > 1 ? 's' : ''}</p>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {trainingData.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
            <h2 className="text-2xl font-bold mb-4 text-yellow-400">Training Progress</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis 
                  dataKey="step" 
                  stroke="#fff"
                  label={{ value: 'Training Step', position: 'insideBottom', offset: -5, fill: '#fff' }}
                />
                <YAxis 
                  stroke="#fff"
                  label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#fff' }}
                  domain={[0, 100]}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }}
                  formatter={(value) => value ? value.toFixed(2) + '%' : 'N/A'}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="trainAcc" 
                  stroke="#3b82f6" 
                  name="Training Accuracy"
                  dot={false}
                  strokeWidth={2}
                />
                <Line 
                  type="monotone" 
                  dataKey="valAcc" 
                  stroke="#10b981" 
                  name="Validation Accuracy"
                  dot={false}
                  strokeWidth={2}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default DotCounterTrainer;
