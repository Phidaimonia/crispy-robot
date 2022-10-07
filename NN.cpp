#include <cmath>
#include <fstream>
#include <intrin.h>
#include "NN.h"
#include <iostream>
#include "string.h"
#include <memory>
#include <random>


void prepareRandomGen(std::mt19937 & gen)
{
	std::random_device rdev;
	std::seed_seq::result_type data[std::mt19937::state_size];
	std::generate_n(data, std::mt19937::state_size, std::ref(rdev));

	std::seed_seq prng_seed(data, data + std::mt19937::state_size);
	gen.seed(prng_seed);
}

NeuralNetwork::NeuralNetwork()
{
	prepareRandomGen(gen);
	LayerCount = 0;
	InputCount = 0;
	MemoryPoolSize = 0;
	WeightLimit = 1.0;
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &Structure, size_t inputdim)
{
	prepareRandomGen(gen);
	LayerCount = Structure.size();
	InputCount = inputdim;
	WeightLimit = 1.0;

	Layer = new NeuralLayer[LayerCount];

	for (size_t l = 0; l < LayerCount; ++l)
	{
		Layer[l].NeuronCount = Structure[l];
		Layer[l].neuron = new Neuron[Structure[l]];
		Layer[l].Delta = new float[Structure[l]];
		
		memset(Layer[l].Delta, 0, sizeof(float) * Layer[l].NeuronCount);
		
		for (size_t n = 0; n < Structure[l]; ++n)
		{
			Layer[l].neuron[n].BiasChange = 0;
			if (l == 0) Layer[l].neuron[n].WeightCount = inputdim;
			else Layer[l].neuron[n].WeightCount = Layer[l - 1].NeuronCount;
			Layer[l].neuron[n].weightsChange = new float[Layer[l].neuron[n].WeightCount];
			memset(Layer[l].neuron[n].weightsChange, 0, sizeof(float) * Layer[l].neuron[n].WeightCount);
		}
	}

	CreateMemoryPool();
}

NeuralNetwork::~NeuralNetwork()
{
	FreeDynamicMemory();
}




void NeuralNetwork::CreateMemoryPool()
{
	MemoryPoolSize = InputCount * Layer[0].NeuronCount; // vahy vstupu, inp pro kazdy vstupny neuron
	Layer[0].OutputOffset = MemoryPoolSize; // pointer to Output
	MemoryPoolSize += Layer[0].NeuronCount; // Output
	WeightLimit = 1.0;

	for (size_t i = 0; i < Layer[0].NeuronCount; ++i)
		Layer[0].neuron[i].WeightOffset = i * Layer[0].neuron[i].WeightCount;   // weightoffset pro kazdy pocatecni neuron 


	for (size_t l = 1; l < LayerCount; ++l)
	{
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			Layer[l].neuron[n].WeightOffset = MemoryPoolSize;
			MemoryPoolSize += Layer[l].neuron[n].WeightCount;
		}

		Layer[l].OutputOffset = MemoryPoolSize; // pointer to Output
		MemoryPoolSize += Layer[l].NeuronCount;
	}


	MemoryPool = new float[MemoryPoolSize];
	memset(MemoryPool, 0, sizeof(float) * MemoryPoolSize);

	for (size_t l = 0; l < LayerCount; ++l)
	{
		Layer[l].Output = MemoryPool + Layer[l].OutputOffset;

		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
			Layer[l].neuron[n].Weights = MemoryPool + Layer[l].neuron[n].WeightOffset;
	}

}


void NeuralNetwork::FreeDynamicMemory()
{
	for (size_t l = 0; l < LayerCount; ++l)
	{
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
			delete[] Layer[l].neuron[n].weightsChange;

		delete[] Layer[l].Delta;
		delete[] Layer[l].neuron;
	}

	if(LayerCount > 0) delete[] Layer;
	if (MemoryPoolSize > 0) delete[] MemoryPool;
}




void NeuralNetwork::Reshape(const std::vector<size_t> &Structure, size_t inputdim)
{
	FreeDynamicMemory();

	LayerCount = Structure.size();
	InputCount = inputdim;
	WeightLimit = 1.0;

	Layer = new NeuralLayer[LayerCount];

	for (size_t l = 0; l < LayerCount; ++l)
	{
		Layer[l].NeuronCount = Structure[l];
		Layer[l].neuron = new Neuron[Structure[l]];
		Layer[l].Delta = new float[Structure[l]];

		memset(Layer[l].Delta, 0, sizeof(float) * Layer[l].NeuronCount);

		for (size_t n = 0; n < Structure[l]; ++n)
		{
			Layer[l].neuron[n].BiasChange = 0;
			if (l == 0) Layer[l].neuron[n].WeightCount = inputdim;
			else Layer[l].neuron[n].WeightCount = Layer[l - 1].NeuronCount;
			Layer[l].neuron[n].weightsChange = new float[Layer[l].neuron[n].WeightCount];
			memset(Layer[l].neuron[n].weightsChange, 0, sizeof(float) * Layer[l].neuron[n].WeightCount);
		}
	}

	CreateMemoryPool();
}




float MulSum(const float * weights, const float * inputs, size_t N)
{
	size_t lim = N / 8 * 8;
	__m256 tmpres = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

	for (size_t i = 0; i < lim; i += 8)
	{
		__m256 w = _mm256_loadu_ps(weights + i);
		__m256 inp = _mm256_loadu_ps(inputs + i);
		tmpres = _mm256_add_ps(tmpres, _mm256_mul_ps(w, inp));
	}

	float tmp[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	_mm256_storeu_ps(tmp, tmpres);

	float res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

	for (size_t i = lim; i < N; ++i)
		res += weights[i] * inputs[i];

	return res;
}

void CalculateDelta(float * deltaWeights, const float * inputs, float delta, size_t N)
{
	
	size_t lim = N / 8 * 8;
	__m256 dt_ps = _mm256_set_ps(-delta, -delta, -delta, -delta, -delta, -delta, -delta, -delta);

	for (size_t i = 0; i < lim; i += 8)
	{
		__m256 inputs_ps = _mm256_loadu_ps(inputs + i);
		__m256 tmpres = _mm256_mul_ps(inputs_ps, dt_ps);
		_mm256_storeu_ps(deltaWeights + i, tmpres);
	} 

	for (size_t i = lim; i < N; ++i)
	{
		deltaWeights[i] = -delta * inputs[i];
	} 
} 


void CalculateDeltaMomentum(float * deltaWeights, const float * inputs, float delta, size_t N)
{
	const double d = 0.0;
	__m256 alpha_ps = _mm256_set_ps(d, d, d, d, d, d, d, d);
	size_t lim = N / 8 * 8;
	__m256 dt_ps = _mm256_set_ps(-delta, -delta, -delta, -delta, -delta, -delta, -delta, -delta);

	for (size_t i = 0; i < lim; i += 8)
	{
		__m256 inputs_ps = _mm256_loadu_ps(inputs + i);
		__m256 tmpres = _mm256_mul_ps(inputs_ps, dt_ps);
		_mm256_storeu_ps(deltaWeights + i, _mm256_add_ps(tmpres, _mm256_mul_ps(_mm256_loadu_ps(deltaWeights + i), alpha_ps)));
	} 

	for (size_t i = lim; i < N; ++i)  // LIM
	{
		double t = -delta * inputs[i];
		deltaWeights[i] = t + deltaWeights[i] * d;  // bounciness
	}
}


void AddArray(float * weights, const float * delta, size_t N)
{

	size_t lim = N / 8 * 8;

	for (size_t i = 0; i < lim; i += 8)
	{
		__m256 weights_ps = _mm256_loadu_ps(weights + i);
		__m256 delta_ps = _mm256_loadu_ps(delta + i);
		_mm256_storeu_ps(weights + i, _mm256_add_ps(weights_ps, delta_ps));
	}

	for (size_t i = lim; i < N; ++i)
		weights[i] += delta[i];
}



short Sign(const float x)
{
	if (x < 0) return -1;
	else return 1;
}



double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double fastSigmoid(double value)
{
	if (value < -1000.0) value = -1000.0;
	if (value > 1000.0) value = 1000.0;

	float x = ::abs(value);
	float x2 = x * x;
	float e = 1.0f + x + x2 * 0.555f + x2 * x2*0.143f;
	float res = 1.0f / (1.0f + (value > 0 ? 1.0f / e : e));
	return res;
}

double dSigmoid(double x)
{
	return x * (1.0 - x);
}



double relu(double x)
{
	if (x > 10000) return 10000;
	if (x < -10000) return -10000;
	if (x > 0) 
		return x;
	return 0;
}

double dRelu(double x)
{
	if (x > 0) 
		return x;
	return 0;
}


void NeuralNetwork::NormalizeWeightsRELU()
{
	double maxW = 0;
	double maxNL = 0;
	for (size_t l = 0; l < LayerCount-1; ++l)
	{
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			for (size_t w = 0; w < Layer[l].neuron[n].WeightCount; ++w)
				if (abs(Layer[l].neuron[n].Weights[w]) > maxW) maxW = abs(Layer[l].neuron[n].Weights[w]);
			if(abs(Layer[l].neuron[n].BiasWeight) > maxW) maxW = abs(Layer[l].neuron[n].BiasWeight);

			for (size_t w = 0; w < Layer[l+1].NeuronCount; ++w)
				if (abs(Layer[l+1].neuron[w].Weights[n]) > maxNL) maxNL = abs(Layer[l+1].neuron[w].Weights[n]);
			
			double norm = sqrt(maxW * maxNL);

			Layer[l].neuron[n].BiasWeight *= norm / maxW;

			for (size_t w = 0; w < Layer[l].neuron[n].WeightCount; ++w)
				Layer[l].neuron[n].Weights[w] *= norm / maxW;

			for (size_t w = 0; w < Layer[l + 1].NeuronCount; ++w)
				Layer[l+1].neuron[w].Weights[n] *= norm / maxNL;

		}
	}
}



void ComputeLayer(NeuralLayer & layer, const float * input)
{
	float tmp = 0;
	for (size_t n = 0; n < layer.NeuronCount; ++n)
	{
		tmp = MulSum(layer.neuron[n].Weights, input, layer.neuron[n].WeightCount);
		tmp += layer.neuron[n].BiasWeight;
		layer.Output[n] = fastSigmoid(tmp);
	}

	//memcpy(layer.Output, layer.newOutput, sizeof(float) * layer.NeuronCount);
}




double ComputeLayerMSE(const NeuralLayer & layer, const float * EstimatedOutput)
{
	double mse = 0;
	for (size_t i = 0; i < layer.NeuronCount; ++i)
	{
		double error = layer.Output[i] - EstimatedOutput[i];
		mse += error * error;
	}

	return mse;
}




double NeuralNetwork::LearnOneLayerNN(const float * input, const float * desiredOutput, float learningRate)
{
	double mse = 0.0;

	Compute(input);

	mse = ComputeLayerMSE(Layer[0], desiredOutput);
	//learningRate *= pow(mse, 0.3);

	for (size_t n = 0; n < Layer[0].NeuronCount; ++n)
	{
		float error = Layer[0].Output[n] - desiredOutput[n];
		float delta = error;

		Layer[0].Delta[n] = delta;
		Layer[0].neuron[n].BiasWeight += Layer[0].neuron[n].BiasChange * 0.5 - learningRate * error;
		Layer[0].neuron[n].BiasChange = -learningRate * error;
		if (Layer[0].neuron[n].BiasWeight < -1000) Layer[0].neuron[n].BiasWeight = -1000;
		if (Layer[0].neuron[n].BiasWeight > 1000) Layer[0].neuron[n].BiasWeight = 1000;

		CalculateDeltaMomentum(Layer[0].neuron[n].weightsChange, input, learningRate * delta, Layer[0].neuron[n].WeightCount);
		AddArray(Layer[0].neuron[n].Weights, Layer[0].neuron[n].weightsChange, Layer[0].neuron[n].WeightCount);
	}
	return mse;
}




double NeuralNetwork::Learn(const float * input, const float * desiredOutput, float learningRate)
{
	size_t LastLayer = LayerCount - 1;
	double mse = 0.0;

	Compute(input);
	mse = ComputeLayerMSE(Layer[LastLayer], desiredOutput);


	for (size_t n = 0; n < Layer[LastLayer].NeuronCount; ++n)
	{
		float error = Layer[LastLayer].Output[n] - desiredOutput[n];
		Layer[LastLayer].Delta[n] = dSigmoid(Layer[LastLayer].Output[n]) * error;

		Layer[LastLayer].neuron[n].BiasWeight += -error * learningRate;

		if (error != 0)	CalculateDeltaMomentum(Layer[LastLayer].neuron[n].weightsChange, Layer[LastLayer - 1].Output, learningRate * Layer[LastLayer].Delta[n], Layer[LastLayer - 1].NeuronCount);
	}


	for (size_t l = LastLayer - 1; l > 0; --l)
	{
		double outSum = 1.0; // bias
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
			outSum += abs(Layer[l].Output[n]);

		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			float error = 0;
			for (size_t i = 0; i < Layer[l + 1].NeuronCount; ++i)
				error += Layer[l + 1].Delta[i] * Layer[l + 1].neuron[i].Weights[n];

			float delta = error * dSigmoid(Layer[l].Output[n]);

			Layer[l].Delta[n] = delta;

			Layer[l].neuron[n].BiasWeight += -error * learningRate;

			if (delta != 0)	CalculateDeltaMomentum(Layer[l].neuron[n].weightsChange, Layer[l - 1].Output, delta * learningRate, Layer[l - 1].NeuronCount);
		}
	}


	double outSum = 1.0;
	for (size_t n = 0; n < Layer[0].NeuronCount; ++n)
		outSum += abs(Layer[0].Output[n]);


	for (size_t n = 0; n < Layer[0].NeuronCount; ++n)
	{
		float error = 0;
		for (size_t i = 0; i < Layer[1].NeuronCount; ++i)
			error += Layer[1].Delta[i] * Layer[1].neuron[i].Weights[n];

		float delta = dSigmoid(Layer[0].Output[n]) * error; // relative output importance

		Layer[0].Delta[n] = delta;

		Layer[0].neuron[n].BiasWeight += -error * learningRate;

		if (delta != 0) CalculateDeltaMomentum(Layer[0].neuron[n].weightsChange, input, delta * learningRate, InputCount);
	} 





	for (size_t l = 0; l <= LastLayer; ++l)
	{
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
			AddArray(Layer[l].neuron[n].Weights, Layer[l].neuron[n].weightsChange, Layer[l].neuron[n].WeightCount);
	} 

	return mse;
}



double NeuralNetwork::LearnLayer(const float * input, const float * desiredOutput, size_t layer, float learningRate)
{
	size_t LastLayer = LayerCount - 1;
	double mse = 0.0;

	Compute(input);
	mse = ComputeLayerMSE(Layer[LastLayer], desiredOutput);


	for (size_t n = 0; n < Layer[LastLayer].NeuronCount; ++n)
	{
		float error = Sign(Layer[LastLayer].Output[n] - desiredOutput[n]);
		Layer[LastLayer].Delta[n] = error;

		if (layer == LastLayer)
		{
			double t = -error * learningRate;

			Layer[LastLayer].neuron[n].BiasChange = t + Layer[LastLayer].neuron[n].BiasChange * 0.95;  // bounciness
			if (abs(Layer[LastLayer].neuron[n].BiasChange) > abs(20 * learningRate * error)) 
				Layer[LastLayer].neuron[n].BiasChange = Sign(Layer[LastLayer].neuron[n].BiasChange) * abs(20 * learningRate * error);

			//Layer[LastLayer].neuron[n].BiasChange = t;
			Layer[LastLayer].neuron[n].BiasWeight += Layer[LastLayer].neuron[n].BiasChange;

			if(error != 0)	CalculateDeltaMomentum(Layer[LastLayer].neuron[n].weightsChange, Layer[LastLayer - 1].Output, learningRate * error, Layer[LastLayer - 1].NeuronCount);
		}
	}


	for (size_t l = LastLayer - 1; l > 0; --l)
	{
		double outSum = 1.0; // bias
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
			outSum += abs(Layer[l].Output[n]);

		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			float error = 0;
			for (size_t i = 0; i < Layer[l + 1].NeuronCount; ++i)
				error += Layer[l + 1].Delta[i] * Layer[l + 1].neuron[i].Weights[n];

			float delta = (abs(Layer[l].Output[n]) / outSum) * error;

			Layer[l].Delta[n] = delta;



			if (layer == l)
			{
				double t = -error * learningRate;

				Layer[l].neuron[n].BiasChange = t + Layer[l].neuron[n].BiasChange * 0.95;  // bounciness
				if (abs(Layer[l].neuron[n].BiasChange) > abs(20 * learningRate * error))
					Layer[l].neuron[n].BiasChange = Sign(Layer[l].neuron[n].BiasChange) * abs(20 * learningRate * error);


				//Layer[l].neuron[n].BiasChange = t;
				Layer[l].neuron[n].BiasWeight += Layer[l].neuron[n].BiasChange;

				if (delta != 0)	CalculateDeltaMomentum(Layer[l].neuron[n].weightsChange, Layer[l - 1].Output, delta * learningRate, Layer[l - 1].NeuronCount);
			}
		}
	}


	double outSum = 1.0;
	for (size_t n = 0; n < Layer[0].NeuronCount; ++n)
		outSum += abs(Layer[0].Output[n]);


	for (size_t n = 0; n < Layer[0].NeuronCount; ++n)
	{
		float error = 0;
		for (size_t i = 0; i < Layer[1].NeuronCount; ++i)
			error += abs(Layer[1].Delta[i] * Layer[1].neuron[i].Weights[n]);

		float delta = abs(Layer[0].Output[n]) / outSum * error; // relative output importance

		Layer[0].Delta[n] = delta;

		if (layer == 0)
		{
			double t = -error * learningRate;

			Layer[0].neuron[n].BiasChange = t + Layer[0].neuron[n].BiasChange * 0.95;  // bounciness
			if (abs(Layer[0].neuron[n].BiasChange) > abs(20 * learningRate * error))
				Layer[0].neuron[n].BiasChange = Sign(Layer[0].neuron[n].BiasChange) * abs(20 * learningRate * error);

			//Layer[0].neuron[n].BiasChange = t;
			Layer[0].neuron[n].BiasWeight += Layer[0].neuron[n].BiasChange;

			if (delta != 0) CalculateDeltaMomentum(Layer[0].neuron[n].weightsChange, input, delta * learningRate, InputCount);
		}
	}


	
	for (size_t n = 0; n < Layer[layer].NeuronCount; ++n)
		AddArray(Layer[layer].neuron[n].Weights, Layer[layer].neuron[n].weightsChange, Layer[layer].neuron[n].WeightCount);
		
	return mse;
}




void NeuralNetwork::ComputeOneLayerNN(const float * InputW, float * outputW)
{
	ComputeLayer(Layer[0], InputW);   // 1 layer
	memcpy(outputW, Layer[0].Output, sizeof(float) * Layer[0].NeuronCount);
}



void NeuralNetwork::Compute(const float * InputW)
{
	ComputeLayer(Layer[0], InputW);  

	for (size_t l = 1; l < LayerCount; ++l)
		ComputeLayer(Layer[l], Layer[l - 1].Output);  

	//memcpy(outputW, Layer[LayerCount - 1].Output, sizeof(float) * Layer[LayerCount - 1].NeuronCount);
}



void NeuralNetwork::Randomize(const float limit)
{
	std::uniform_int_distribution<std::mt19937::result_type> rnd(1, 4000000000);

	size_t layers = LayerCount;
	WeightLimit = limit;
	for (size_t l = 0; l < layers; ++l)
	{
		size_t neurons = Layer[l].NeuronCount;
		memset(Layer[l].Output, 0, sizeof(float) * Layer[l].NeuronCount);

		for (size_t n = 0; n < neurons; ++n)
		{
			Layer[l].neuron[n].BiasWeight = -WeightLimit + 2 * rnd(gen) * 0.0000000005 * WeightLimit;
			for (size_t w = 0; w < Layer[l].neuron[n].WeightCount; ++w)
				Layer[l].neuron[n].Weights[w] = -WeightLimit + 2 * rnd(gen) * 0.0000000005 * WeightLimit;
		}
	}
}



void NeuralNetwork::Mutate(const double chances, NeuralNetwork * result)
{
	const std::uniform_int_distribution<std::mt19937::result_type> rnd(1, 4000000000);


	if (!result) return;
	size_t layers = LayerCount;

	for (size_t l = 0; l < layers; l++)
	{
		size_t neurons = Layer[l].NeuronCount;

		memset(Layer[l].Output, 0, sizeof(float) * Layer[l].NeuronCount);  // reset recurrent state

		for (size_t n = 0; n < neurons; n++)
		{
			if (rnd(gen) * 0.00000000025 < chances)
				result->Layer[l].neuron[n].BiasWeight = Layer[l].neuron[n].BiasWeight;
			else  result->Layer[l].neuron[n].BiasWeight = -WeightLimit + 2 * rnd(gen) * 0.0000000005 * WeightLimit;

			for (size_t w = 0; w < Layer[l].neuron[n].WeightCount; w++)
				if (rnd(gen) * 0.00000000025 < chances)
					result->Layer[l].neuron[n].Weights[w] = Layer[l].neuron[n].Weights[w];
				else result->Layer[l].neuron[n].Weights[w] = -WeightLimit + 2 * rnd(gen) * 0.0000000005 * WeightLimit;
		}
	}

}







void NeuralNetwork::SaveToFile(const std::string filename)
{
	std::ofstream out(filename, std::ios::out | std::ios_base::binary);

	size_t layers = LayerCount;
	out.write(reinterpret_cast<const char*>(&layers), sizeof(size_t));             // layers

	for (size_t l = 0; l < layers; ++l)
	{
		out.write(reinterpret_cast<const char*>(&Layer[l].NeuronCount), sizeof(size_t));                          // pocet neuronu
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			out.write(reinterpret_cast<const char*>(&Layer[l].neuron[n].WeightCount), sizeof(size_t));
			out.write(reinterpret_cast<const char*>(&Layer[l].neuron[n].BiasWeight), sizeof(float));
		}
	}


	for (size_t l = 0; l < layers; ++l)
	{
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			for (size_t w = 0; w < Layer[l].neuron[n].WeightCount; ++w)
				out.write(reinterpret_cast<const char*>(&Layer[l].neuron[n].Weights[w]), sizeof(float));
		}
	}

	out.close();

	return;
}


void NeuralNetwork::ReLoadFromFile(const std::string filename)
{
	FreeDynamicMemory();

	std::fstream file;
	file.open(filename, std::ios::in | std::ios_base::binary);

	file.read(reinterpret_cast<char*>(&LayerCount), sizeof(size_t));         // layers
	Layer = new NeuralLayer[LayerCount];

	for (size_t l = 0; l < LayerCount; ++l)
	{
		file.read(reinterpret_cast<char*>(&Layer[l].NeuronCount), sizeof(size_t));

		Layer[l].neuron = new Neuron[Layer[l].NeuronCount];

		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			file.read(reinterpret_cast<char*>(&Layer[l].neuron[n].WeightCount), sizeof(size_t));
			file.read(reinterpret_cast<char*>(&Layer[l].neuron[n].BiasWeight), sizeof(float));
		}

	}

	CreateMemoryPool();

	for (size_t l = 0; l < LayerCount; ++l)
		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			for (size_t w = 0; w < Layer[l].neuron[n].WeightCount; ++w)
				file.read(reinterpret_cast<char*>(&Layer[l].neuron[n].Weights[w]), sizeof(float));
		}


	file.close();
	return;
}




void NeuralNetwork::CopyFrom(const NeuralNetwork * source)
{
	//FreeDynamicMemory();

	LayerCount = source->LayerCount;
	InputCount = source->InputCount;
	WeightLimit = source->WeightLimit;
	/*Layer = new NeuralLayer[LayerCount];

	for (size_t l = 0; l < LayerCount; ++l)
	{
		Layer[l].NeuronCount = source->Layer[l].NeuronCount;
		Layer[l].neuron = new Neuron[Layer[l].NeuronCount];
		Layer[l].newOutput = new float[Layer[l].NeuronCount];

		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			Layer[l].neuron[n].WeightCount = source->Layer[l].neuron[n].WeightCount;
			Layer[l].neuron[n].BiasWeight = source->Layer[l].neuron[n].BiasWeight;
		}
	}

	CreateMemoryPool();*/

	for (size_t l = 0; l < LayerCount; ++l)
	{
		memset(Layer[l].Output, 0, sizeof(float) * Layer[l].NeuronCount);  // reset recurrent state
		memset(Layer[l].Delta, 0, sizeof(float) * Layer[l].NeuronCount);

		for (size_t n = 0; n < Layer[l].NeuronCount; ++n)
		{
			Layer[l].neuron[n].BiasWeight = source->Layer[l].neuron[n].BiasWeight;

			memcpy(Layer[l].neuron[n].Weights, source->Layer[l].neuron[n].Weights, sizeof(float) * Layer[l].neuron[n].WeightCount);
		}
	}
}



void NeuralNetwork::Reset()
{
	for (size_t l = 0; l < LayerCount; ++l)
	{
		memset(Layer[l].Output, 0, sizeof(float) * Layer[l].NeuronCount);  // reset recurrent state
		memset(Layer[l].Delta, 0, sizeof(float) * Layer[l].NeuronCount);
	}
}






