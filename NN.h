#pragma once
#include <string>
#include <vector>
#include <random>


        struct Neuron
        {
           size_t WeightCount;
           size_t WeightOffset;

           float * Weights;        //WC
		   float * weightsChange;  
           float BiasWeight;
		   float BiasChange;
        };


        struct NeuralLayer
        {
            size_t NeuronCount;
            size_t OutputOffset;
            Neuron * neuron;
            float * Output;
			float * Delta;
        };


        class NeuralNetwork
        {
        public:
            NeuralNetwork(const std::vector<size_t> &Structure, const size_t inputdim);
			NeuralNetwork();
            ~NeuralNetwork();
            void Compute(const float * InputW);
			void ComputeOneLayerNN(const float * InputW, float * outputW);
			double Learn(const float * input, const float * desiredOutput, float learningRate);
			double LearnLayer(const float * input, const float * desiredOutput, size_t layer, float learningRate);
			double LearnOneLayerNN(const float * input, const float * desiredOutput, float learningRate);
			void Randomize(const float limit);
            void Mutate(const double chances, NeuralNetwork * result);
            void SaveToFile(const std::string filename);
            void ReLoadFromFile(const std::string filename);
			void CopyFrom(const NeuralNetwork * source);
			void Reset();
			void NormalizeWeightsRELU();
			void Reshape(const std::vector<size_t> &Structure, const size_t inputdim);
			
        public:
            NeuralLayer * Layer;
            size_t LayerCount;
            size_t InputCount;
			size_t MemoryPoolSize;
            float * MemoryPool;
			float WeightLimit;
			std::mt19937 gen;

        private:
            virtual void FreeDynamicMemory();
            virtual void CreateMemoryPool();
        };


		short Sign(const float x);











