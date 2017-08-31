#pragma once
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>

#include <random>
#include <iomanip>

#include "neuron.h"

std::default_random_engine rng;

double randInitWeight()
{
	rng.seed(std::random_device()());
	std::uniform_real_distribution<double> dist_a_b(-0.5, 0.5);
	return dist_a_b(rng);
}

struct Sample
{
  Sample(std::vector<double> vec, int classM) { point = vec; classMark = classM; }
  std::vector<double> point;
  int classMark;
};

class Perceptron
{
public:
  Perceptron(int nInputs, int nOutputs, int nLayers, std::vector<int>& nNeurons);
  ~Perceptron() {};

  int startLearning(std::vector<Sample>& samples, 
										int nEpochs, 
										double learningSpeed, 
										double quality, 
										std::vector<Sample>& test_samples,
										std::vector<std::vector<double>>& train_plot,
										std::vector<std::vector<double>>& test_plot);
	int getOutputMatrix(std::vector<std::vector<double>>& output_maxtrix, std::vector<double> input_point);
	double startTest(std::vector<Sample>& samples);
	double MSE(std::vector<double> vec1, std::vector<double> vec2);

	int saveToTxt();
	int loadNetWeights(char* path);

private:
  std::vector<HiddenLayer> layers_;

  int nInputs_;
  int nOutputs_;
};




Perceptron::Perceptron(int nInputs, int nOutputs, int nLayers, std::vector<int>& nNeurons)
{
  if (nLayers != nNeurons.size())
  {
    std::cout << "nLayers != nNeurons.size()" << std::endl;
    exit(0);
  }

  nInputs_ = nInputs;
  nOutputs_ = nOutputs;

  layers_.resize(nLayers);

  HiddenLayer first_lr;
  std::vector<double> weightMatrix(nInputs + 1);
	for (int p = 0; p < weightMatrix.size(); ++p) weightMatrix[p] = randInitWeight();

  for (int j = 0; j < nNeurons[0]; ++j)
    first_lr.push_back(NeuronLogistic(weightMatrix));
  layers_[0] = first_lr;

  for (int i = 1; i < nLayers; ++i)
  {
    HiddenLayer lr;
    /* Filling layer with neurons */
    for (int j = 0; j < nNeurons[i]; ++j)
    {
      /* Creating neuron and weight matrix, filled with initW */
      std::vector<double> weightMatrix(nNeurons[i-1] + 1);
			for (int p = 0; p < weightMatrix.size(); ++p) weightMatrix[p] = randInitWeight();

      NeuronLogistic neuron(weightMatrix);
      lr.push_back(neuron);
    }

    layers_[i] = lr;
  }

	HiddenLayer lastLayer;
  for (int i = 0; i < nOutputs_; ++i)
  {
		std::vector<double> weightMatrix(nNeurons[nNeurons.size() - 1] + 1);
		for (int p = 0; p < weightMatrix.size(); ++p) weightMatrix[p] = randInitWeight();

    lastLayer.push_back(NeuronLogistic(weightMatrix));
  }
	layers_.push_back(lastLayer);

}

int Perceptron::startLearning(std::vector<Sample>& samples, 
															int nEpochs, 
															double learningSpeed, 
															double quality,
															std::vector<Sample>& test_samples,
															std::vector<std::vector<double>>& train_plot,
															std::vector<std::vector<double>>& test_plot)
{
  double error = 0.0;
  int rightAnsw = 0;
  int wrongAnsw = 0;

	int countEpoch = 0;
  for (int epochInd = 0; epochInd < nEpochs; ++epochInd)
  {
    rightAnsw = 0;
    wrongAnsw = 0;

    auto engine = std::default_random_engine{};
    std::shuffle(std::begin(samples), std::end(samples), engine);

		double sum_MSE = 0.0;
    for (int ind_sample = 0; ind_sample < samples.size(); ++ind_sample)
    {
			
			/*
			*********************************************************
			* FORWARD PROPAGATION
			*********************************************************
			*/

			std::vector<std::vector<double>> output_matrix;
			getOutputMatrix(output_matrix, samples[ind_sample].point);

			/*
			*********************************************************
			* BACKPROPAGATION 
			*********************************************************
			*/

			/* Calculate correct answer vector */
			std::vector<double> correct_output(nOutputs_);
			for (int i = 0; i < nOutputs_; ++i)
			{
				if (i == samples[ind_sample].classMark)
					correct_output[i] = 1.0;
				else
					correct_output[i] = 0.0;
			}

			std::vector<std::vector<double>> delta_matrix(layers_.size());

			/* Compute deltas for the last layer */
			delta_matrix[layers_.size() - 1].resize(nOutputs_);
			for (int ind_output = 0; ind_output < nOutputs_; ++ind_output)
			{
				double out = output_matrix[layers_.size() - 1][ind_output];
				double target = correct_output[ind_output];
				delta_matrix[layers_.size() - 1][ind_output] = -out * (1 - out) * (target - out);
			}

			for (int ind_layer = layers_.size()-2; ind_layer >= 0; ind_layer--)
			{
				delta_matrix[ind_layer].resize(layers_[ind_layer].size());
				for (int ind_neuron = 0; ind_neuron < layers_[ind_layer].size(); ++ind_neuron)
				{
					NeuronLogistic& neuron = layers_[ind_layer][ind_neuron];
					
					double sum = 0.0;
					for (int j = 0; j < layers_[ind_layer + 1].size(); ++j)
					{
						double delta_k = delta_matrix[ind_layer + 1][j];
						double w_jk = layers_[ind_layer + 1][j].getWeightMatrix()[ind_neuron];
						sum += delta_k * w_jk;
					}

					double out_j = output_matrix[ind_layer][ind_neuron];
					delta_matrix[ind_layer][ind_neuron] = out_j * (1 - out_j) * sum;
				}
			}

			/* Calculate new weights for each neuron */
			for (int ind_layer = 0; ind_layer < layers_.size(); ++ind_layer)
			{
				for (int ind_neuron = 0; ind_neuron < layers_[ind_layer].size(); ++ind_neuron)
				{
					NeuronLogistic& neuron = layers_[ind_layer][ind_neuron];
					int count = 0;
					for (double& w : neuron.getWeightMatrix())
					{
						double delta_w = 0.0;
						std::vector<double> input_vec = samples[ind_sample].point;
						input_vec.push_back(1.0); /* include bias */

						if (ind_layer == 0)
							delta_w = learningSpeed * delta_matrix[ind_layer][ind_neuron] * input_vec[count];
						else
						{
							if (count == neuron.getWeightMatrix().size() - 1)
								delta_w = learningSpeed * delta_matrix[ind_layer][ind_neuron] * 1.0; /* bias */
							else
								delta_w = learningSpeed * delta_matrix[ind_layer][ind_neuron] * output_matrix[ind_layer - 1][count];
						}							
						w -= delta_w;
						count++;
					}
				}
			}



			std::vector<double>::iterator result;
			result = std::max_element(output_matrix[layers_.size()-1].begin(), output_matrix[layers_.size() - 1].end());
			int net_answer = std::distance(output_matrix[layers_.size() - 1].begin(), result);

      if (net_answer == samples[ind_sample].classMark)
        rightAnsw++;
      else
        wrongAnsw++;

			/* Calculate train point coords */
			sum_MSE += MSE(output_matrix[output_matrix.size() - 1], correct_output);
			

      /*std::cout << "anws: " << net_answer << " correct: " << samples[ind_sample].classMark << std::endl;
      std::cout << "point class: " << samples[ind_sample].classMark << std::endl;
      for (int i = 0; i < output_matrix[layers_.size() - 1].size(); ++i)
        std::cout << output_matrix[layers_.size() - 1][i] << " ";
      std::cout << std::endl;
      for (int i = 0; i < correct_output.size(); ++i)
        std::cout << correct_output[i] << " ";

      std::cout << std::endl;
      std::cout << std::endl;*/
    }
		countEpoch++;

		

		error = wrongAnsw * 1.0 / samples.size();

		if (error < quality)
		{
			std::cout << "error is less than " << quality << "  epoch: " << epochInd << std::endl;
			return 0;
		}
		
		
		
		if (countEpoch == 1)
		{
			std::cout << "---------------------------------------------------------" << std::endl;
			std::cout << "TRAIN ERROR: " << std::setw(6) << std::setprecision(2) << std::fixed << error << " ,   F: " << wrongAnsw << " / " << samples.size() << std::endl;


			/* Calculate train point coords */
			std::vector<double> tr_p;
			tr_p.push_back(epochInd);
			tr_p.push_back(sum_MSE);
			train_plot.push_back(tr_p);

			/* Calculate train point coords */
			std::vector<double> test_p;
			test_p.push_back(epochInd);
			test_p.push_back(startTest(test_samples));
			test_plot.push_back(test_p);

			countEpoch = 0;
		}
  }
	return 0;
}

double Perceptron::startTest(std::vector<Sample>& samples)
{
	int wrongAnsw = 0;
	double sum_MSE = 0.0;
	for (int ind_sample = 0; ind_sample < samples.size(); ++ind_sample)
	{

		std::vector<std::vector<double>> output_matrix;
		getOutputMatrix(output_matrix, samples[ind_sample].point);

		/* Calculate correct answer vector */
		std::vector<double> correct_output(nOutputs_);
		for (int i = 0; i < nOutputs_; ++i)
		{
			if (i == samples[ind_sample].classMark)
				correct_output[i] = 1.0;
			else
				correct_output[i] = 0.0;
		}

		std::vector<double>::iterator result;
		result = std::max_element(output_matrix[layers_.size() - 1].begin(), output_matrix[layers_.size() - 1].end());
		int net_answer = std::distance(output_matrix[layers_.size() - 1].begin(), result);

		if (net_answer != samples[ind_sample].classMark)
			wrongAnsw++;

		sum_MSE += MSE(output_matrix[output_matrix.size() - 1], correct_output);
		/*std::cout << "anws: " << net_answer << " correct: " << samples[ind_sample].classMark << std::endl;
		std::cout << "point class: " << samples[ind_sample].classMark << std::endl;
		for (int i = 0; i < output_matrix[layers_.size() - 1].size(); ++i)
		std::cout << output_matrix[layers_.size() - 1][i] << " ";
		std::cout << std::endl;
		for (int i = 0; i < correct_output.size(); ++i)
		std::cout << correct_output[i] << " ";

		std::cout << std::endl;
		std::cout << std::endl;*/
	}

	double error = wrongAnsw * 1.0 / samples.size();
	std::cout << "TEST ERROR:  " << std::setw(6) << std::setprecision(2) << std::fixed << error << " ,   F: " << wrongAnsw << " / " << samples.size() << std::endl;

	return sum_MSE;
}










int Perceptron::getOutputMatrix(std::vector<std::vector<double>>& output_matrix, std::vector<double> input_point)
{
	output_matrix.resize(layers_.size());

	for (int ind_layer = 0; ind_layer < layers_.size(); ++ind_layer)
	{
		output_matrix[ind_layer].resize(layers_[ind_layer].size());
		for (int ind_neuron = 0; ind_neuron < layers_[ind_layer].size(); ++ind_neuron)
		{
			NeuronLogistic& neuron = layers_[ind_layer][ind_neuron];
			std::vector<double> input_vec;

			if (ind_layer == 0)
				input_vec = input_point;
			else
				input_vec = output_matrix[ind_layer - 1];

			/* include bias */
			input_vec.push_back(1.0);

			/* calculate neuron's output */
			output_matrix[ind_layer][ind_neuron] = neuron.process(input_vec);
		}
	}

	return 0;
}
int Perceptron::saveToTxt()
{

	return 0;
}
inline double Perceptron::MSE(std::vector<double> out, std::vector<double> target)
{
	double res = 0.0;

	for (int i = 0; i < out.size(); ++i)
	{
		res += (target[i] - out[i]) * (target[i] - out[i]);
	}

	return res * 0.5;
}


#endif PERCEPTRON_H// !PERCEPTRON_H

