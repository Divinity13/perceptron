#pragma once
#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>

#define HiddenLayer std::vector<NeuronLogistic>

/* Abstract class */
class Neuron
{
public:
  std::vector<double>& getWeightMatrix() { return weightMatrix_; };
  void                 setWeightMatrix(std::vector<double>& weightMatrix) { weightMatrix_ = weightMatrix; };

protected:
  virtual double ActivationFunc(const double x) = 0;
  virtual double ActivationFunc_Derivative(const double x) = 0;

  std::vector<double> weightMatrix_;
};


class NeuronLogistic : public Neuron
{
public:
  NeuronLogistic(std::vector<double> weightMatrix) { weightMatrix_ = weightMatrix; }
  ~NeuronLogistic() {}

  double process(const std::vector<double> input_vec);
private:
  /* Logistic func */
  double ActivationFunc(const double x) { return 1 / (1 + exp( -x ));  } 
  double ActivationFunc_Derivative(const double x) {
    return ActivationFunc(x) * (1 - ActivationFunc(x));
  }
};

double NeuronLogistic::process(const std::vector<double> input_vec)
{
  if (input_vec.size() != weightMatrix_.size())
  {
    std::cout << "input_vec.size() != weightMatrix_.size(): " << input_vec.size() << " : " << weightMatrix_.size() << std::endl;
    exit(-1);
  }

  double res(0);
  /* Summator */
  for (int i = 0; i < input_vec.size(); ++i)
  {
    res += input_vec[i] * weightMatrix_[i];
  }
  res = ActivationFunc(res);

  return res;
}


#endif // !NEURON_H