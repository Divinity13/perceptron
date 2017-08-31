#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>

#include "perceptron.h"


//for files with point of one class
int loadPoints(char* filepath, int markClass, std::vector<Sample>& samples)
{
  std::ifstream fstr(filepath);
  if (!fstr) {
    std::cout << "Failed to open file " << filepath << std::endl;
    return -1;
  }

  std::string s;
  while (std::getline(fstr, s))
  {
    std::vector<double> p;

    /* split string */
    std::istringstream iss(s);
    while (iss)
    {
      double coord = 0;
      iss >> coord;
      p.push_back(coord);
    };

    p.pop_back(); // delete symbol of string's end
    Sample samp(p, markClass);
    samples.push_back(samp);
  }

  return 0;
}

// format: "2.943 4.5656 1.1233 0 
void loadTxt(char* filepath, std::vector<Sample>& samples)
{
	std::string line;
	std::ifstream myfile(filepath);

	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			std::stringstream iss(line);
			double number;
			std::vector<double> coords;
			while (iss >> number)
			{
				coords.push_back(number);

				if (iss.peek() == ' ')
					iss.ignore();
			}

			int class_mark = static_cast<int>(coords.back());
			coords.pop_back();

			Sample sample(coords, class_mark);
			samples.push_back(sample);
		}
		myfile.close();
	}
}

// format: "2.943,4.5656,1.1233,0 
void loadCSV(char* filepath, std::vector<Sample>& samples)
{
	std::ifstream in(filepath);
  std::vector<double> coords;
	if (in) {
		std::string line;
		while (std::getline(in, line))
		{
			std::stringstream sep(line);
			std::string coord;
			while (std::getline(sep, coord, ',')) 
			{
				try {
					coords.push_back(stod(coord));
				}
				catch (const char* msg)
				{}
			}
			int class_mark = static_cast<int>(coords.back() - 1);
			coords.pop_back();
			Sample sample(coords, class_mark);
			samples.push_back(sample);
		}
	}
	in.close();
}

void writePointsToTXT(char* filename, std::vector<std::vector<double>> vec)
{
	std::ofstream out(filename);
	if (!out) {
		std::cout << "Cannot open file.\n";
		exit(1);
	}
	for (int i = 0; i < vec.size(); ++i)
	{
		out << vec[i][0] << " " << vec[i][1] << "\n";
	}
	out.close();
}

int main()
{
	int train_percent = 80;

	std::vector<Sample> samples;

	loadTxt("C:/Users//Desktop/generated_data_non-separable.txt", samples);

	// normalization (if we need this)
	for (Sample& sampl : samples)
	{
		for (double& coord : sampl.point)
			//std::cout << coord << " ";
			coord /= 100;
		//std::cout << sampl.classMark << std::endl;
	}
	

	auto engine = std::default_random_engine{};
	std::shuffle(std::begin(samples), std::end(samples), engine);

	std::vector<Sample> trainSet, testSet;
	int ind_border = train_percent/100.0 * samples.size();
	for (int i = 0; i < ind_border; ++i)							trainSet.push_back(samples[i]);
	for (int i = ind_border; i < samples.size(); ++i) testSet.push_back(samples[i]);

	std::cout.width(2);
	std::cout << "TOTAL NUMBER OF SAMPLES:      " << samples.size() << " (100%)" << std::endl;
	std::cout << "Number of samples for TRAIN:  " << trainSet.size() << " (" << train_percent << "%)" << std::endl;
	std::cout << "Number of samples for TEST:   " << testSet.size() << " (" << 100 - train_percent << "%)" << std::endl;

	std::vector<std::vector<double>> test_plot;
	std::vector<std::vector<double>> train_plot;


  std::vector<int> nNeurons;
  nNeurons.push_back(15);
	nNeurons.push_back(15);
  /* Perceptron(int nInputs, int nOutputs, int nLayers, std::vector<int>& nNeurons); */
  Perceptron nn(15, 14, 2, nNeurons);
	// startLearning(std::vector<Sample>& samples, int nEpochs, double learningSpeed, double quality);
  nn.startLearning(trainSet, 300, 0.36, 0.00, testSet, train_plot, test_plot);

	std::cout << "\ntest begins..." << std::endl;
	nn.startTest(testSet);
	
	writePointsToTXT("train.txt", train_plot);
	writePointsToTXT("test.txt", test_plot);


	std::cout << "Done!" << std::endl;
  return 0;
}