#include <stdio.h>
#include <array>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <math.h>
using namespace std;

float delta_x = 0.05;
double L = 4.0;
int N = (int)(L / delta_x) + 1;

void writeToFile(double *x, double *u, string filename)
{
  ofstream myfile;
  myfile.open(filename + ".dat");

  for (int i = 0; i < N; i++)
  {
    myfile << x[i] << " ";
  }
  myfile << endl;
  for (int i = 0; i < N; i++)
  {
    myfile << u[i] << " ";
  }
  myfile << endl;

  myfile.close();
}

double genRan()
{
  return ((double)rand() / (RAND_MAX));
}
float gaussian(float x, float mu, float sigma)
{
  return exp(-(pow(x - mu, 2.0)) / (2.0 * sigma * sigma)) / (pow(2.0 * M_PI * sigma * sigma, 0.5));
}

main(int argc, char const *argv[])
{
  int N = 1000;

  // init arrays

#pragma omp parallel
  {
    
    int thread_id = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    // cast to appropriate types
    
    double mu, sigma;
    N = 1000;
    mu = 0;
    sigma = 1;

    // array to store samples
    double *samples = new double[N];


    ofstream myfile;
    myfile.open("sample" + to_string(thread_id)+ ".dat");

    // initialize at mean
    samples[0] = mu;
    myfile << samples[0] << "\n";

    for (int i = 1; i < N; i++)
    {
      //using libraries
      // samples[i] = distribution(generator);

      double dx = (genRan() - 0.5) * sigma;

      double cand = samples[i - 1] + dx;

      double alpha = gaussian(cand, mu, sigma) / gaussian(samples[i - 1], mu, sigma);

      if (genRan() <= alpha)
      {
        samples[i] = cand;
      }
      else
      {
        samples[i] = samples[i - 1];
      }

      myfile << samples[i] << "\n";
    }
    myfile.close();
  }

  return 0;
}
