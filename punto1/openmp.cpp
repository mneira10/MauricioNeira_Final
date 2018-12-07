#include <stdio.h>
#include <array>
#include <algorithm>
#include <iterator>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <omp.h>
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


main(int argc, char const *argv[])
{
  int N = 1000;

  // init arrays
  
  #pragma omp parallel
  {
    double x[N];
    int thread_id = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    printf("Hello from thread number: %d out of: %d\n",
           thread_id, thread_count);
    // writeToFile(x, u, "advecParallel");
  }

  

  return 0;
}
