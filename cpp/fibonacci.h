/*
 * fibonacci.h
 *
 *  Created on: 18 aug 2014
 *      Author: 8nfl
 */

#ifndef FIBONACCI_H_
#define FIBONACCI_H_

#include <iostream>
#include <vector>
using namespace std;

unsigned int get_index_of_nearest_higher_Fibonacci_number(unsigned int x); // get index of nearest (same or higher) Fibonacci number
unsigned int get_index_of_nearest_lower_Fibonacci_number(unsigned int x);
unsigned int get_nearest_higher_Fibonacci_number(unsigned int x); // get nearest (same or higher) Fibonacci number
unsigned int get_nearest_lower_Fibonacci_number(unsigned int x);
unsigned int* get_Fibonacci_sequence_with_final_number(unsigned int final_number, unsigned int &final_index);

#endif /* FIBONACCI_H_ */
