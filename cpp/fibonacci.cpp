/*
 * fibonacci.cpp
 *
 *  Created on: 18 aug 2014
 *      Author: 8nfl
 */

#include "fibonacci.h"

// get index of nearest (same or higher) Fibonacci number
unsigned int get_index_of_nearest_higher_Fibonacci_number(unsigned int x) {
	if (x==0) return 1;
	unsigned int index = 1;
	unsigned int prev = 1; // Fibonacci number 0
	unsigned int cur = 1; // Fibonacci number 1
	unsigned int tmp;
	while(x>cur) {
		tmp = cur;
		cur = cur+prev;
		prev = tmp;
		++index;
	}
	return index;
}

unsigned int get_index_of_nearest_lower_Fibonacci_number(unsigned int x) {
	unsigned int index = 1; // Starting at index 1
	unsigned int prev = 1; // Fibonacci number 0
	unsigned int cur = 1; // Fibonacci number 1
	unsigned int tmp;
	while(x>cur) {
		tmp = cur;
		cur = cur+prev;
		prev = tmp;
		++index;
	}
	if (x<cur) return index-1;
	return index;
}

// get nearest (same or higher) Fibonacci number
unsigned int get_nearest_higher_Fibonacci_number(unsigned int x) {
	if (x==0) return 1;
	unsigned int prev = 1; // Fibonacci number 0
	unsigned int cur = 1; // Fibonacci number 1
	unsigned int tmp;
	while(x>cur) {
		tmp = cur;
		cur = cur+prev;
		prev = tmp;
	}
	return cur;
}

unsigned int get_nearest_lower_Fibonacci_number(unsigned int x) {
	if (x==0) {
		cout << "WARNING: all Fibonacci numbers are >0" << endl;
		return 0;
	}
	unsigned int prev = 1; // Fibonacci number 0
	unsigned int cur = 1; // Fibonacci number 1
	unsigned int tmp;
	while(x>cur) {
		tmp = cur;
		cur = cur+prev;
		prev = tmp;
	}
	if(x<cur) return prev;
	return cur;
}

unsigned int* get_Fibonacci_sequence_with_final_number(unsigned int final_number, unsigned int &final_index) {
	unsigned int* Fib = NULL;
	if (final_number == 0) {
		cout << "WARNING: " << final_number << " is not a Fibonacci number" << endl;
		return Fib;
	}
	vector<unsigned int> F;
	F.push_back(1);
	F.push_back(1);
	final_index = 1;
	while(final_number > F[final_index]) {
		F.push_back(F[final_index]+F[final_index-1]);
		final_index++;
	}
	if (final_number!=F[final_index]) {
		cout << "WARNING: " << final_number << " is not a Fibonacci number" << endl;
		final_index--;
	}
	Fib = new unsigned int[final_index+1];
	for (unsigned int n=0; n<=final_index; n++)
		Fib[n]=F[n];
	return Fib;
}
