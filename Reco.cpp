#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <cmath>
using namespace std;

//	+------------------+
//	| CREATE VARIABLES |
//	+------------------+
double** w1 = new double* [784]; double** w2 = new double* [256]; // inputs - layer1
double* inputs = new double[784]; double* l2 = new double[256];   // l2 - layer2
double* l3 = new double[10]; int rights; double a = -1, b = 1;  // l3 - layer3
double expected[10]; double* theta2; double* theta3; double LR = 0.001; double momentum = 0.9; double epsilon = 0.001; // try momentum = 0.7 or lower
double** delta1 = new double* [784]; double** delta2 = new double* [256];
double outputs[10];

//	+-----------------+
//	| ACTIVATION FUNC |
//	+-----------------+
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

//	+------------------+
//	| NORMALIZE OUTPUT |
//	+------------------+
void Softmax() {
	double exps[10];
	double sum_of_exp = 0;
	for (int i = 0; i < 10; i++) {
		exps[i] = exp(l3[i]);
		sum_of_exp += exps[i];
	}
	for (int i = 0; i < 10; i++) {
		outputs[i] = exps[i] / sum_of_exp;
	}
}

//	+----------------------+
//	| INITIALIZE VARIABLES |
//	+----------------------+
void init() {
	for (int i = 0; i < 784; i++) {
		w1[i] = new double[256];
		delta1[i] = new double[256];
	}
	for (int i = 0; i < 256; i++) {
		w2[i] = new double[10];
		delta2[i] = new double[10];
		l2[i] = 0;
	}
	for (int i = 0; i < 10; i++) {
		l3[i] = 0;
	}
	theta2 = new double[256];
	theta3 = new double[10];

//	+-------------------+
//	|RANDOM WEIGHTS INIT|
//  	+-------------------+
	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < 256; j++) {
			w1[i][j] = (double)(rand()) / RAND_MAX * (b - a) + a;
		}
	}
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 10; j++) {
			w2[i][j] = (double)(rand()) / RAND_MAX * (b - a) + a;
		}
	}

}
void FeedForward() {
	for (int i = 0; i < 256; i++) {
		l2[i] = 0;
	}
	for (int i = 0; i < 10; i++) {
		l3[i] = 0;
	}

	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < 256; j++) {
			l2[j] += w1[i][j] * inputs[i];
		}
	}
//	+---------------------+
//	| ACTIVATE L2 NEURONS |
//	+---------------------+
	for (int i = 0; i < 256; i++) {
		l2[i] = sigmoid(l2[i]);
	}

	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 10; j++) {
			l3[j] += w2[i][j] * l2[i];
		}
	}
//	+---------------------+
//	| ACTIVATE L3 NEURONS |
//	+---------------------+
	for (int i = 0; i < 10; i++) {
		l3[i] = sigmoid(l3[i]);
	}
}

//	+-----------------+
//	| FINDS THE ERROR |
//	+-----------------+
double Square_Err() {
	double res = 0;
	for (int i = 0; i < 10; i++) {
		res += (l3[i] - expected[i]) * (l3[i] - expected[i]);
	}
	res *= 0.5;
	return res;
}

//	+-----------------+
//	| LEARNING METHOD |
//	+-----------------+
void backprop() {
	double sum;
	for (int i = 0; i < 10; i++) {
		theta3[i] = l3[i] * (1 - l3[i]) * (expected[i] - l3[i]);
	}
	for (int i = 0; i < 256; i++) {
		sum = 0;
		for (int j = 0; j < 10; j++) {
			sum += w2[i][j] * theta3[j];
		}
		theta2[i] = l2[i] * (1 - l2[i]) * sum;
	}
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 10; j++) {
			delta2[i][j] = (LR * theta3[j] * l2[i]) + (momentum * delta2[i][j]);
			w2[i][j] += delta2[i][j];
		}
	}
	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < 256; j++) {
			delta1[i][j] = (LR * theta2[j] * inputs[i]) + (momentum * delta1[i][j]);
			w1[i][j] += delta1[i][j];
		}
	}
}

void Train() {
	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < 256; j++) {
			delta1[i][j] = 0;
		}
	}
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 10; j++) {
			delta2[i][j] = 0;
		}
	}
//	+--------------------+
//	| NUM OF EPOCHS: 256 |
//	+--------------------+
	for (int i = 0; i < 256; i++) {
		FeedForward();
		backprop();
		if (Square_Err() < epsilon) {
			continue;
		}
	}
}

void Write_Weights() {
	fstream fout;
	fout.open("Data\\weights2.txt", std::ios::out);
	fout.close();
	cout << "WEIGHTS FILE CLEARED BEFORE OVERWRITTEN" << endl;
	fout.open("weights2.txt");
	if (fout.is_open()) {
		for (int i = 0; i < 784; i++) {
			for (int j = 0; j < 256; j++) {
				fout << w1[i][j] << " ";
			}
		}
		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 10; j++) {
				fout << w2[i][j] << " ";
			}
		}
		cout << "+-------------------------+" << endl;
		cout << "| ALL WEIGHTS ARE WRITTEN |" << endl;
		cout << "+-------------------------+" << endl;
	}
	else {
		cout << "Some error with 'weights2.txt' file" << endl;
		cout << "Unable to write weights to file" << endl;
	}
	fout.close();
	
}
void Read_Weights() {
	fstream fin;
	fin.open("Data\\weights2.txt");
	if (fin.is_open()) {
		cout << "Weights file opened" << endl;
		for (int i = 0; i < 784; i++) {
			for (int j = 0; j < 256; j++) {
				fin >> w1[i][j];
			}
		}
		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 10; j++) {
				fin >> w2[i][j];
			}
		}
		cout << "Weights loaded" << endl;
	}
	else {
		cout << "Some error with 'weights2.txt' file" << endl;
		cout << "Unable to read weights from file" << endl;
	}
}

int main()
{
	// INIT -> TRAIN(FEEDFORWARD -> BACKPROP) 
	fstream file;
	string ff, path = "Data\\MNIST_train.txt";
	int choose, correct_ans = 0; double accuracy;
	cout << "What you want to do(1-Train; 2-Test): ";
	cin >> choose;
	switch (choose)
	{
	case 1:
		file.open(path);
		if (file.is_open()) {
			cout << "Train file opened." << endl;
			init();
			file >> ff;
			file >> ff;
			for (int i = 0; i < 60048; i++) {
				if (i % 1000 == 0) {
					system("cls");
				}
				for (int j = 0; j < 10; j++) {
					expected[j] = 0;
				}
				file >> rights;
				expected[rights] = 1;
				for (int j = 0; j < 784; j++) {
					file >> inputs[j];
				}
				Train();
				Softmax();
				cout << "Example: " << i << endl;
				cout << "Right: " << rights << endl;
				cout << "Error: " << Square_Err() << endl;
			}
		}
		else {
			cout << "Some error with 'MNIST_train.txt' file." << endl;
		}
		file.close();
		//	+-----------------------+
		//	| WRITE WEIGHTS TO FILE |
		//	+-----------------------+
		Write_Weights();
		break;
	case 2:
		init();
		Read_Weights();
		file.open("Data\\MNIST_test.txt");
		if (file.is_open()) {
			cout << "Test file Opened" << endl;
			file >> ff;
			file >> ff;
			cout << "Testing";
			for (int i = 0; i < 10000; i++) {
				if (i % 1000 == 0) {
					cout << ".";
				}
				file >> rights;
				for (int j = 0; j < 784; j++) {
					file >> inputs[j];
				}
				FeedForward();
				Softmax();
				double max = outputs[0];
				for (int j = 0; j < 10; j++) {
					if (outputs[j] > max) {
						max = outputs[j];
					}
				}
				for (int j = 0; j < 10; j++) {
					if (max == outputs[j]) {
						if (j == rights) {
							++correct_ans;
						}
					}
				}
			}
			cout << "\nTest completed. Results: " << endl;
			accuracy = double(correct_ans) / 10000 * 100;
			cout << "+------------------+" << endl;
			cout << "| Accuracy: " << accuracy << "% |" << endl;
			cout << "+------------------+" << endl;
			cout << "+-----------------------+" << endl;
			cout << "| Correct answers: " << correct_ans << " |" << endl;
			cout << "+-----------------------+" << endl;
			cout << "+---------------------------+" << endl;
			cout << "| Amount of examples: 10000 |" << endl;
			cout << "+---------------------------+" << endl;
		}
		else {
			cout << "Some error with 'MNIST_test.txt' file" << endl;
		}
		break;
	default:
		cout << "Wrong action" << endl;
		break;
	}
	
//	+------------------+
//	| DELETE ALL STUFF |
//	+------------------+
	delete[] inputs, l2, l3, theta2, theta3;
	for (int i = 0; i < 784; i++) {
		delete[] w1[i], delta1[i];
	}
	for (int i = 0; i < 256; i++) {
		delete[] w2[i], delta2[i];
	}
	delete[] w1, delta1, w2, delta2;
}
